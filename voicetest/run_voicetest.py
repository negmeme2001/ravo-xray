from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from livekit import api, rtc

from ravo_core import monitor
from voicetest.audio import VOICE_RMS_THRESHOLD, find_clip, load_pcm16, pcm_rms, sha256_file, speech_span
from voicetest.outcome import score_turn, summarize_runs

SAMPLE_RATE = 24_000
FRAME_MS = 10
FRAME_S = FRAME_MS / 1_000
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1_000
FRAME_BYTES = FRAME_SAMPLES * 2
GREETING_TIMEOUT_S = 45
AGENT_QUIET_S = 1.0
TURN_TIMEOUT_S = 25
POST_RESULT_S = 3.0
REQUIRED_ENV = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")


def voice_timing(
    voiced_at: Sequence[float], clip_started_at: float, clip_ended_at: float
) -> tuple[bool, float | None]:
    cut_in = any(clip_started_at <= timestamp <= clip_ended_at for timestamp in voiced_at)
    after = [timestamp for timestamp in voiced_at if timestamp > clip_ended_at]
    return cut_in, min(after) - clip_ended_at if after else None


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _token(room_name: str, identity: str, agent_name: str) -> str:
    configuration = api.RoomConfiguration(name=room_name)
    configuration.agents.append(api.RoomAgentDispatch(agent_name=agent_name))
    return (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_name("Shopper")
        .with_ttl(timedelta(minutes=10))
        .with_room_config(configuration)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )


async def _capture_agent_audio(track: rtc.Track, voiced_at: list[float]) -> None:
    async for event in rtc.AudioStream(
        track, sample_rate=SAMPLE_RATE, num_channels=1, frame_size_ms=FRAME_MS
    ):
        if pcm_rms(event.frame.data) > VOICE_RMS_THRESHOLD:
            voiced_at.append(time.perf_counter())


async def _wait_until_quiet(voiced_at: list[float]) -> None:
    while not voiced_at or time.perf_counter() - voiced_at[-1] < AGENT_QUIET_S:
        await asyncio.sleep(0.05)


FrameItem = tuple[bytes, asyncio.Future[float]] | None


async def _feed_microphone(source: rtc.AudioSource, frames: asyncio.Queue[FrameItem]) -> None:
    silence = bytes(FRAME_BYTES)
    while True:
        try:
            item = frames.get_nowait()
        except asyncio.QueueEmpty:
            item = (silence, asyncio.get_running_loop().create_future())
        if item is None:
            return
        chunk, captured_at = item
        timestamp = time.perf_counter()
        if not captured_at.done():
            captured_at.set_result(timestamp)
        await source.capture_frame(rtc.AudioFrame(chunk, SAMPLE_RATE, 1, len(chunk) // 2))


async def _stream_clip(frames: asyncio.Queue[FrameItem], pcm: bytes) -> tuple[float, float]:
    """Stream the clip; return when its speech started and ended (not the file edges)."""
    chunks = [pcm[offset : offset + FRAME_BYTES] for offset in range(0, len(pcm), FRAME_BYTES)]
    if not chunks:
        chunks = [bytes(FRAME_BYTES)]
    first_voiced, last_voiced = speech_span(pcm)
    timestamps: list[asyncio.Future[float]] = []
    for chunk in chunks:
        if len(chunk) < FRAME_BYTES:
            chunk += bytes(FRAME_BYTES - len(chunk))
        captured_at = asyncio.get_running_loop().create_future()
        timestamps.append(captured_at)
        await frames.put((chunk, captured_at))
    return await timestamps[first_voiced], await timestamps[min(last_voiced, len(timestamps) - 1)]


async def run_session(
    scenario: dict[str, Any], pcm: bytes, agent_name: str, show_transcripts: bool = False,
    monitor_ids: list[str] | None = None,
) -> dict[str, Any]:
    room_name = f"vt-{uuid.uuid4().hex[:12]}"
    identity = f"tester-{uuid.uuid4().hex[:8]}"
    room = rtc.Room()
    greeting = asyncio.Event()
    voiced_at: list[float] = []
    user_transcripts: list[str] = []
    replies: list[str] = []
    cards: list[dict[str, Any]] = []
    trace_steps: list[dict[str, Any]] = []
    clip_started_at: float | None = None
    result_updated_at: float | None = None
    audio_tasks: set[asyncio.Task[None]] = set()

    async def transcription_handler(reader: Any, participant_identity: str) -> None:
        nonlocal result_updated_at
        text = (await reader.read_all()).strip()
        if participant_identity != identity and text:
            if not greeting.is_set():
                greeting.set()
            elif clip_started_at is not None:
                replies.append(text)
                result_updated_at = time.perf_counter()
        elif (
            participant_identity == identity
            and clip_started_at is not None
            and reader.info.attributes.get("lk.transcription_final") == "true"
            and text
        ):
            user_transcripts.append(text)

    async def cards_handler(reader: Any, _participant_identity: str) -> None:
        nonlocal result_updated_at
        payload = json.loads(await reader.read_all())
        if clip_started_at is not None:
            cards.extend(payload.get("cards", []))
            result_updated_at = time.perf_counter()

    async def trace_handler(reader: Any, _participant_identity: str) -> None:
        nonlocal trace_steps, result_updated_at
        payload = json.loads(await reader.read_all())
        if monitor_ids is not None and payload.get("monitor_id"):
            monitor_ids.append(payload["monitor_id"])
        if clip_started_at is not None:
            trace_steps = payload.get("steps", [])
            result_updated_at = time.perf_counter()

    stream_tasks: set[asyncio.Task[None]] = set()

    def in_task(handler: Any) -> Any:
        # The room calls stream handlers synchronously; an async one would never be awaited.
        def start(reader: Any, participant_identity: str) -> None:
            task = asyncio.create_task(handler(reader, participant_identity))
            stream_tasks.add(task)
            task.add_done_callback(stream_tasks.discard)

        return start

    room.register_text_stream_handler("lk.transcription", in_task(transcription_handler))
    room.register_text_stream_handler("ravo.cards", in_task(cards_handler))
    room.register_text_stream_handler("ravo.trace", in_task(trace_handler))

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, _publication: Any, _participant: Any) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            task = asyncio.create_task(_capture_agent_audio(track, voiced_at))
            audio_tasks.add(task)
            task.add_done_callback(audio_tasks.discard)

    frames: asyncio.Queue[FrameItem] = asyncio.Queue()
    feeder: asyncio.Task[None] | None = None
    try:
        await room.connect(os.environ["LIVEKIT_URL"], _token(room_name, identity, agent_name))
        source = rtc.AudioSource(SAMPLE_RATE, 1)
        track = rtc.LocalAudioTrack.create_audio_track("mic", source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        feeder = asyncio.create_task(_feed_microphone(source, frames))
        await asyncio.wait_for(greeting.wait(), timeout=GREETING_TIMEOUT_S)
        await asyncio.wait_for(_wait_until_quiet(voiced_at), timeout=GREETING_TIMEOUT_S)
        clip_started_at, clip_ended_at = await _stream_clip(frames, pcm)

        deadline = clip_ended_at + TURN_TIMEOUT_S
        while time.perf_counter() < deadline:
            if replies and trace_steps and result_updated_at is not None:
                if time.perf_counter() - result_updated_at >= POST_RESULT_S:
                    break
            await asyncio.sleep(0.05)

        cut_in, eot_delay = voice_timing(voiced_at, clip_started_at, clip_ended_at)
        transcript = " ".join(user_transcripts)
        reply = " ".join(replies)
        if show_transcripts:
            print(f"{scenario['id']}: transcript={transcript!r} reply={reply!r}")
        return score_turn(
            scenario,
            user_transcript=transcript,
            reply=reply,
            cards=cards,
            trace_steps=trace_steps,
            cut_in=cut_in,
            eot_delay_s=eot_delay,
        )
    finally:
        if feeder is not None:
            await frames.put(None)
            await feeder
        for task in (*audio_tasks, *stream_tasks):
            task.cancel()
        await room.disconnect()


async def run_all(
    scenarios: Sequence[dict[str, Any]], clips_dir: Path, agent_name: str, reps: int,
    show_transcripts: bool, monitor_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    clips: dict[str, str] = {}
    loaded: list[tuple[dict[str, Any], bytes]] = []
    for scenario in scenarios:
        clip_path = find_clip(clips_dir, str(scenario["clip"]))
        clips[str(scenario["id"])] = sha256_file(clip_path)
        loaded.append((scenario, load_pcm16(clip_path, SAMPLE_RATE)))
    for scenario, pcm in loaded:
        for rep in range(1, reps + 1):
            base = {"scenario": scenario["id"], "rep": rep}
            try:
                rows.append({
                    **base,
                    **await run_session(scenario, pcm, agent_name, show_transcripts, monitor_ids),
                })
            except Exception as exc:
                rows.append({**base, "error": f"{type(exc).__name__}: {exc}"})
    return rows, clips


def _forget_test_answers(monitor_ids: list[str]) -> None:
    """Take this run's answers back out of the live monitor (an older agent records them)."""
    if not monitor_ids:
        return
    removed = monitor.forget(monitor_ids)
    if removed < len(monitor_ids):
        print(
            f"Warning: {len(monitor_ids) - removed} of this run's answers are still in the live "
            "monitor; set KV_REST_API_URL and KV_REST_API_TOKEN to remove them"
        )
    else:
        print(f"Removed this run's {removed} answers from the live monitor")


def _print_summary(summary: dict[str, Any]) -> None:
    print("scenario\theard\tno-cut-in\tlanguage\tclarified\tanswered\tp50\tp95")
    for scenario, scores in summary["scenarios"].items():
        latency = [scores["p50_eot_delay_s"], scores["p95_eot_delay_s"]]
        rendered = ["-" if value is None else f"{value:.2f}" for value in latency]
        print("\t".join([scenario, scores["heard"], scores["no_cut_in"],
                         scores["right_language"], scores["clarified"],
                         scores["answered_right"], *rendered]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score recorded clips against a live voice agent.")
    parser.add_argument("--clips", type=Path, required=True, help="directory containing clips")
    parser.add_argument("--agent", required=True, help="LiveKit dispatched agent name")
    parser.add_argument("--reps", type=int, required=True, help="repetitions per scenario")
    parser.add_argument("--max-sessions", type=int, required=True, help="hard session cost cap")
    parser.add_argument("--label", required=True, help="name for this result set")
    parser.add_argument("--only", help="comma-separated scenario IDs")
    parser.add_argument("--scenarios", type=Path, default=Path("voicetest/scenarios.json"))
    parser.add_argument("--out", type=Path, default=Path("voicetest/results.json"))
    parser.add_argument("--show-transcripts", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        parser.error(f"required LiveKit environment variables are unset: {', '.join(missing)}")
    if args.reps < 1 or args.max_sessions < 1:
        parser.error("--reps and --max-sessions must be at least 1")
    manifest = json.loads(args.scenarios.read_text(encoding="utf-8"))
    scenarios = manifest["scenarios"]
    if args.only:
        selected = {item.strip() for item in args.only.split(",") if item.strip()}
        scenarios = [scenario for scenario in scenarios if str(scenario["id"]) in selected]
        missing_ids = selected - {str(scenario["id"]) for scenario in scenarios}
        if missing_ids:
            parser.error(f"unknown scenario IDs: {', '.join(sorted(missing_ids))}")
    sessions = len(scenarios) * args.reps
    if sessions > args.max_sessions:
        parser.error(f"refusing {sessions} sessions: --max-sessions is {args.max_sessions}")
    monitor_ids: list[str] = []
    try:
        rows, clips = asyncio.run(
            run_all(scenarios, args.clips, args.agent, args.reps, args.show_transcripts, monitor_ids)
        )
        report = (
            json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {"runs": {}}
        )
        if not isinstance(report.get("runs"), dict):
            raise ValueError(f"{args.out} does not contain a runs object")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    finally:
        _forget_test_answers(monitor_ids)
    summary = summarize_runs(rows)
    report["runs"][args.label] = {
        "generated_at": datetime.now(UTC).isoformat(), "git_commit": git_commit(),
        "agent": args.agent, "reps": args.reps, "clips": clips, "rows": rows,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_summary(summary)
    print(f"Wrote scores to {args.out}")


if __name__ == "__main__":
    main()
