from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import statistics
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
from livekit import rtc
from livekit.agents import inference, stt

from voicetest.audio import VOICE_RMS_THRESHOLD, duration_s, find_clip, load_pcm16, pcm_rms, sha256_file
from voicetest.scoring import cer, keyword_recall, text_language

SAMPLE_RATE = 24_000
FRAME_MS = 10
FRAME_BYTES = SAMPLE_RATE * 2 * FRAME_MS // 1_000
# After the clip, keep streaming silence the way a live microphone does while the agent
# waits for the turn to end (up to 3 s in the bilingual profile); a final transcript that
# has not arrived by then would not have reached the agent in time either.
SILENCE_FRAMES = 4_000 // FRAME_MS
REQUIRED_ENV = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model: str
    language: str | None


def parse_model_spec(value: str) -> ModelSpec:
    model, separator, language = value.rpartition("@")
    if not separator:
        model = value
        language = ""
    if "/" not in model or not all(model.split("/", maxsplit=1)):
        raise ValueError(f"invalid model spec {value!r}; expected provider/model[@language]")
    if separator and not language:
        raise ValueError(f"invalid model spec {value!r}; language after @ cannot be empty")
    return ModelSpec(label=value, model=model, language=language or None)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(results: list[dict[str, Any]], models: list[str]) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for model in models:
        scored = [row for row in results if row["model"] == model and "error" not in row]
        by_language: dict[str, list[float]] = defaultdict(list)
        for row in scored:
            recall = row["keyword_recall"]
            if recall is not None:
                by_language[row["scenario_language"]].append(recall)
        latencies = [row["final_latency_s"] for row in scored if row["final_latency_s"] is not None]
        summary[model] = {
            "keyword_recall_en": _mean_or_none(by_language["en"]),
            "keyword_recall_ar": _mean_or_none(by_language["ar"]),
            "keyword_recall_mixed": _mean_or_none(by_language["mixed"]),
            "mean_cer": _mean_or_none([row["cer"] for row in scored]),
            "empty_rate": _mean_or_none([float(row["empty"]) for row in scored]),
            "p50_final_latency_s": percentile(latencies, 0.50),
            "p95_final_latency_s": percentile(latencies, 0.95),
        }
    return summary


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


async def _push_audio(stream: Any, pcm: bytes, timing: dict[str, float]) -> None:
    loop = asyncio.get_running_loop()
    next_push = loop.time()
    for offset in range(0, len(pcm), FRAME_BYTES):
        chunk = pcm[offset : offset + FRAME_BYTES]
        if len(chunk) % 2:
            chunk += b"\x00"
        stream.push_frame(rtc.AudioFrame(chunk, SAMPLE_RATE, 1, len(chunk) // 2))
        # Latency counts from the last spoken frame, not the recording's trailing silence.
        if pcm_rms(chunk) > VOICE_RMS_THRESHOLD:
            timing["last_speech"] = time.perf_counter()
        next_push += len(chunk) / 2 / SAMPLE_RATE
        await asyncio.sleep(max(0.0, next_push - loop.time()))
    silence = b"\x00" * FRAME_BYTES
    for _ in range(SILENCE_FRAMES):
        stream.push_frame(rtc.AudioFrame(silence, SAMPLE_RATE, 1, FRAME_BYTES // 2))
        next_push += FRAME_MS / 1_000
        await asyncio.sleep(max(0.0, next_push - loop.time()))


async def _collect_events(
    stream: Any, finals: list[str], detected: dict[str, str | None], timing: dict[str, float]
) -> None:
    async for event in stream:
        if event.type != stt.SpeechEventType.FINAL_TRANSCRIPT or not event.alternatives:
            continue
        alternative = event.alternatives[0]
        if alternative.text:
            finals.append(alternative.text.strip())
        detected["language"] = alternative.language or None
        timing["last_final"] = time.perf_counter()


async def transcribe_once(client: inference.STT, pcm: bytes) -> tuple[str, str | None, float | None]:
    finals: list[str] = []
    detected: dict[str, str | None] = {"language": None}
    timing: dict[str, float] = {}
    # The gateway keeps the stream open after the audio ends, so stop listening once the
    # audio and its trailing silence have been sent rather than waiting for it to close.
    async with client.stream() as stream:
        collector = asyncio.create_task(_collect_events(stream, finals, detected, timing))
        try:
            await _push_audio(stream, pcm, timing)
        finally:
            collector.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await collector
    latency = None
    if "last_speech" in timing and "last_final" in timing:
        latency = timing["last_final"] - timing["last_speech"]
    return " ".join(finals), detected["language"], latency


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(args.scenarios.read_text(encoding="utf-8"))
    model_specs = [parse_model_spec(item.strip()) for item in args.models.split(",") if item.strip()]
    if not model_specs:
        raise ValueError("at least one model is required")

    clips: dict[str, str] = {}
    available: list[tuple[dict[str, Any], bytes]] = []
    for scenario in manifest["scenarios"]:
        try:
            clip_path = find_clip(args.clips, str(scenario["clip"]))
        except FileNotFoundError:
            print(f"Skipping missing clip {scenario['clip']}")
            continue
        clips[str(scenario["id"])] = sha256_file(clip_path)
        available.append((scenario, load_pcm16(clip_path, SAMPLE_RATE)))

    results: list[dict[str, Any]] = []
    async with aiohttp.ClientSession() as session:
        for scenario, pcm in available:
            for spec in model_specs:
                for rep in range(1, args.reps + 1):
                    base = {"scenario": scenario["id"], "model": spec.label, "rep": rep}
                    try:
                        client_options: dict[str, Any] = {"model": spec.model, "http_session": session}
                        if spec.language is not None:
                            client_options["language"] = spec.language
                        client = inference.STT(**client_options)
                        timeout = duration_s(pcm, SAMPLE_RATE) + 10
                        transcript, detected, latency = await asyncio.wait_for(
                            transcribe_once(client, pcm), timeout=timeout
                        )
                        row = {
                            **base,
                            "scenario_language": scenario["language"],
                            "keyword_recall": keyword_recall(transcript, scenario["keywords"]),
                            "cer": cer(scenario["reference"], transcript),
                            "empty": not bool(transcript),
                            "final_latency_s": latency,
                            "detected_language": detected,
                            "transcript_language": text_language(transcript),
                            "timed_out": False,
                        }
                        results.append(row)
                        if args.show_transcripts:
                            print(f"{scenario['id']} {spec.label} rep {rep}: {transcript!r}")
                    except TimeoutError as exc:
                        results.append({**base, "timed_out": True, "error": f"{type(exc).__name__}: {exc}"})
                    except Exception as exc:
                        results.append({**base, "timed_out": False, "error": f"{type(exc).__name__}: {exc}"})

    labels = [spec.label for spec in model_specs]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "models": labels,
        "reps": args.reps,
        "clips": clips,
        "results": results,
        "summary": summarize(results, labels),
    }


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def print_summary(summary: dict[str, dict[str, float | None]]) -> None:
    print("model\tkw-en\tkw-ar\tkw-mixed\tCER\tempty\tp50\tp95")
    for model, scores in summary.items():
        values = [
            scores["keyword_recall_en"], scores["keyword_recall_ar"],
            scores["keyword_recall_mixed"], scores["mean_cer"], scores["empty_rate"],
            scores["p50_final_latency_s"], scores["p95_final_latency_s"],
        ]
        rendered = ["-" if value is None else f"{value:.3f}" for value in values]
        print("\t".join([model, *rendered]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark recorded clips with LiveKit Inference STT.")
    parser.add_argument("--clips", type=Path, required=True, help="directory containing recorded clips")
    parser.add_argument("--models", required=True, help="comma-separated provider/model[@language] specs")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per clip and model (default: 3)")
    parser.add_argument("--scenarios", type=Path, default=Path("voicetest/scenarios.json"))
    parser.add_argument("--out", type=Path, default=Path("voicetest/stt_results.json"))
    parser.add_argument("--show-transcripts", action="store_true", help="print transcript text to stdout")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        parser.error(f"required LiveKit environment variables are unset: {', '.join(missing)}")
    if args.reps < 1:
        parser.error("--reps must be at least 1")
    try:
        report = asyncio.run(run_benchmark(args))
    except ValueError as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(report["summary"])
    print(f"Wrote scores to {args.out}")


if __name__ == "__main__":
    main()
