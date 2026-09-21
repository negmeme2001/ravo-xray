from __future__ import annotations

import hashlib
import math
import subprocess
from collections.abc import Sequence
from pathlib import Path


def find_clip(clips_dir: Path | str, clip_id: str) -> Path:
    directory = Path(clips_dir)
    matches = sorted(path for path in directory.iterdir() if path.is_file() and path.stem == clip_id)
    if not matches:
        raise FileNotFoundError(f"no clip with id {clip_id!r} found in {directory}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"multiple clips with id {clip_id!r} found in {directory}: {names}")
    return matches[0]


def load_pcm16(path: Path | str, sample_rate: int = 24_000) -> bytes:
    source = Path(path)
    command = [
        "/usr/bin/ffmpeg",
        "-v",
        "error",
        "-i",
        str(source),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"could not run ffmpeg for {source}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {source}: {detail or 'unknown error'}")
    return completed.stdout


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration_s(pcm: bytes, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return len(pcm) / (sample_rate * 2)


# 10 ms of mono 16-bit audio at 24 kHz, the frame size both runners stream.
FRAME_BYTES_24K_10MS = 480
# A frame louder than this (RMS of int16 samples) counts as speech.
VOICE_RMS_THRESHOLD = 500


def pcm_rms(data: bytes | memoryview | Sequence[int]) -> float:
    if isinstance(data, (bytes, bytearray, memoryview)):
        samples = memoryview(data).cast("h")
    else:
        samples = data
    if not samples:
        return 0.0
    return math.sqrt(sum(int(sample) ** 2 for sample in samples) / len(samples))



def speech_span(pcm: bytes, frame_bytes: int = FRAME_BYTES_24K_10MS) -> tuple[int, int]:
    """Indexes of the first and last voiced 10 ms frames of a clip.

    Phone recordings carry silence before and after the words; timing the agent against the
    file edges would count a correct reply during that trailing silence as a cut-in. Pauses
    inside the speech stay inside the span, which is what the pause scenarios test.
    """
    voiced = [
        index
        for index, offset in enumerate(range(0, len(pcm), frame_bytes))
        if pcm_rms(pcm[offset : offset + frame_bytes - (frame_bytes % 2)]) > VOICE_RMS_THRESHOLD
    ]
    if not voiced:
        last = max(0, (len(pcm) - 1) // frame_bytes)
        return 0, last
    return voiced[0], voiced[-1]

