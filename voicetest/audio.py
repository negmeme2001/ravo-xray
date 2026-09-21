from __future__ import annotations

import hashlib
import subprocess
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
