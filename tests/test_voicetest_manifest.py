import json
import shutil
import sys
import wave
from pathlib import Path

import pytest

from voicetest import manifest


pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


def write_wav(path: Path, duration_s: float = 0.1, sample_rate: int = 8_000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * round(duration_s * sample_rate))


def test_manifest_adds_audio_and_reports_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    clip = clips / "01.wav"
    write_wav(clip)
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text(
        json.dumps(
            {
                "version": 1,
                "scenarios": [
                    {"id": "01", "clip": "01", "reference": "hello"},
                    {"id": "02", "clip": "02", "reference": "missing"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["manifest", "--clips", str(clips), "--scenarios", str(scenarios)])

    manifest.main()

    output = capsys.readouterr().out
    assert "01: 01.wav" in output
    assert "Missing clips: 02" in output
    document = json.loads(scenarios.read_text(encoding="utf-8"))
    audio = document["scenarios"][0]["audio"]
    assert audio["file"] == "01.wav"
    assert len(audio["sha256"]) == 64
    assert audio["duration_s"] == 0.1
    assert audio["sample_rate"] == 24_000
    assert "audio" not in document["scenarios"][1]
