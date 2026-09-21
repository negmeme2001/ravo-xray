from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from voicetest.audio import duration_s, find_clip, load_pcm16, sha256_file

SAMPLE_RATE = 24_000


def update_manifest(clips_dir: Path, scenarios_path: Path) -> tuple[list[str], list[str]]:
    document: dict[str, Any] = json.loads(scenarios_path.read_text(encoding="utf-8"))
    updated: list[str] = []
    missing: list[str] = []
    for scenario in document["scenarios"]:
        clip_id = str(scenario["clip"])
        try:
            clip_path = find_clip(clips_dir, clip_id)
        except FileNotFoundError:
            missing.append(clip_id)
            continue
        pcm = load_pcm16(clip_path, SAMPLE_RATE)
        length = round(duration_s(pcm, SAMPLE_RATE), 2)
        scenario["audio"] = {
            "file": clip_path.name,
            "sha256": sha256_file(clip_path),
            "duration_s": length,
            "sample_rate": SAMPLE_RATE,
        }
        updated.append(f"{clip_id}: {clip_path.name} ({length:.2f}s)")
    scenarios_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return updated, missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add recorded audio metadata to voice scenarios.")
    parser.add_argument("--clips", type=Path, required=True, help="directory containing recorded clips")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("voicetest/scenarios.json"),
        help="scenario manifest (default: voicetest/scenarios.json)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    updated, missing = update_manifest(args.clips, args.scenarios)
    for summary in updated:
        print(summary)
    if missing:
        print(f"Missing clips: {', '.join(missing)}")
    print(f"Updated {len(updated)} clip(s); missing {len(missing)}.")


if __name__ == "__main__":
    main()
