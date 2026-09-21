from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ravo_core import assistant, tools
from voicetest.outcome import score_turn, summarize_runs
from voicetest.scoring import arabic_ratio

REQUIRED_ENV = ("OPENAI_API_KEY", "DATABASE_URL")


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def run_scenarios(
    scenarios: Sequence[dict[str, Any]],
    reps: int,
    run_turn: Callable[[list[dict[str, str]]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    queries: list[str] = []
    original_products = tools.search_products
    original_documents = tools.search_documents

    def record_products(query: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        queries.append(query)
        return original_products(query, *args, **kwargs)

    def record_documents(query: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        queries.append(query)
        return original_documents(query, *args, **kwargs)

    tools.search_products = record_products
    tools.search_documents = record_documents
    rows: list[dict[str, Any]] = []
    try:
        turn = run_turn or assistant.run_turn
        for scenario in scenarios:
            for rep in range(1, reps + 1):
                queries.clear()
                base = {"scenario": scenario["id"], "rep": rep}
                try:
                    result = turn([{"role": "user", "content": scenario["reference"]}])
                    scores = score_turn(
                        scenario,
                        user_transcript=scenario["reference"],
                        reply=result["reply"],
                        cards=result.get("cards", []),
                        trace_steps=result.get("trace", {}).get("steps", []),
                        cut_in=None,
                        eot_delay_s=None,
                    )
                    rows.append(
                        {
                            **base,
                            **scores,
                            "english_queries": (
                                all(arabic_ratio(query) == 0 for query in queries) if queries else None
                            ),
                        }
                    )
                except Exception as exc:
                    rows.append({**base, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        tools.search_products = original_products
        tools.search_documents = original_documents
    return rows


def _load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": {}}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document.get("runs"), dict):
        raise ValueError(f"{path} does not contain a runs object")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run voice-test scenarios through the typed assistant.")
    parser.add_argument("--label", required=True, help="name for this result set")
    parser.add_argument("--reps", type=int, default=1, help="repetitions per scenario (default: 1)")
    parser.add_argument("--scenarios", type=Path, default=Path("voicetest/scenarios.json"))
    parser.add_argument("--out", type=Path, default=Path("voicetest/typed_results.json"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        parser.error(f"required environment variables are unset: {', '.join(missing)}")
    if args.reps < 1:
        parser.error("--reps must be at least 1")
    manifest = json.loads(args.scenarios.read_text(encoding="utf-8"))
    rows = run_scenarios(manifest["scenarios"], args.reps)
    report = _load_report(args.out)
    report["runs"][args.label] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "agent": "typed",
        "reps": args.reps,
        "clips": {},
        "rows": rows,
        "summary": summarize_runs(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} scored turn(s) to {args.out}")


if __name__ == "__main__":
    main()
