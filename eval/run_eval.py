"""Retrieval evaluation runner: scores search configs and writes eval/results.json.

Usage:
    uv run python -m eval.run_eval
    uv run python -m eval.run_eval --skip-ingest
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval import metrics
from ravo_core import tools
from ravo_core.db import connect
from seed import ingest_pdfs, ingest_products
from seed.generate_catalog import BRANDS

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
DOCS_DIR = Path(__file__).parent.parent / "seed" / "docs"

PRODUCT_CONFIGS = [
    ("name", "Product name only"),
    ("full", "Name, category and description"),
]
DOCUMENT_CONFIGS = [
    (100, 20),
    (200, 40),
    (500, 40),
]


def require_env() -> None:
    missing = [v for v in ("DATABASE_URL", "OPENAI_API_KEY") if not os.environ.get(v)]
    if missing:
        sys.exit(f"Missing required environment variable(s): {', '.join(missing)}")


def pdf_paths() -> list[str]:
    return sorted(str(p) for p in DOCS_DIR.glob("*.pdf"))


def check_product_labels(cases: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop cases whose relevant_names never appear in the products table.

    Returns (valid_cases, invalid_label_queries).
    """
    conn = connect()
    try:
        invalid = []
        for case in cases:
            if not case.get("relevant_names"):
                continue
            count = conn.execute(
                "SELECT count(*) FROM products WHERE name = ANY(%s)",
                (case["relevant_names"],),
            ).fetchone()[0]
            if count == 0:
                invalid.append(case["query"])
    finally:
        conn.close()
    valid = [c for c in cases if c["query"] not in invalid]
    return valid, invalid


def evaluate_products(cases: list[dict], brands: list[str]) -> tuple[dict, dict, list[dict]]:
    hit_at_1, hit_at_5, rrs, diversities, search_ms = [], [], [], [], []
    by_tag = {}
    failures = []
    for case in cases:
        result = tools.search_products(case["query"], k=5)
        names = [item["name"] for item in result["items"]]
        flags = [metrics.is_relevant_product(n, case, brands) for n in names]
        hit_at_1.append(metrics.hit_at(flags, 1))
        hit_at_5.append(metrics.hit_at(flags, 5))
        rrs.append(metrics.reciprocal_rank(flags))
        diversities.append(metrics.diversity([metrics.product_type(n, brands) for n in names]))
        search_ms.append(result["timing"]["search_ms"])
        tag = case["tag"]
        bucket = by_tag.setdefault(tag, {"hit_at_5": [], "n": 0})
        bucket["hit_at_5"].append(hit_at_5[-1])
        bucket["n"] += 1
        if hit_at_5[-1] == 0:
            failures.append({"query": case["query"], "tag": tag, "top": names[:3]})
    metrics_out = {
        "hit_at_1": round(statistics.mean(hit_at_1), 3),
        "hit_at_5": round(statistics.mean(hit_at_5), 3),
        "mrr": round(statistics.mean(rrs), 3),
        "diversity_at_5": round(statistics.mean(diversities), 1),
        "p50_search_ms": metrics.p50(search_ms),
    }
    by_tag_out = {
        tag: {"hit_at_5": round(statistics.mean(b["hit_at_5"]), 3), "n": b["n"]}
        for tag, b in by_tag.items()
    }
    return metrics_out, by_tag_out, failures


def evaluate_documents(cases: list[dict]) -> tuple[dict, dict, list[dict]]:
    hit_at_1, hit_at_3, rrs, search_ms = [], [], [], []
    by_tag = {}
    failures = []
    for case in cases:
        result = tools.search_documents(case["query"], k=3)
        items = result["items"]
        flags = [metrics.is_relevant_chunk(item["content"], case["answer_phrase"]) for item in items]
        hit_at_1.append(metrics.hit_at(flags, 1))
        hit_at_3.append(metrics.hit_at(flags, 3))
        rrs.append(metrics.reciprocal_rank(flags))
        search_ms.append(result["timing"]["search_ms"])
        tag = case["tag"]
        bucket = by_tag.setdefault(tag, {"hit_at_3": [], "n": 0})
        bucket["hit_at_3"].append(hit_at_3[-1])
        bucket["n"] += 1
        if hit_at_3[-1] == 0:
            failures.append({
                "query": case["query"],
                "tag": tag,
                "top": [item["title"] for item in items],
            })
    metrics_out = {
        "hit_at_1": round(statistics.mean(hit_at_1), 3),
        "hit_at_3": round(statistics.mean(hit_at_3), 3),
        "mrr": round(statistics.mean(rrs), 3),
        "p50_search_ms": metrics.p50(search_ms),
    }
    by_tag_out = {
        tag: {"hit_at_3": round(statistics.mean(b["hit_at_3"]), 3), "n": b["n"]}
        for tag, b in by_tag.items()
    }
    return metrics_out, by_tag_out, failures


def run_products(source_cases: list[dict], skip_ingest: bool) -> tuple[list[dict], list[str]]:
    configs = []
    invalid_labels: list[str] = []
    valid_cases = source_cases
    for index, (mode, name) in enumerate(PRODUCT_CONFIGS):
        if not skip_ingest:
            ingest_products.main(16, 64, mode)
        if index == 0:
            valid_cases, invalid_labels = check_product_labels(source_cases)
        metrics_out, by_tag, failures = evaluate_products(valid_cases, BRANDS)
        configs.append({
            "key": mode,
            "name": name,
            "metrics": metrics_out,
            "by_tag": by_tag,
            "failures": failures,
        })
    return configs, invalid_labels


def run_documents(source_cases: list[dict], skip_ingest: bool) -> list[dict]:
    configs = []
    for chunk_size, overlap in DOCUMENT_CONFIGS:
        if not skip_ingest:
            ingest_pdfs.main(pdf_paths(), chunk_size, overlap, 16, 64)
        metrics_out, by_tag, failures = evaluate_documents(source_cases)
        configs.append({
            "key": f"chunk-{chunk_size}",
            "name": f"{chunk_size}-word chunks",
            "metrics": metrics_out,
            "by_tag": by_tag,
            "failures": failures,
        })
    return configs


def restore_production() -> None:
    ingest_products.main(16, 64, "full")
    ingest_pdfs.main(pdf_paths(), 200, 40, 16, 64)


def print_summary(results: dict) -> None:
    print("\nProducts (k=5)")
    print(f"  {'config':<8} {'hit@1':>6} {'hit@5':>6} {'mrr':>6} {'div@5':>6} "
          f"{'p50ms':>6} {'fails':>5}")
    for config in results["products"]["configs"]:
        m = config["metrics"]
        print(f"  {config['key']:<8} {m['hit_at_1']:>6.3f} {m['hit_at_5']:>6.3f} "
              f"{m['mrr']:>6.3f} {m['diversity_at_5']:>6.1f} "
              f"{m['p50_search_ms']:>6.1f} {len(config['failures']):>5}")
    print("\nDocuments (k=3)")
    print(f"  {'config':<10} {'hit@1':>6} {'hit@3':>6} {'mrr':>6} "
          f"{'p50ms':>6} {'fails':>5}")
    for config in results["documents"]["configs"]:
        m = config["metrics"]
        print(f"  {config['key']:<10} {m['hit_at_1']:>6.3f} {m['hit_at_3']:>6.3f} "
              f"{m['mrr']:>6.3f} {m['p50_search_ms']:>6.1f} {len(config['failures']):>5}")
    if results["invalid_labels"]:
        print(f"\nInvalid labels: {', '.join(results['invalid_labels'])}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score search configs against eval/testset.json and write eval/results.json"
    )
    parser.add_argument("--skip-ingest", action="store_true",
                        help="skip re-ingesting tables (assume the DB is already loaded)")
    args = parser.parse_args()

    require_env()
    testset = json.loads(TESTSET_PATH.read_text())

    products_configs, invalid_labels = run_products(testset["products"], args.skip_ingest)
    documents_configs = run_documents(testset["documents"], args.skip_ingest)

    if not args.skip_ingest:
        restore_production()

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": "local replica",
        "products": {"k": 5, "production_config": "full", "configs": products_configs},
        "documents": {"k": 3, "production_config": "chunk-200", "configs": documents_configs},
        "invalid_labels": invalid_labels,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print_summary(results)


if __name__ == "__main__":
    main()