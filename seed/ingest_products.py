"""
Ingest catalog.jsonl into Postgres: embed every product, bulk-load, build an HNSW index.

This is the miniature of Ravo's ingestion pipeline (§10 of onboarding):
external feed -> standardized JSON -> embed (text-embedding-3-small) -> products + products_embedding.

Usage:
    uv run python ingest.py                 # defaults: m=16, ef_construction=64
    uv run python ingest.py --m 16 --efc 200
"""

import argparse
import json

from db import connect, embed_batch, product_text, EMBED_DIM, timer


def load_rows(path="catalog.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main(m: int, efc: int):
    rows = load_rows()
    print(f"Loaded {len(rows)} products from catalog.jsonl")

    conn = connect()
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Fresh start each run so experiments are clean.
    conn.execute("DROP TABLE IF EXISTS products")
    conn.execute(f"""
        CREATE TABLE products (
            id          bigserial PRIMARY KEY,
            code        text UNIQUE,
            name        text,
            category    text,
            brand       text,
            description text,
            price_egp   double precision,
            stock       int,
            embedding   vector({EMBED_DIM})
        )
    """)

    # --- Embed in batches (Pillar 1) ---
    print("Embedding products (batches of 200)...")
    with timer() as t_embed:
        embeddings = []
        for i in range(0, len(rows), 200):
            chunk = rows[i:i + 200]
            embeddings.extend(embed_batch([product_text(p) for p in chunk]))
            print(f"  embedded {min(i + 200, len(rows))}/{len(rows)}")
    print(f"Embedding done in {t_embed.ms/1000:.1f}s")

    # --- Bulk load with COPY (fastest way in; pgvector docs recommend it) ---
    print("Bulk loading with COPY...")
    with timer() as t_copy:
        with conn.cursor().copy(
            "COPY products (code, name, category, brand, description, price_egp, stock, embedding) "
            "FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["text", "text", "text", "text", "text", "float8", "integer", "vector"])
            for p, emb in zip(rows, embeddings):
                copy.write_row([p["code"], p["name"], p["category"], p["brand"],
                                p["description"], p["price_egp"], p["stock"], emb])
    print(f"COPY done in {t_copy.ms:.0f}ms")

    # --- Build the HNSW index AFTER loading (pgvector best practice) ---
    # m               : max connections per node (graph density) — build-time
    # ef_construction : candidate list size while building — build-time
    # vector_cosine_ops : cosine distance, matching Ravo's metric
    print(f"Building HNSW index (m={m}, ef_construction={efc})...")
    with timer() as t_index:
        conn.execute(
            f"CREATE INDEX ON products USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {m}, ef_construction = {efc})"
        )
    print(f"Index built in {t_index.ms/1000:.1f}s")

    n = conn.execute("SELECT count(*) FROM products").fetchone()[0]
    print(f"\nDone. {n} products indexed. Try:  uv run python search.py \"something for a headache\"")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=64, help="ef_construction")
    args = ap.parse_args()
    main(args.m, args.efc)
