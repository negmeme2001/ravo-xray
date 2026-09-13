"""
Ingest PDF documents into Postgres: extract text, chunk it, embed each chunk,
bulk-load, build an HNSW index.

This is the miniature of Ravo's document pipeline (§8/§10 of onboarding):
a "collection" (PDF) -> chunks -> embed (text-embedding-3-small) -> collections_embedding.

Chunking is the PDF-specific Pillar 1 lever: what text lands in each embedding
decides what "similar" means. CHUNK_SIZE / CHUNK_OVERLAP are the knobs — change
them and re-run ingest + benchmark to see retrieval quality move.

Usage:
    uv run python ingest_pdfs.py                            # all PDFs in docs/
    uv run python ingest_pdfs.py docs/a.pdf docs/b.pdf      # specific files
    uv run python ingest_pdfs.py --chunk-size 100 --overlap 20 --m 16 --efc 200
"""

import argparse
import glob
import json

import pdfplumber

from db import connect, embed_batch, EMBED_DIM, timer


def extract_text(path: str) -> str:
    """Pull all text out of a PDF, page by page."""
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window chunking over words.

    chunk_size : words per chunk            (bigger = richer context, coarser)
    overlap    : words shared with the previous chunk  (smooths cuts mid-topic)
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        if len(chunk) >= chunk_size // 2:      # drop tiny trailing stubs
            chunks.append(" ".join(chunk))
    return chunks


def load_docs(paths: list[str], chunk_size: int, overlap: int) -> list[dict]:
    rows = []
    for path in paths:
        text = extract_text(path)
        chunks = chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            rows.append({
                "code": f"DOC-{path}-{i}",
                "source": path,
                "chunk_index": i,
                "title": path.split("/")[-1],
                "content": chunk,
            })
        print(f"  {path}: {len(chunks)} chunks")
    return rows


def main(paths: list[str], chunk_size: int, overlap: int, m: int, efc: int):
    rows = load_docs(paths, chunk_size, overlap)
    print(f"Loaded {len(rows)} chunks across {len(paths)} PDF(s)")

    conn = connect()
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Fresh start each run so experiments are clean.
    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute(f"""
        CREATE TABLE documents (
            id          bigserial PRIMARY KEY,
            code        text UNIQUE,
            source      text,
            title       text,
            chunk_index int,
            content     text,
            embedding   vector({EMBED_DIM})
        )
    """)

    # --- Embed in batches (Pillar 1) ---
    print("Embedding chunks (batches of 200)...")
    with timer() as t_embed:
        embeddings = []
        for i in range(0, len(rows), 200):
            chunk = rows[i:i + 200]
            embeddings.extend(embed_batch([r["content"] for r in chunk]))
            print(f"  embedded {min(i + 200, len(rows))}/{len(rows)}")
    print(f"Embedding done in {t_embed.ms/1000:.1f}s")

    # --- Bulk load with COPY (fastest way in; pgvector docs recommend it) ---
    print("Bulk loading with COPY...")
    with timer() as t_copy:
        with conn.cursor().copy(
            "COPY documents (code, source, title, chunk_index, content, embedding) "
            "FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["text", "text", "text", "integer", "text", "vector"])
            for r, emb in zip(rows, embeddings):
                copy.write_row([r["code"], r["source"], r["title"],
                                r["chunk_index"], r["content"], emb])
    print(f"COPY done in {t_copy.ms:.0f}ms")

    # --- Build the HNSW index AFTER loading (pgvector best practice) ---
    # m               : max connections per node (graph density) — build-time
    # ef_construction : candidate list size while building — build-time
    # vector_cosine_ops : cosine distance, matching Ravo's metric
    print(f"Building HNSW index (m={m}, ef_construction={efc})...")
    with timer() as t_index:
        conn.execute(
            f"CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {m}, ef_construction = {efc})"
        )
    print(f"Index built in {t_index.ms/1000:.1f}s")

    n = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    print(f"\nDone. {n} document chunks indexed. Try:  "
          f"uv run python search.py \"what is the return policy\"")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="*", help="PDF paths; defaults to docs/*.pdf")
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=64, help="ef_construction")
    args = ap.parse_args()

    paths = args.pdfs or sorted(glob.glob("docs/*.pdf"))
    if not paths:
        raise SystemExit("No PDFs found. Run `uv run python generate_docs.py` first.")
    main(paths, args.chunk_size, args.overlap, args.m, args.efc)
