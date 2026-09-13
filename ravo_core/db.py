"""Shared database and embedding helpers for the Ravo X-Ray demo.

Importing this module is side-effect free and needs no credentials; the OpenAI
client is created lazily on first embed call.
"""

import os
import time

import psycopg
from pgvector.psycopg import register_vector

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

_client = None


def _openai_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI()
    return _client


def connect() -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg.connect(url, autocommit=True)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
    return conn


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = _openai_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_one(text: str) -> list[float]:
    return embed_batch([text])[0]


class timer:
    """Context manager recording elapsed time in `.ms` after the block."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ms = (time.perf_counter() - self._start) * 1000