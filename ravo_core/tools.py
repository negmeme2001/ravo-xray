"""Vector search tools over products and documents."""

import threading

import psycopg
from pgvector import Vector

from ravo_core import db
from ravo_core.embed_cache import embed_cached

EF_SEARCH = 80

_conn = None
# One shared connection per instance; concurrent requests must not interleave
# transactions on it, or one request's SET LOCAL could leak into another's.
_conn_lock = threading.RLock()


def get_conn():
    global _conn
    with _conn_lock:
        if _conn is None or _conn.closed or getattr(_conn, "broken", False):
            _conn = db.connect()
        return _conn


def set_conn(conn) -> None:
    global _conn
    with _conn_lock:
        _conn = conn


def _reset_conn() -> None:
    global _conn
    with _conn_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
        _conn = None


def _query_once(sql: str, params: list) -> tuple[list, float]:
    conn = get_conn()
    with conn.transaction():
        conn.execute(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}")
        with db.timer() as t:
            rows = conn.execute(sql, params).fetchall()
    return rows, round(t.ms, 1)


def _run(sql: str, params: list) -> tuple[list, float]:
    """Run a search query, reconnecting once if the database dropped an idle connection."""
    with _conn_lock:
        try:
            return _query_once(sql, params)
        except psycopg.OperationalError:
            _reset_conn()
            return _query_once(sql, params)


def search_products(query: str, k: int = 5, in_stock_only: bool = False) -> dict:
    vector, info = embed_cached(query)
    query_vector = Vector(vector)
    sql = (
        "SELECT code, name, category, price_egp, stock, "
        "1 - (embedding <=> %s) AS similarity "
        "FROM products"
    )
    params: list = [query_vector]
    if in_stock_only:
        sql += " WHERE stock > 0"
    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params += [query_vector, k]
    rows, search_ms = _run(sql, params)
    items = [
        {
            "code": row[0],
            "name": row[1],
            "category": row[2],
            "price_egp": float(row[3]),
            "stock": int(row[4]),
            "similarity": round(float(row[5]), 3),
        }
        for row in rows
    ]
    timing = {"embed_ms": round(info["ms"], 1), "cache": info["cache"], "search_ms": search_ms}
    return {"items": items, "timing": timing}


def search_documents(query: str, k: int = 3, title: str | None = None) -> dict:
    vector, info = embed_cached(query)
    query_vector = Vector(vector)
    sql = (
        "SELECT title, chunk_index, content, 1 - (embedding <=> %s) AS similarity "
        "FROM documents"
    )
    params: list = [query_vector]
    if title is not None:
        sql += " WHERE title = %s"
        params.append(title)
    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params += [query_vector, k]
    rows, search_ms = _run(sql, params)
    items = [
        {
            "title": row[0],
            "chunk_index": row[1],
            "content": row[2],
            "similarity": round(float(row[3]), 3),
        }
        for row in rows
    ]
    timing = {"embed_ms": round(info["ms"], 1), "cache": info["cache"], "search_ms": search_ms}
    return {"items": items, "timing": timing}