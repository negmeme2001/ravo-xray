import os

import psycopg
import pytest
from pgvector import Vector

import ravo_core.tools as tools
from ravo_core.db import EMBED_DIM


class FakeTimer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ms = 3.456
        return False


class FakeConn:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []
        self.executed = []
        self.closed = False

    def transaction(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return self

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_shared_connection():
    yield
    tools.set_conn(None)


def _records(monkeypatch, rows):
    conn = FakeConn(rows)
    tools.set_conn(conn)
    monkeypatch.setattr(tools, "embed_cached", lambda text: ([0.1, 0.2], {"cache": "miss", "ms": 1.23}))
    monkeypatch.setattr(tools.db, "timer", FakeTimer)
    return conn


def test_search_products_shapes_and_timing(monkeypatch):
    conn = _records(monkeypatch, [["P1", "Almarai Milk", "Dairy", 25.5, 10, 0.912345]])
    result = tools.search_products("milk")
    assert result["items"] == [
        {
            "code": "P1",
            "name": "Almarai Milk",
            "category": "Dairy",
            "price_egp": 25.5,
            "stock": 10,
            "similarity": 0.912,
        }
    ]
    assert result["timing"] == {"embed_ms": 1.2, "cache": "miss", "search_ms": 3.5}


def test_search_runs_ef_search_before_select(monkeypatch):
    conn = _records(monkeypatch, [])
    tools.search_products("milk")
    assert conn.executed[0][0] == f"SET LOCAL hnsw.ef_search = {tools.EF_SEARCH}"
    assert conn.executed[0][1] is None
    assert "SELECT code, name, category, price_egp, stock" in conn.executed[1][0]
    assert "SET LOCAL" not in conn.executed[1][0]


def test_in_stock_only_controls_filter(monkeypatch):
    conn = _records(monkeypatch, [])
    tools.search_products("milk", in_stock_only=True)
    assert "WHERE stock > 0" in conn.executed[1][0]
    tools.search_products("milk", in_stock_only=False)
    assert "WHERE stock > 0" not in conn.executed[3][0]


def test_product_params_use_vector(monkeypatch):
    conn = _records(monkeypatch, [])
    tools.search_products("milk", in_stock_only=True, k=7)
    params = conn.executed[1][1]
    assert isinstance(params[0], Vector)
    assert isinstance(params[1], Vector)
    assert params[2] == 7


def test_search_documents_title_is_parameter(monkeypatch):
    conn = _records(monkeypatch, [["Guide", 2, "some content", 0.8765432]])
    result = tools.search_documents("refund", title="Return Policy", k=3)
    sql, params = conn.executed[1]
    assert "WHERE title = %s" in sql
    assert "Return Policy" not in sql
    assert isinstance(params[0], Vector)
    assert params[1] == "Return Policy"
    assert isinstance(params[2], Vector)
    assert params[3] == 3
    assert result["items"] == [
        {"title": "Guide", "chunk_index": 2, "content": "some content", "similarity": 0.877}
    ]
    assert result["timing"] == {"embed_ms": 1.2, "cache": "miss", "search_ms": 3.5}


def test_search_documents_without_title(monkeypatch):
    conn = _records(monkeypatch, [])
    tools.search_documents("refund")
    sql, params = conn.executed[1]
    assert "WHERE" not in sql
    assert isinstance(params[0], Vector)
    assert isinstance(params[1], Vector)
    assert params[2] == 3


class DroppedConn(FakeConn):
    """A connection the database closed while idle: the first query fails."""

    def execute(self, sql, params=None):
        raise psycopg.OperationalError("server closed the connection unexpectedly")


def test_reconnects_once_after_idle_connection_drop(monkeypatch):
    fresh = FakeConn([["P1", "Almarai Milk", "Dairy", 25.5, 10, 0.9]])
    connects = []
    monkeypatch.setattr(tools.db, "connect", lambda: connects.append(1) or fresh)
    monkeypatch.setattr(tools, "embed_cached", lambda text: ([0.1], {"cache": "hit", "ms": 0.5}))
    monkeypatch.setattr(tools.db, "timer", FakeTimer)
    tools.set_conn(DroppedConn())

    result = tools.search_products("milk")

    assert connects == [1]
    assert result["items"][0]["code"] == "P1"


@pytest.mark.db
def test_search_products_against_real_db(monkeypatch):
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")
    monkeypatch.setattr(
        tools, "embed_cached", lambda text: ([0.05] * EMBED_DIM, {"cache": "off", "ms": 0.1})
    )
    result = tools.search_products("whole milk")
    assert isinstance(result["items"], list)
    assert set(result["timing"]) == {"embed_ms", "cache", "search_ms"}


@pytest.mark.db
def test_search_documents_against_real_db(monkeypatch):
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")
    monkeypatch.setattr(
        tools, "embed_cached", lambda text: ([0.05] * EMBED_DIM, {"cache": "off", "ms": 0.1})
    )
    result = tools.search_documents("refund")
    assert isinstance(result["items"], list)
    assert set(result["timing"]) == {"embed_ms", "cache", "search_ms"}