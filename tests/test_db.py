import os
import time
from types import SimpleNamespace

import pytest

import ravo_core.db as db


def _fake_client(records: list | None = None):
    records = [] if records is None else records

    class FakeEmbeddings:
        def create(self, model, input):
            records.append((model, list(input)))
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[float(len(t)), 2.0]) for t in input]
            )

    return SimpleNamespace(embeddings=FakeEmbeddings()), records


def test_timer_measures_elapsed_time():
    with db.timer() as t:
        time.sleep(0.05)
    assert t.ms >= 40


def test_client_is_lazy():
    assert db._client is None


def test_embed_batch_returns_vectors(monkeypatch):
    fake, records = _fake_client()
    monkeypatch.setattr(db, "_openai_client", lambda: fake)
    assert db.embed_batch(["ab", "abcdef"]) == [[2.0, 2.0], [6.0, 2.0]]
    assert records == [(db.EMBED_MODEL, ["ab", "abcdef"])]


def test_embed_one_returns_first_vector(monkeypatch):
    fake, _ = _fake_client()
    monkeypatch.setattr(db, "_openai_client", lambda: fake)
    assert db.embed_one("hello") == [5.0, 2.0]


def test_connect_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        db.connect()


@pytest.mark.db
def test_connect_uses_database_url():
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set")
    conn = db.connect()
    try:
        assert not conn.closed
    finally:
        conn.close()