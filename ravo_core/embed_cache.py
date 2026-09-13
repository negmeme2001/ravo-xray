"""Redis query-embedding cache with float32 vector serialization."""

import base64
import hashlib
import logging
import os
import re
from array import array

from upstash_redis import Redis

from ravo_core.db import EMBED_MODEL, embed_one, timer

CACHE_TTL_SECONDS = 30 * 24 * 3600

logger = logging.getLogger(__name__)

_redis = None
_whitespace = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _whitespace.sub(" ", text.strip().lower())


def cache_key(text: str) -> str:
    digest = hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
    return f"emb:{EMBED_MODEL}:{digest}"


def encode_vector(vec: list[float]) -> str:
    return base64.b64encode(array("f", vec).tobytes()).decode("ascii")


def decode_vector(s: str) -> list[float]:
    return array("f", base64.b64decode(s)).tolist()


def _redis_credentials() -> tuple[str | None, str | None]:
    """Upstash's own names first, then the KV_* names the Vercel Marketplace injects."""
    url = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")
    return url, token


def get_redis() -> Redis | None:
    global _redis
    if _redis is None:
        url, token = _redis_credentials()
        if url and token:
            _redis = Redis(url=url, token=token)
    return _redis


def set_redis(client) -> None:
    global _redis
    _redis = client


def embed_cached(text: str) -> tuple[list[float], dict]:
    client = get_redis()
    if client is None:
        with timer() as t:
            vector = embed_one(text)
        return vector, {"cache": "off", "ms": round(t.ms, 1)}

    key = cache_key(text)
    with timer() as t:
        redis_error = False
        try:
            cached = client.get(key)
        except Exception:
            logger.warning("redis get failed", exc_info=True)
            redis_error = True
            cached = None
        if cached is not None:
            vector = decode_vector(cached)
            cache_state = "hit"
        else:
            vector = embed_one(text)
            cache_state = "miss"
            try:
                client.set(key, encode_vector(vector), ex=CACHE_TTL_SECONDS)
            except Exception:
                logger.warning("redis set failed", exc_info=True)
                redis_error = True
        if redis_error:
            cache_state = "off"
    return vector, {"cache": cache_state, "ms": round(t.ms, 1)}