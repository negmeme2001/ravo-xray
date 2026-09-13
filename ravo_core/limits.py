"""Fixed-window rate limiting backed by the shared Redis client."""

import logging

from ravo_core.embed_cache import get_redis

logger = logging.getLogger(__name__)


def allow(key: str, limit: int, window_s: int) -> bool:
    """Return whether the key's count in the fixed window is within limit.

    Fails open, logging a warning, when there is no Redis client or Redis
    raises: rate limiting must never take the demo down.
    """
    client = get_redis()
    if client is None:
        logger.warning("rate limit %s skipped: no redis client", key)
        return True
    try:
        count = client.incr(key)
        if count == 1:
            client.expire(key, window_s)
        return count <= limit
    except Exception:
        logger.warning("rate limit %s skipped: redis error", key, exc_info=True)
        return True