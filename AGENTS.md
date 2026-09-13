# Ravo X-Ray — agent rules

A hosted demo: a conversational retail store assistant (FastAPI on Vercel) whose every reply shows an
"x-ray" latency trace, plus a Redis query-embedding cache and a retrieval eval scoreboard.

## Stack
- Python 3.12, managed with **uv** (`uv add`, `uv run`). Never use pip directly.
- FastAPI app at `app.py` (Vercel zero-config entrypoint). Static assets in `public/`.
- Postgres 16 + pgvector (Neon in production), `psycopg` v3 + `pgvector` Python package.
- Upstash Redis via the `upstash-redis` package (HTTP client).
- OpenAI: embeddings `text-embedding-3-small` (1536-dim), chat `gpt-4.1-mini`.
- Shared logic lives in the `ravo_core/` package; `app.py` and the future voice agent both import it.

## Gates (run all before finishing; all must pass)
```
uv sync
uv run python -m compileall -q ravo_core app.py seed eval
uv run pytest -q
```

## Rules
- Tests must never call real external services. Mock OpenAI and use an in-memory fake for Redis.
  Tests that need Postgres are marked `@pytest.mark.db` and skipped unless `DATABASE_URL` is set.
- Never write secrets, API keys, or `.env` files with real values. Read config from environment
  variables only. `.env.example` may list variable names with empty values.
- Do not run `git add` or `git commit`. Leave changes in the working tree.
- Keep changes scoped to the task you were given. No unrelated refactors or renames.
- No comments containing process language ("MVP", "for now", "phase N", "TODO later").
- Type-hint public functions. Keep functions small and named for what they do.
