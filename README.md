# Ravo X-Ray

A retail store assistant you can talk to, where every answer comes with an **x-ray**: a trace of
where that answer's time went. Built to show three additions to a Ravo-style retrieval stack:

- **Faster**: a Redis query-embedding cache, plus a per-turn latency trace.
- **Measurable**: a labelled eval set that scores every search setup.
- **Spoken too**: press Talk and the same assistant answers out loud (LiveKit Cloud agent,
  streaming speech recognition), using the same search and cache.
- **Honest about failures**: known weak spots (negation, near-duplicates) are kept on the
  scoreboard as the targets for the next improvements.

The data is synthetic: a 1,500-product catalog and three store documents. The baseline is
standard vector search.

## What it found

| Finding | Number |
|---|---|
| Postgres + pgvector search over 1,500 products | **~3–6 ms** |
| Turning a question into an embedding (OpenAI API, from the server) | **~160 ms**, spiking past 800 ms |
| The same question again, from the Redis cache | **~3 ms** |
| Share of a full answer spent in the language model (2 calls) | **about 90%** |
| Right product first / in top 5 (name, category and description embedded) | **81% / 84%** |
| Negation questions ("milk that isn't from a cow") answered correctly | **0 of 3** |
| Distinct products in the top 5 | **1.8 of 5** |

So the database is not the bottleneck. The cache removes the embedding call, and the biggest
remaining lever is the language model: one call instead of two, and streaming the reply.

## How it works

```
Browser ─► FastAPI on Vercel (fra1) ─► ravo_core
                                        ├─ assistant.py   tool-calling loop (gpt-4.1-mini) + trace
                                        ├─ models.py      shared model IDs
                                        ├─ prompts/       versioned prompts, messages and manifest
                                        ├─ tools.py       search_products / search_documents (pgvector, HNSW, ef_search=80)
                                        ├─ embed_cache.py query-embedding cache (Upstash Redis, 30-day TTL)
                                        ├─ monitor.py     live monitor: each answer's timings (no text) in Redis
                                        └─ limits.py      rate limits
                              Neon Postgres 16 + pgvector   Upstash Redis
```

- `app.py`: access-code gate (HMAC-signed cookie), `/api/chat`, `/api/monitor`, `/api/eval`, `/api/health`.
- The live monitor on the page draws the last 40 real answers (typed and spoken) from `/api/monitor`;
  its headline medians use typed answers only, because a spoken turn is timed from the shopper going quiet.
- `templates/app.html`, `public/`: the page. Plain HTML, CSS and JavaScript.
- `agent/`: the LiveKit voice agent (Deepgram Flux speech recognition and Cartesia Sonic 3 speech through
  LiveKit Inference, with an OpenAI fallback voice), deployed to LiveKit Cloud with the `Dockerfile`.
- `seed/`: generates and loads the catalog and documents.
- `eval/`: `testset.json` (31 product and 17 document questions with checkable answers),
  `run_eval.py` (scores setups and writes `results.json`), `metrics.py`.

## Usage and cost

Typed turns count model requests and tokens, embedding tokens, cache outcomes, result outcomes, and
reliability flags in small text-free Redis hashes. Costs are estimates from public list prices as of
2026-09-29, not billing data; update the versioned rates and sources in `ravo_core/pricing.py`.

## Run it locally

```bash
uv sync
export DATABASE_URL=...  OPENAI_API_KEY=...  DEMO_ACCESS_CODE=...  SESSION_SECRET=...
# optional cache: UPSTASH_REDIS_REST_URL / _TOKEN (or KV_REST_API_URL / _TOKEN)

uv run python -m seed.generate_catalog
uv run python -m seed.ingest_products
uv run python -m seed.ingest_pdfs --chunk-size 100 --overlap 20
uv run uvicorn dev:app --reload          # http://127.0.0.1:8000
```

## Tests and eval

```bash
uv run pytest -q                  # unit tests mock OpenAI and Redis; db tests need DATABASE_URL
uv run python -m eval.run_eval    # re-ingests each setup, scores it, writes eval/results.json
```

## How each addition plugs into Ravo

- **Cache:** one Redis lookup before the embedding call in the search service. Ravo already
  runs Redis.
- **Trace:** timing around the embedding call, the search and each model call, written to the
  existing logging.
- **Scoreboard:** a labelled question set per store, run on every release.

## Next

1. One model call instead of two for common questions, and a streamed reply.
2. Hybrid keyword + vector search in the same Postgres (brand and exact-name queries).
3. Collapse near-duplicates in search results (raises variety in the top 5).
4. Rewrite negated questions before searching (the negation rows).
5. Faster spoken turns: the spoken x-ray showed a 2.5 s end-of-turn wait; capping it at 1.2 s took a spoken turn
   from 3.5 s to 3.2 s and exposed the model's first call as the next cost (item 1).

Built by Mohamed Negm.
