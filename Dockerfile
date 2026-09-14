FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group voice

COPY ravo_core/ ravo_core/
COPY agent/ agent/

# Use the venv built above directly: `uv run` would re-sync the default (dev)
# dependency group at build time and again on every container start.
RUN .venv/bin/python -m livekit.agents download-files

CMD [".venv/bin/python", "-m", "agent.main", "start"]