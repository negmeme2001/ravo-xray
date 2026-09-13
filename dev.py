"""Local dev server: app routes plus public files served under /.

Run with `uv run uvicorn dev:app --reload`. Vercel serves public/ from its
CDN in production, so app.py never mounts it; this module does for local runs.
"""

from fastapi.staticfiles import StaticFiles

from app import app

app.mount("/", StaticFiles(directory="public", html=True), name="public")