"""Vercel zero-config FastAPI entrypoint for the Ravo X-Ray demo.

Serves the page behind an access-code gate and exposes the chat and eval
APIs. Static assets live in public/ and are served by Vercel's CDN.
"""

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from livekit.api import AccessToken, RoomAgentDispatch, RoomConfiguration, VideoGrants

from ravo_core import assistant, tools
from ravo_core.embed_cache import get_redis
from ravo_core.limits import allow

ROOT = Path(__file__).parent

SESSION_COOKIE = "rx_session"
SESSION_TTL_SECONDS = 7 * 24 * 3600
EVAL_RESULTS_PATH = ROOT / "eval" / "results.json"
CONVERSATION_TURN_LIMIT = 20
MAX_USER_CHARS = 500
MAX_ASSISTANT_CHARS = 2000

VOICE_ROOM_TTL_MINUTES = 10
VOICE_PER_IP_LIMIT = 10
VOICE_PER_IP_WINDOW_SECONDS = 3600
VOICE_AGENT_NAME = "ravo-xray"

logger = logging.getLogger(__name__)

app = FastAPI(title="Ravo X-Ray")


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


def _session_signature(value: str) -> str:
    return hmac.new(
        os.environ["SESSION_SECRET"].encode("utf-8"),
        value.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()


def _make_session_value(expiry: int) -> str:
    return f"{expiry}.{_session_signature(str(expiry))}"


def _verify_session(value: str | None) -> bool:
    if not value or not os.environ.get("SESSION_SECRET"):
        return False
    raw_expiry, dot, signature = value.partition(".")
    if not dot or not raw_expiry.isdigit() or not signature:
        return False
    expected = _session_signature(raw_expiry)
    if not hmac.compare_digest(signature.encode("utf-8"), expected.encode("ascii")):
        return False
    return int(raw_expiry) > time.time()


def require_session(request: Request) -> None:
    if not _verify_session(request.cookies.get(SESSION_COOKIE)):
        raise HTTPException(status_code=401, detail={"error": "Enter the access code first."})


def _is_configured() -> bool:
    return bool(os.environ.get("DEMO_ACCESS_CODE") and os.environ.get("SESSION_SECRET"))


def _voice_is_enabled() -> bool:
    return bool(
        os.environ.get("LIVEKIT_URL")
        and os.environ.get("LIVEKIT_API_KEY")
        and os.environ.get("LIVEKIT_API_SECRET")
    )


def _validate_chat_payload(payload: dict) -> str | None:
    messages = payload.get("messages")
    conversation_id = payload.get("conversation_id")
    if not isinstance(messages, list) or not messages:
        return "messages must be a non-empty list."
    if not isinstance(conversation_id, str) or not conversation_id:
        return "conversation_id is required."
    if len(conversation_id) > 64:
        return "conversation_id must be at most 64 characters."
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            return "message roles must be user or assistant."
        # The history comes from the browser, so every message is capped: otherwise a
        # client could pad earlier turns to inflate the model's input cost.
        content = message.get("content")
        limit = MAX_USER_CHARS if message["role"] == "user" else MAX_ASSISTANT_CHARS
        if not isinstance(content, str) or len(content) > limit:
            return f"each {message['role']} message must be text of at most {limit} characters."
    last = messages[-1]
    if last.get("role") != "user":
        return "the last message must come from the user."
    content = last.get("content")
    if not isinstance(content, str) or not content.strip():
        return "the last message needs content."
    if len(content) > 500:
        return "the last message must be at most 500 characters."
    return None


def _conv_count(key: str) -> int:
    client = get_redis()
    if client is None:
        return 1
    try:
        raw = client.get(key)
    except Exception:
        logger.warning("could not read conversation count for %s", key, exc_info=True)
        return 1
    try:
        return int(raw) if raw is not None else 1
    except (TypeError, ValueError):
        return 1


@app.get("/")
def index(request: Request):
    if not _verify_session(request.cookies.get(SESSION_COOKIE)):
        return RedirectResponse("/login.html", status_code=303)
    html = (ROOT / "templates" / "app.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.post("/api/auth")
def auth(request: Request, payload: dict = Body(...)):
    if not _is_configured():
        raise HTTPException(status_code=500, detail={"error": "The demo isn't configured yet."})

    ip = _client_ip(request)
    if not allow(f"auth:{ip}", 10, 900):
        raise HTTPException(
            status_code=429,
            detail={"error": "Too many attempts. Wait a few minutes."},
        )

    entered = payload.get("code")
    if not isinstance(entered, str) or not hmac.compare_digest(
        entered.encode("utf-8"), os.environ["DEMO_ACCESS_CODE"].encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail={"error": "That code isn't right."})

    expiry = int(time.time()) + SESSION_TTL_SECONDS
    response = JSONResponse({"ok": True})
    response.set_cookie(
        SESSION_COOKIE,
        _make_session_value(expiry),
        max_age=SESSION_TTL_SECONDS,
        expires=datetime.fromtimestamp(expiry, tz=timezone.utc),
        path="/",
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response


@app.get("/api/health")
def health(_: None = Depends(require_session)):
    db_ok = True
    try:
        tools.get_conn().execute("SELECT 1")
    except Exception:
        logger.warning("health check could not reach the database", exc_info=True)
        db_ok = False
    return {"ok": True, "db": db_ok}


@app.post("/api/chat")
def chat(request: Request, payload: dict = Body(...), _: None = Depends(require_session)):
    error = _validate_chat_payload(payload)
    if error is not None:
        raise HTTPException(status_code=400, detail={"error": error})

    messages = payload["messages"][-10:]
    ip = _client_ip(request)

    if not allow(f"chat:{ip}", 60, 3600):
        raise HTTPException(
            status_code=429,
            detail={"error": "Too many questions from this network. Try again in an hour."},
        )

    conversation_id = payload["conversation_id"]
    conv_key = f"conv:{conversation_id}"
    if not allow(conv_key, CONVERSATION_TURN_LIMIT, 86400):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "This conversation has reached its limit. Start a new one to keep going."
            },
        )

    try:
        result = assistant.run_turn(messages)
    except Exception:
        logger.warning("assistant.run_turn failed", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail={"error": "The assistant couldn't answer just now. Try again in a moment."},
        )

    result["turns_left"] = max(0, CONVERSATION_TURN_LIMIT - _conv_count(conv_key))
    return result


@app.get("/api/eval")
def eval_scores(_: None = Depends(require_session)):
    if not EVAL_RESULTS_PATH.is_file():
        raise HTTPException(status_code=404, detail={"error": "No scores yet."})
    return json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))


@app.get("/api/voice/status")
def voice_status(_: None = Depends(require_session)):
    return {"enabled": _voice_is_enabled()}


@app.post("/api/voice/token")
def voice_token(request: Request, _: None = Depends(require_session)):
    if not _voice_is_enabled():
        raise HTTPException(status_code=503, detail={"error": "Voice isn't set up yet."})

    ip = _client_ip(request)
    if not allow(f"voice:{ip}", VOICE_PER_IP_LIMIT, VOICE_PER_IP_WINDOW_SECONDS):
        raise HTTPException(
            status_code=429,
            detail={"error": "Too many voice sessions from this network. Try again later."},
        )

    room = f"rx-{uuid.uuid4().hex[:12]}"
    identity = f"shopper-{uuid.uuid4().hex[:8]}"

    room_config = RoomConfiguration(name=room)
    room_config.agents.append(RoomAgentDispatch(agent_name=VOICE_AGENT_NAME))

    token = (
        AccessToken(
            os.environ["LIVEKIT_API_KEY"],
            os.environ["LIVEKIT_API_SECRET"],
        )
        .with_identity(identity)
        .with_name("Shopper")
        .with_ttl(timedelta(minutes=VOICE_ROOM_TTL_MINUTES))
        .with_room_config(room_config)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )

    return {"url": os.environ["LIVEKIT_URL"], "token": token, "room": room}