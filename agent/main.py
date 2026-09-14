"""LiveKit voice agent: the spoken version of the Ravo X-Ray store assistant."""

import asyncio
import json
import logging

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    function_tool,
    inference,
)
from livekit.plugins import openai, silero

from ravo_core import assistant, tools

AGENT_NAME = "ravo-xray"
SESSION_LIMIT_SECONDS = 180
VOICE_MODEL = "gpt-4.1-mini"
MAX_PRODUCTS_FOR_VOICE = 5
CARDS_TOPIC = "ravo.cards"
BASE_RULES = assistant.BASE_RULES

logger = logging.getLogger(__name__)

VOICE_PROMPT = assistant.BASE_RULES + (
    " Your reply is spoken aloud, so say no lists, no markdown and no symbols. "
    "Use one or two short sentences. Say prices naturally and always name the "
    "currency: 'about sixty-five pounds' is wrong, say 'sixty-five Egyptian "
    "pounds'."
)


def build_cards_payload(items: list[dict], timing: dict) -> dict:
    """Build the JSON payload the client uses to render product cards."""
    return {
        "cards": [
            {
                "code": item["code"],
                "name": item["name"],
                "category": item["category"],
                "price_egp": item["price_egp"],
                "stock": item["stock"],
            }
            for item in items
        ],
        "timing": {
            "embed_ms": timing["embed_ms"],
            "cache": timing["cache"],
            "search_ms": timing["search_ms"],
        },
    }


async def _search_products(query: str) -> tuple[str, list[dict], dict]:
    result = await asyncio.to_thread(
        tools.search_products, query, k=assistant.SEARCH_CANDIDATES
    )
    items = assistant.unique_by_name(result["items"])[:MAX_PRODUCTS_FOR_VOICE]
    return assistant.render_products(items), items, result["timing"]


async def _search_documents(query: str) -> str:
    result = await asyncio.to_thread(tools.search_documents, query)
    return assistant.render_documents(result["items"])


async def _send_cards(session: AgentSession, items: list[dict], timing: dict) -> None:
    payload = json.dumps(build_cards_payload(items, timing))
    await session.room_io.room.local_participant.send_text(payload, topic=CARDS_TOPIC)


class StoreAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=VOICE_PROMPT)

    @function_tool
    async def search_products(self, query: str) -> str:
        """Search the store catalog for products, prices and stock levels."""
        text, items, timing = await _search_products(query)
        try:
            await _send_cards(self.session, items, timing)
        except Exception:
            logger.exception("could not send product cards to the room")
        return text

    @function_tool
    async def search_documents(self, query: str) -> str:
        """Search store documents for policies, returns, hours and delivery."""
        return await _search_documents(query)

    async def on_enter(self) -> None:
        self.session.say("Hi, I'm the store assistant. What are you looking for?")


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def _say_goodbye_after_limit(session: AgentSession, ctx: JobContext) -> None:
    await asyncio.sleep(SESSION_LIMIT_SECONDS)
    try:
        await session.say("That's all the time I have today. Goodbye!")
    except Exception:
        logger.warning("could not say the goodbye message", exc_info=True)
    finally:
        ctx.shutdown(reason="session time limit reached")


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    session = AgentSession(
        stt=inference.STT(model="deepgram/flux-general", language="en"),
        tts=inference.TTS(model="inworld/inworld-tts-2:Ashley"),
        llm=openai.LLM(model=VOICE_MODEL),
        vad=ctx.proc.userdata["vad"],
    )
    await session.start(agent=StoreAgent(), room=ctx.room)
    asyncio.create_task(_say_goodbye_after_limit(session, ctx))


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=AGENT_NAME,
        )
    )