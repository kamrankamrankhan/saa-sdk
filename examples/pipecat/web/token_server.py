# All-in-one dev /session endpoint for the SAA + Daily web demo — NOT for production
# Creates an ephemeral Daily room, mints user + bot meeting tokens, summons
# the hidden SAA agent, and (when provider keys are present) spawns a Pipecat
# voice agent into the same room so the demo talks back end-to-end.
#
# The SAA API key and Daily API key stay server-side; the browser only
# receives the Daily room URL + its own user meeting token.
import asyncio
import logging
import os
import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from saa_pipecat_client import attention_agent_token, start_attention_session

logger = logging.getLogger("token-server")
logging.basicConfig(level=logging.INFO)

DAILY_API = "https://api.daily.co/v1"

# Provider keys for the optional in-process voice agent.
# Needed to run the voice_agent
VOICE_AGENT_PROVIDER_KEYS = ("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY")

def _voice_agent_enabled() -> tuple[bool, list[str]]:
    missing = [k for k in VOICE_AGENT_PROVIDER_KEYS if not os.environ.get(k)]
    return (not missing, missing)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Running voice-agent tasks keyed by room name, so the shutdown hook can
# cancel them
_agent_tasks: dict[str, asyncio.Task] = {}


@app.on_event("startup")
async def _log_mode() -> None:
    enabled, missing = _voice_agent_enabled()
    if enabled:
        logger.info(
            "Voice agent ENABLED — each /session will spawn a Pipecat bot "
            "(Silero VAD -> Deepgram -> OpenAI -> Cartesia)",
        )
    else:
        logger.warning(
            "Voice agent DISABLED — missing env vars: %s. "
            "The demo will run in overlay-only mode (SAA predictions render in "
            "the browser, but no agent talks back). Add the missing keys to "
            ".env to enable talkback.",
            ", ".join(missing),
        )


@app.on_event("shutdown")
async def _shutdown_agents() -> None:
    tasks = list(_agent_tasks.values())
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await asyncio.wait_for(t, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass
    _agent_tasks.clear()


def _daily_headers() -> dict[str, str]:
    key = os.environ.get("DAILY_API_KEY")
    if not key:
        raise HTTPException(500, "DAILY_API_KEY not set")
    return {"Authorization": f"Bearer {key}"}


async def _create_daily_room() -> dict:
    # ephemeral room that auto-expires 1h out, dev-only
    name = f"saa-demo-{int(time.time())}"
    exp = int(time.time()) + 3600
    async with httpx.AsyncClient(timeout=20.0) as http:
        r = await http.post(
            f"{DAILY_API}/rooms",
            headers=_daily_headers(),
            json={
                "name": name,
                "properties": {
                    "exp": exp,
                    "app_message_size_limit": 65536,
                },
            },
        )
    if r.status_code >= 300:
        raise HTTPException(r.status_code, f"daily rooms create failed: {r.text}")
    body = r.json()
    return {"name": body["name"], "url": body["url"]}


async def _mint_meeting_token(
    room_name: str, user_name: str, *, is_owner: bool = False,
) -> str:
    # Normal-user meeting token (full canSend, visible in participant list).
    # The voice agent gets is_owner=True
    async with httpx.AsyncClient(timeout=20.0) as http:
        r = await http.post(
            f"{DAILY_API}/meeting-tokens",
            headers=_daily_headers(),
            json={
                "properties": {
                    "room_name": room_name,
                    "user_name": user_name,
                    "is_owner": is_owner,
                    "exp": int(time.time()) + 3600,
                }
            },
        )
    if r.status_code >= 300:
        raise HTTPException(r.status_code, f"daily token mint failed: {r.text}")
    return r.json()["token"]


async def _spawn_voice_agent(
    *, room_url: str, bot_token: str, saa_agent_identity: str, room_name: str,
) -> None:
    """Start the embedded Pipecat voice agent as a background asyncio task.

    Lazy-imports the agent module so an env without pipecat-ai installed can
    still serve the overlay-only path. Failures are logged but do NOT raise
    from /session — the browser still gets a working overlay even if the
    talkback bot crashes at construction time.
    """
    try:
        from voice_agent import run_voice_agent
    except Exception:
        logger.exception(
            "Failed to import voice_agent module — overlay-only for room=%s",
            room_url,
        )
        return

    async def _wrap() -> None:
        try:
            await run_voice_agent(
                room_url=room_url,
                bot_token=bot_token,
                saa_agent_identity=saa_agent_identity,
                openai_api_key=os.environ["OPENAI_API_KEY"],
                deepgram_api_key=os.environ["DEEPGRAM_API_KEY"],
                cartesia_api_key=os.environ["CARTESIA_API_KEY"],
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("voice agent crashed for room=%s", room_url)
        finally:
            _agent_tasks.pop(room_name, None)

    task = asyncio.create_task(_wrap(), name=f"voice-agent-{room_name}")
    _agent_tasks[room_name] = task
    logger.info("voice agent spawned for room=%s (task=%s)", room_url, task.get_name())


@app.get("/session")
async def session(room: Optional[str] = None) -> dict:
    # Use the named room if pinned, otherwise spin up a fresh ephemeral one.
    if room:
        room_name = room
        domain = os.environ.get("DAILY_DOMAIN")
        if not domain:
            raise HTTPException(400, "DAILY_DOMAIN must be set to use ?room=")
        room_url = f"https://{domain}.daily.co/{room_name}"
    else:
        info = await _create_daily_room()
        room_name = info["name"]
        room_url = info["url"]

    # Single identity, used for both the meeting token and the SAA bot's
    # target lookup. The SAA bot matches against participant["info"]["userName"]
    # which is what the meeting token's user_name resolves to on the daily-js
    # side.
    human_identity = f"user-{int(time.time())}"
    user_token = await _mint_meeting_token(room_name, human_identity)

    # Hidden SAA bot meeting token, minted with OUR Daily API key.
    # In production the customer mints this with their own key.
    saa_agent_token = attention_agent_token(
        daily_api_key=os.environ["DAILY_API_KEY"],
        room_name=room_name,
    )

    # Summon the hidden SAA bot into the room.
    handle = await start_attention_session(
        api_key=os.environ["SAA_API_KEY"],
        room_url=room_url,
        agent_token=saa_agent_token,
        participant_identity=human_identity,
    )

    # Optional in-process voice agent. Spawns iff all provider keys are set.
    enabled, missing = _voice_agent_enabled()
    if enabled:
        bot_token = await _mint_meeting_token(
            room_name, "SAA Voice Agent", is_owner=True,
        )
        await _spawn_voice_agent(
            room_url=room_url,
            bot_token=bot_token,
            saa_agent_identity=handle.agent_identity,
            room_name=room_name,
        )

    return {
        "room_url": room_url,
        "user_token": user_token,
        "user_name": human_identity,
        "agent_identity": handle.agent_identity,
        "session_id": handle.session_id,
        "voice_agent_enabled": enabled,
        "voice_agent_missing_env": missing,
    }


# serve index.html + app.js + styles.css from the same origin
# (declared after /session so the route resolves first)
app.mount("/", StaticFiles(directory=os.path.dirname(__file__), html=True), name="static")
