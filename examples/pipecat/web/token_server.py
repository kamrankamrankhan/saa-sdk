# minimal dev /session endpoint for the SAA + Daily web demo — NOT for production
# mints an ephemeral room + user meeting token via Daily REST and summons the
# hidden SAA agent for the room. The SAA API key stays server-side; the browser
# never sees it.
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from saa_pipecat_client import attention_agent_token, start_attention_session

DAILY_API = "https://api.daily.co/v1"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _daily_headers() -> dict[str, str]:
    key = os.environ.get("DAILY_API_KEY")
    if not key:
        raise HTTPException(500, "DAILY_API_KEY not set")
    return {"Authorization": f"Bearer {key}"}


async def _create_daily_room() -> dict:
    # ephemeral room — auto-expires 1h out, dev-only
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
                    # raise app-message ceiling so SAA turn_chunks fit comfortably
                    "app_message_size_limit": 65536,
                },
            },
        )
    if r.status_code >= 300:
        raise HTTPException(r.status_code, f"daily rooms create failed: {r.text}")
    body = r.json()
    return {"name": body["name"], "url": body["url"]}


async def _mint_user_token(room_name: str, user_name: str) -> str:
    # normal-user meeting token: full canSend, visible in the participant list
    async with httpx.AsyncClient(timeout=20.0) as http:
        r = await http.post(
            f"{DAILY_API}/meeting-tokens",
            headers=_daily_headers(),
            json={
                "properties": {
                    "room_name": room_name,
                    "user_name": user_name,
                    "exp": int(time.time()) + 3600,
                }
            },
        )
    if r.status_code >= 300:
        raise HTTPException(r.status_code, f"daily token mint failed: {r.text}")
    return r.json()["token"]


@app.get("/session")
async def session(room: str | None = None) -> dict:
    # use the named room if the caller pinned one, otherwise spin up a fresh
    # ephemeral one
    if room:
        room_name = room
        room_url = f"https://{os.environ.get('DAILY_DOMAIN', '')}.daily.co/{room_name}"
        if not os.environ.get("DAILY_DOMAIN"):
            raise HTTPException(400, "DAILY_DOMAIN must be set to use ?room=")
    else:
        info = await _create_daily_room()
        room_name = info["name"]
        room_url = info["url"]

    # one identity, used for both the meeting token and for telling SAA
    human_identity = f"user-{int(time.time())}"
    user_token = await _mint_user_token(room_name, human_identity)

    # mint hidden-bot meeting token using OUR Daily API key (dev demo only —
    # in production the customer mints this on their side with their own key)
    agent_token = attention_agent_token(
        daily_api_key=os.environ["DAILY_API_KEY"],
        room_name=room_name,
    )

    # summon the hidden SAA agent into the room
    handle = await start_attention_session(
        api_key=os.environ["SAA_API_KEY"],
        room_url=room_url,
        agent_token=agent_token,
        participant_identity=human_identity,
    )

    return {
        "room_url": room_url,
        "user_token": user_token,
        "user_name": human_identity,
        "agent_identity": handle.agent_identity,
        "session_id": handle.session_id,
    }


# serve index.html + app.js + styles.css from the same origin
# (declared after /session so the route resolves first)
app.mount("/", StaticFiles(directory=os.path.dirname(__file__), html=True), name="static")
