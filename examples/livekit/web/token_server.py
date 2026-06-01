# minimal dev token endpoint for the SAA + LiveKit web demo — NOT for production
# mints a browser LiveKit join token and serves the static files
#
# SAA is summoned by the voice-agent worker in the room, NOT here — run one of
# examples/livekit/voice_agent_cascaded or voice_agent_realtime in the same
# LiveKit project alongside this demo. The agent owns the SAA session and
# answers you; this server just lets the browser join the room.
import os
from datetime import timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from livekit.api import AccessToken, VideoGrants

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/token")
async def token(room: str, identity: str) -> dict:
    # browser join token — publish cam+mic, subscribe to the agent's audio
    user_jwt = (
        AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_grants(
            VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)
        )
        .with_ttl(timedelta(hours=1))
        .to_jwt()
    )
    return {"url": os.environ["LIVEKIT_URL"], "token": user_jwt}


# serve index.html + app.js + styles.css from the same origin
# (declared after /token so the route resolves first)
app.mount("/", StaticFiles(directory=os.path.dirname(__file__), html=True), name="static")
