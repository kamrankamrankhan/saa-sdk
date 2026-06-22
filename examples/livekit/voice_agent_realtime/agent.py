# SAA-gated realtime (speech-to-speech) voice agent for LiveKit Agents 1.5.x
# OpenAI Realtime, gated by Attention Labs SAA — the case where stock LiveKit has
# no VAD slot to gate on, so SAA is the only way to give a realtime model attention
import asyncio
import logging
import os
import time
from logging.handlers import RotatingFileHandler

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.voice import room_io
from livekit.plugins import openai, silero

from saa_livekit_client import (
    AttentionEngine,
    attention_agent_token,
    start_attention_session,
)

from pathlib import Path
from dotenv import load_dotenv

# auto-load the shared examples/livekit/.env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger("voice-agent-realtime")

# per-run log file at DEBUG; also captures livekit-agents + openai plugin internals
_fh = RotatingFileHandler(f"saa-agent-{int(time.time())}.log", maxBytes=5_000_000, backupCount=3)
_fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
_fh.setLevel(logging.DEBUG)
for _name in ("voice-agent-realtime", "livekit", "saa_livekit_client"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.DEBUG)
    _lg.addHandler(_fh)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice assistant. Be brief")


def _prewarm(proc) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(setup_fnc=_prewarm)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    user = await ctx.wait_for_participant()

    saa = await start_attention_session(
        api_key=os.environ["SAA_API_KEY"],
        livekit_url=os.environ["LIVEKIT_URL"],
        agent_token=attention_agent_token(
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"],
            room_name=ctx.room.name,
        ),
        room_name=ctx.room.name,
        participant_identity=user.identity,
        attention_config={"frames_per_turn": 0},
    )
    ctx.add_shutdown_callback(saa.stop)

    # speech-to-speech — Silero is kept only as a sanity signal, the realtime
    # model runs its own turn-taking
    # swap openai.realtime for google.realtime to use Gemini Live (one-line change)
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        vad=ctx.proc.userdata["vad"],
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(video_input=True),
    )

    engine = AttentionEngine(ctx.room, agent_identity=saa.agent_identity)
    ctx.add_shutdown_callback(engine.stop)

    gate = {"on": None}

    @engine.on_prediction
    def _(p) -> None:
        # gates audio before it reaches RealtimeModel.push_audio
        want = p.aligned_class == 2
        session.input.set_audio_enabled(want)
        if want != gate["on"]:
            logger.info("GATE %s aligned=%s conf=%.3f",
                        "OPEN" if want else "CLOSE", p.aligned_class, p.confidence)
            gate["on"] = want

    @engine.on_interrupt
    def _(ev) -> None:
        logger.info("interrupt conf=%.3f", ev.confidence)
        # for a realtime model this calls _rt_session.interrupt() for provider-side cancel
        session.interrupt()

    @engine.on_interjection
    async def _(ev) -> None:
        logger.info("interjection reason=%s", ev.reason)
        await session.generate_reply(instructions="Briefly check if the user needs anything")

    # tell SAA when our agent is speaking — arms interrupt, suppresses interjection
    @session.on("agent_state_changed")
    def _(ev) -> None:
        logger.info("agent_state %s->%s", ev.old_state, ev.new_state)
        if ev.new_state == "speaking":
            asyncio.create_task(engine.responding_start())
        elif ev.old_state == "speaking":
            asyncio.create_task(engine.responding_stop())

    @session.on("user_state_changed")
    def _(ev) -> None:
        logger.info("user_state %s->%s", ev.old_state, ev.new_state)

    @session.on("user_input_transcribed")
    def _(ev) -> None:
        logger.info("transcribed final=%s %r", ev.is_final, ev.transcript)

    @session.on("conversation_item_added")
    def _(ev) -> None:
        logger.info("item_added role=%s", getattr(ev.item, "role", "?"))

    @session.on("speech_created")
    def _(ev) -> None:
        logger.info("speech_created source=%s", ev.source)

    @session.on("error")
    def _(ev) -> None:
        logger.error("session error src=%s err=%r", type(ev.source).__name__, ev.error)

    await engine.start()
    logger.info("SAA gating active (agent=%s)", saa.agent_identity)


if __name__ == "__main__":
    cli.run_app(server)
