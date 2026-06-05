"""Embedded SAA-gated voice agent for the web demo.

A reusable `run_voice_agent(...)` coroutine that joins a Daily room as the
talkback agent and lets SAA gate STT. The standalone reference shape lives in
`examples/pipecat/voice_agent_cascaded/src/agent.py` — this is the same logic
factored into a function that token_server.py can spawn per /session call so a
tester gets the full overlay + talkback experience from a single `uvicorn`
process.

Differences from the standalone agent:
  * Takes the SAA `agent_identity` as an argument — token_server.py already
    called `start_attention_session(...)` before this coroutine runs, so the
    agent doesn't re-mint a session.
  * Takes provider keys as arguments rather than reading env directly, so
    failures are caught at startup not in the middle of a /session call.
  * No `__main__` entrypoint — invoked via `asyncio.create_task` from
    token_server.py.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

# Pipecat 1.x canonical import paths. Same set as the standalone agent —
# see voice_agent_cascaded/src/agent.py for the rationale.
from pipecat.transports.daily.transport import DailyTransport, DailyParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionTaskFrame,
    LLMMessagesAppendFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)

from saa_pipecat_client import AttentionEngine

logger = logging.getLogger("web.voice_agent")


class _AddresseeGate(FrameProcessor):
    """Drops user audio when SAA says the user is talking to another human."""

    def __init__(self) -> None:
        super().__init__()
        self.suppressed = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if self.suppressed and isinstance(frame, InputAudioRawFrame):
            return
        await self.push_frame(frame, direction)


class _BotSpeakingObserver(FrameProcessor):
    """Bridges TTS lifecycle frames to SAA's responding_start/stop."""

    def __init__(self) -> None:
        super().__init__()
        self._engine: Optional[AttentionEngine] = None

    def bind_engine(self, engine: AttentionEngine) -> None:
        self._engine = engine

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if self._engine is not None:
            if isinstance(frame, TTSStartedFrame):
                asyncio.create_task(self._engine.responding_start())
            elif isinstance(frame, TTSStoppedFrame):
                asyncio.create_task(self._engine.responding_stop())
        await self.push_frame(frame, direction)


async def run_voice_agent(
    *,
    room_url: str,
    bot_token: str,
    saa_agent_identity: str,
    openai_api_key: str,
    deepgram_api_key: str,
    cartesia_api_key: str,
    system_prompt: str = "You are a helpful voice assistant. Keep replies short and natural.",
) -> None:
    """Join `room_url` as a talkback voice agent and let SAA gate STT.

    Returns when the human leaves the room or the pipeline task is cancelled.
    Exceptions inside the pipeline propagate; the caller (token_server.py)
    logs them and reaps the task.
    """
    transport = DailyTransport(
        room_url,
        bot_token,
        "SAA Voice Agent",
        DailyParams(
            audio_in_enabled=True,
            audio_in_user_tracks=True,
            video_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
        ),
    )

    stt = DeepgramSTTService(api_key=deepgram_api_key, model="nova-3")
    llm = OpenAILLMService(api_key=openai_api_key, model="gpt-4o-mini")
    tts = CartesiaTTSService(api_key=cartesia_api_key, model="sonic-2")

    context = LLMContext(messages=[{"role": "system", "content": system_prompt}])
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    addressee_gate = _AddresseeGate()
    bot_speaking = _BotSpeakingObserver()

    pipeline = Pipeline(
        [
            transport.input(),
            addressee_gate,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            bot_speaking,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
    task = PipelineTask(pipeline)

    # SAA hosted session was already started by token_server.py; we only need
    # its agent_identity to construct the engine.
    engine = AttentionEngine(transport, agent_identity=saa_agent_identity)
    engine.bind_task(task)
    bot_speaking.bind_engine(engine)

    @engine.on_prediction
    def _(p) -> None:
        addressee_gate.suppressed = p.aligned_class == 1 and p.confidence > 0.7

    @engine.on_interrupt
    async def _(ev) -> None:
        logger.info("SAA interrupt fired (confidence=%.2f)", ev.confidence)
        await task.queue_frames([InterruptionTaskFrame()])
        await engine.responding_stop()

    @engine.on_interjection
    async def _(ev) -> None:
        logger.info("SAA interjection fired (reason=%s)", ev.reason)
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "system",
                            "content": "Briefly offer to help in one short sentence.",
                        }
                    ],
                    run_llm=True,
                ),
            ]
        )

    @transport.event_handler("on_first_participant_joined")
    async def _on_first_participant_joined(transport_, participant):
        # Engine is already bound; start it now so we don't miss the first
        # `started` envelope from the SAA bot (which may have joined first).
        await engine.start()
        logger.info(
            "SAA gating active for room=%s (saa_agent=%s)",
            room_url, saa_agent_identity,
        )

    @transport.event_handler("on_participant_left")
    async def _on_participant_left(transport_, participant, reason):
        # When the human leaves, drain and exit. The SAA hosted session is
        # owned by token_server.py — it stays alive until token_server.py
        # explicitly tears it down (or the broker reaps it on idle).
        user_name = participant.get("info", {}).get("userName")
        if user_name and user_name.startswith("user-"):
            logger.info(
                "human %s left (%s); shutting down voice agent for room=%s",
                user_name, reason, room_url,
            )
            try:
                await engine.stop()
            except Exception:
                logger.exception("engine.stop failed")
            await task.cancel()

    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        try:
            await engine.stop()
        except Exception:
            pass
