# SAA + LiveKit Agents

Reference samples that add **Attention Labs SAA** addressee gating to LiveKit voice agents. SAA decides, per utterance, whether speech in the room was meant for the agent — so your STT / LLM / TTS only run on audio the user actually directed at the device.

## The integration shape — hosted bridge

SAA integrates with LiveKit as a **hosted bridge**, not an in-process plugin:

1. Your agent calls `start_attention_session(...)`, which POSTs to the SAA broker.
2. A **hidden participant** joins your LiveKit room, subscribes to the user's audio+video, and runs the classifier on Attention Labs' infrastructure.
3. It publishes events (`prediction`, `vad`, `turn_ready`, `interrupt`, `interjection`) on the `"saa"` LiveKit data topic.
4. Your agent consumes them via `AttentionEngine` and gates the session.

No model weights, no ML dependencies, and no media ever enter your process — the client ([`packages/saa-livekit-client`](../../packages/saa-livekit-client)) is ~50 KB of pure Python.

## Samples

| Sample | Stack | Run |
|---|---|---|
| [`voice_agent_cascaded/`](./voice_agent_cascaded) | Silero VAD → Deepgram STT → OpenAI LLM → Cartesia TTS, SAA-gated | `python src/agent.py dev` |
| [`voice_agent_realtime/`](./voice_agent_realtime) | OpenAI Realtime (speech-to-speech), SAA-gated — the case stock LiveKit can't gate | `python agent.py dev` |
| [`web/`](./web) | Vanilla HTML + `livekit-client` browser client rendering the prediction overlay | `uvicorn token_server:app` |

All target **LiveKit Agents 1.5.x** using the `AgentServer` + `@server.rtc_session()` shape. (`WorkerOptions(entrypoint_fnc=...)` also works on 1.5.x and is the older idiom.)

## Shared environment

Every sample needs the SAA + LiveKit credentials:

```
SAA_API_KEY=                 # Attention Labs hosted bridge
LIVEKIT_URL=                 # wss://your-project.livekit.cloud
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
```

The cascaded sample additionally needs provider keys (`DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `CARTESIA_API_KEY`) unless you switch it to LiveKit's inference gateway; the realtime sample needs `OPENAI_API_KEY`. See each sample's `.env.example`.

## The five lines that integrate SAA

```python
saa = await start_attention_session(api_key=..., livekit_url=..., agent_token=..., room_name=..., participant_identity=...)
engine = AttentionEngine(ctx.room, agent_identity=saa.agent_identity)

@engine.on_prediction
def _(p): session.input.set_audio_enabled(p.aligned_class == 2)   # the gate

@engine.on_interrupt
def _(ev): session.interrupt()                                    # barge-in

@engine.on_interjection
async def _(ev): await session.generate_reply(instructions="...")  # proactive
```

Plus a `@session.on("agent_state_changed")` hook that calls `engine.responding_start()` / `responding_stop()` so SAA knows when your agent is the one speaking — required for interrupt and interjection to fire correctly.

## Requirements & limitations

- The agent's LiveKit URL must be reachable from the SAA cloud.
- Both audio **and** video tracks should be available.
- One target participant per session. Multi-user rooms need one `start_attention_session` call each.
