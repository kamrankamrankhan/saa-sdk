# saa-livekit-client

LiveKit client for the [saa](https://attentionlabs.ai).

Adds attention-aware gating, barge-in, and proactive interjection to any LiveKit voice agent.

## Install

```bash
pip install saa-livekit-client
```

## Quickstart — existing voice agent

```python
import os
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import openai
from saa_livekit_client import (
    AttentionEngine, start_attention_session, attention_agent_token,
)


class MyAssistant(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful assistant.")


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    user = await ctx.wait_for_participant()

    # Issue a hidden-participant token for the saa agent.
    agent_token = attention_agent_token(
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
        room_name=ctx.room.name,
    )

    # Summon the saa hosted agent into the room.
    session = await start_attention_session(
        api_key=os.environ["SAA_API_KEY"],
        livekit_url=os.environ["LIVEKIT_URL"],
        agent_token=agent_token,
        room_name=ctx.room.name,
        participant_identity=user.identity,
        attention_config={"frames_per_turn": 3, "vad_threshold": 0.5},
    )

    # Stand up the voice agent.
    voice = AgentSession(llm=openai.realtime.RealtimeModel(voice="alloy"))

    # Wire attention events into the voice agent.
    attention = AttentionEngine(ctx.room, agent_identity=session.agent_identity)

    @attention.on_prediction
    def _(p):
        # Gate the model's mic — only class 2 (talking-to-device) gets through.
        voice.input.set_audio_enabled(p.aligned_class == 2)

    @attention.on_interrupt
    def _(ev):
        voice.interrupt()

    @attention.on_interjection
    async def _(ev):
        await voice.say("Want me to help with something?")

    ctx.add_shutdown_callback(session.stop)
    await attention.start()
    await voice.start(agent=MyAssistant(), room=ctx.room)


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
```

That's the whole integration. Works with cascaded pipelines AND
`RealtimeModel` speech-to-speech

## Greenfield — `build_attention_entrypoint`

For new voice agents that don't have an existing pipeline:

```python
from livekit import agents
from saa_livekit_client import build_attention_entrypoint, TurnReadyEvent

async def handle_turn(event: TurnReadyEvent, ctx: agents.JobContext):
    # event.audio_pcm16 = int16 mono 16 kHz; event.frames = list[TurnFrame]
    response_pcm = await my_llm.respond(event.audio_pcm16, frames=event.frames)
    await publish_response_audio(ctx.room, response_pcm)

entrypoint = build_attention_entrypoint(on_turn=handle_turn)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
```

Environment: `SAA_API_KEY`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_URL`.

## Event types

| Event | Fires | Payload |
|---|---|---|
| `PredictionEvent` | every 250 ms | `raw_class`, `aligned_class` (0/1/2), `confidence`, `source`, `num_faces` |
| `VADEvent` | every 250 ms | `is_speech`, `probability` |
| listening_start / listening_cancelled | state edges | — |
| `TurnReadyEvent` | end of user turn | `audio_pcm16`, `duration`, `frames`, `context` |
| `InterruptEvent` | user barges in during AI playback | `confidence` |
| `InterjectionEvent` | humans went quiet after side-chat | `reason`, `audio_pcm16`, `duration` |
| `ErrorEvent` | out-of-band errors | `code`, `message` |

Classes: `0`=silent, `1`=human-to-human, `2`=human-to-device.

## Upstream actions

```python
await attention.mute()                       # stop feeding mic to processor
await attention.unmute()
await attention.responding_start()           # AI is now speaking
await attention.responding_stop()
await attention.set_threshold(0.65)          # model class-2 confidence threshold
```

These are routed only to the hidden agent (`destination_identities=[...]`)
so they never leak to other room participants.

## Requirements

- Python 3.10+
- LiveKit URL must be publicly reachable from our cloud (no private VPC)
- Audio + video tracks must both be available (the model is multimodal)
- Customer voice agent and hosted attention agent share the same LiveKit room

## License

Proprietary. © Attention Labs.
