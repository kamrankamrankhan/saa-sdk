# voice_agent — SAA-gated ElevenLabs Conversational AI

An [ElevenLabs Conversational AI](https://elevenlabs.io/docs/eleven-agents/overview) agent with **Attention Labs SAA** addressee gating wired on top, via the streaming SDK's `feed_audio` ingestion.

## The integration

Single file ([`agent.py`](./agent.py)). The moving parts:

- `AttentionClient(token=..., enable_audio=False, enable_video=False)` -> streaming SDK in **feed mode**: it opens the cloud WebSocket but captures nothing itself.
- `SAAFeedAudioInterface(DefaultAudioInterface(), saa)` -> wraps ElevenLabs' audio interface. Its mic tee calls `saa.feed_audio(chunk)` for every frame, then forwards to the agent **only when the gate is open**.
- `@saa.on_prediction` → `attn.set_gate_open(ev.cls == 2)` -> **the gate**. Direct analog of the LiveKit realtime sample's `session.input.set_audio_enabled(p.aligned_class == 2)`: only device-directed speech reaches the agent.
- `output()` / `interrupt()` → `saa.mark_responding(True/False)` (via a short idle watchdog, since ElevenLabs has no clean end-of-turn callback) — so SAA knows when the agent itself is speaking.

```python
saa = AttentionClient(token=ATTENLABS_TOKEN, enable_audio=False, enable_video=False)
attn = SAAFeedAudioInterface(DefaultAudioInterface(), saa, gate=True)

@saa.on_prediction
def _(ev): attn.set_gate_open(ev.cls == 2)        # the gate

conversation = Conversation(client=ElevenLabs(...), agent_id=..., requires_auth=True, audio_interface=attn)
saa.start()                                       # feed_audio live before the mic tee starts
conversation.start_session()
```

## Quickstart

```bash
cd examples/elevenlabs
cp .env.example .env     # fill ATTENLABS_TOKEN, ELEVENLABS_API_KEY, ELEVENLABS_AGENT_ID

cd voice_agent
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ../../../packages/saa-py              # local dev against this repo
pip install -r requirements.txt
python agent.py
```

Talk to it. The agent answers only when you're addressing it; speech you direct at another person in the room never reaches the model.

## Cost note

SAA streaming is billed per session-minute; the ElevenLabs agent is billed by ElevenLabs.
