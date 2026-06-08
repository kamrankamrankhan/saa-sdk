# SAA + Pipecat (on Daily)

Reference samples that add **Attention Labs SAA** addressee gating to [Pipecat](https://github.com/pipecat-ai/pipecat) voice agents running on Daily. SAA decides, per utterance, whether speech in the room was meant for the agent — so your STT / LLM / TTS only run on audio the user actually directed at the device.

## The integration shape — hosted bridge

SAA integrates with Pipecat-on-Daily as a **hosted bridge**, not an in-process Pipecat plugin:

1. Your bot calls `start_attention_session(...)`, which POSTs to the SAA broker.
2. A **hidden participant** joins your Daily room, subscribes to the user's audio+video, and runs the classifier on Attention Labs' infrastructure.
3. It publishes events (`prediction`, `vad`, `turn_ready`, `interrupt`, `interjection`) on the `"saa"` **Daily app-message topic**.
4. Your bot consumes them via `AttentionEngine`, which hooks `@transport.event_handler("on_app_message")` on your `DailyTransport`.
5. Upstream actions (`responding_start`, `set_threshold`, …) queue a `DailyOutputTransportMessageUrgentFrame` onto your bound `PipelineTask`.

No model weights, no ML dependencies, and no media ever enter your process — the client ([`packages/saa-pipecat-client`](../../packages/saa-pipecat-client)) is pure Python.

## Samples

| Sample | Stack | Run |
|---|---|---|
| [`voice_agent_cascaded/`](./voice_agent_cascaded) | Silero VAD -> Deepgram STT -> OpenAI LLM -> Cartesia TTS on Daily, SAA-gated | `python src/agent.py` |
| [`web/`](./web) | Vanilla HTML + `@daily-co/daily-js` browser client rendering the prediction overlay | `uvicorn token_server:app` |

## Prerequisites

### Supported platforms

| Platform | Status | Notes |
|---|---|---|
| **macOS** (arm64 + x86_64) | ✅ supported | daily-python ships wheels for both architectures. |
| **Linux** (x86_64 + aarch64) | ✅ supported | daily-python wheels are built against manylinux_2_28 |
| **Windows** (native) | ❌ not supported | daily-python publishes no Windows wheels and no source distribution. Use **WSL2** and follow the Linux instructions inside your WSL shell. |

### Python 3.11+

pipecat-ai 1.x dropped Python 3.10 support, so 3.11+ is required. `pip install` on 3.10 fails with `Package 'saa-pipecat-client' requires a different Python: 3.10.x not in '>=3.11'`.

| Platform | Install Python 3.11 |
|---|---|
| macOS (Homebrew) | `brew install python@3.11` |
| Debian / Ubuntu | `sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv` |
| Fedora / RHEL | `sudo dnf install -y python3.11` |
| Arch | `sudo pacman -S python311` (AUR; or use `pyenv install 3.11`) |
| WSL2 (Ubuntu) | Same as Debian / Ubuntu — inside the WSL shell, not PowerShell. |
| Any platform | [`pyenv install 3.11`](https://github.com/pyenv/pyenv) is the framework-agnostic option. |

### Activating the venv

The shared shell snippet in each sample's quickstart is:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
```

## Shared environment

Every sample needs the SAA + Daily credentials:

```
SAA_API_KEY=                 # Attention Labs hosted bridge (shared with LiveKit samples)
DAILY_API_KEY=               # Daily.co REST key from dashboard.daily.co
```

The cascaded sample additionally needs:

```
DAILY_ROOM_URL=              # https://your-org.daily.co/sess-xyz
DAILY_BOT_TOKEN=             # meeting token for your Pipecat bot to join the room
DEEPGRAM_API_KEY=
OPENAI_API_KEY=
CARTESIA_API_KEY=
```

See each sample's `.env.example`.

## The five lines that integrate SAA

```python
session = await start_attention_session(
    api_key=SAA_API_KEY, room_url=ROOM_URL,
    agent_token=attention_agent_token(daily_api_key=DAILY_API_KEY, room_name=room_name),
    participant_identity=human_identity,
)
engine = AttentionEngine(transport, agent_identity=session.agent_identity)
engine.bind_task(task)

@engine.on_prediction
def _(p): addressee_gate.suppressed = (p.aligned_class == 1 and p.confidence > 0.7)  # the gate

@engine.on_interrupt
async def _(ev): await task.queue_frames([InterruptionTaskFrame()])                  # barge-in

@engine.on_interjection
async def _(ev): await task.queue_frames([LLMMessagesAppendFrame(messages=[...], run_llm=True)])
```

Plus a `BotSpeakingObserver` FrameProcessor that watches `TTSStartedFrame` / `TTSStoppedFrame` and calls `engine.responding_start()` / `responding_stop()` so SAA knows when your agent is the one speaking — required for interrupt and interjection to fire correctly.

## Requirements & limitations

- The Daily room must be reachable from the SAA cloud (Daily Cloud rooms are public by default).
- Both audio **and** video tracks should be available — the classifier is multimodal.
- One target participant per session. Multi-user rooms need one `start_attention_session` call each.
- `DailyParams(audio_in_user_tracks=True)` is required when your bot shares the room with the human — otherwise the bot's own TTS feeds back as `InputAudioRawFrame`s.
- Identity matching uses the nested `participant["info"]["userName"]` (not the top-level `userName`). The cascaded sample handles this; if you build your own, mirror that lookup.

## Deploy targets

The same agent code runs on:

- **Pipecat Cloud** (`pipecat.daily.co`). Put the auto-provisioned Daily REST key (from Settings -> Daily (WebRTC) tab) in `DAILY_API_KEY`. Your `pk_...` Pipecat Cloud token is for deploy/manage operations, not for SAA at runtime.
- **Modal**, **k8s**, your own VM, anywhere `pipecat-ai[daily]` runs. Get the Daily REST key from `dashboard.daily.co -> Developers` (or from Pipecat Cloud Settings -> Daily (WebRTC) tab if that's where your account lives).

The hidden-bot session always lives on SAA's infrastructure regardless of where your Pipecat bot runs.
