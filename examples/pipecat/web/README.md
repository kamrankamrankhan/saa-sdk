# web — SAA + Daily browser demo

A vanilla HTML/JS client that joins a Daily room, publishes cam+mic, and renders SAA's prediction stream as a live overlay — no build step, no framework. The address-decision flips the pill green only when you're talking *to* the device.

This is the Daily/Pipecat sibling of [`examples/livekit/web`](../../livekit/web). Same UI, same SAA event shapes, different transport underneath.

## Files

| File | What it is |
|---|---|
| [`token_server.py`](./token_server.py) | tiny FastAPI endpoint — creates an ephemeral Daily room, mints a user meeting token, and summons the hidden SAA agent. Serves the static files too. |
| [`index.html`](./index.html) | UI shell (prediction pill, confidence bar, VAD/faces, video). |
| [`app.js`](./app.js) | call-object Daily client, publishes tracks, consumes the `"saa"` app-message topic. |
| [`turn-parser.js`](./turn-parser.js) | decodes the binary turn payload (PCM16 + JPEGs). |
| [`styles.css`](./styles.css) | minimal styling. |

## Run

```bash
cd examples/pipecat/web
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
pip install -e ../../../packages/saa-pipecat-client   # local dev against this repo

cp .env.example .env   # then fill in SAA_API_KEY and DAILY_API_KEY
set -a && source .env && set +a

uvicorn token_server:app --port 8000
# open http://localhost:8000 and click Start
```

## How the overlay is wired

SAA events arrive as JSON on Daily's **app-message channel** under the `"saa"` topic. The integration surface in `app.js` is two functions, identical to the LiveKit demo:

- `renderPrediction(msg)` — reads `msg.aligned_class` (0/1/2), `msg.confidence`, `msg.num_faces`
- `renderVAD(msg)` — reads `msg.is_speech`

The bot publishes envelopes like `{topic:"saa", type:"prediction", ...}`. The Daily `app-message` event wraps that as `{ data, fromId }`, so `app.js` destructures `data` and filters on `data.topic === "saa"`.

### turn_ready chunk reassembly

Daily has no byte-stream primitive, so the per-turn binary blob (PCM16 + JPEGs, LiveKit-identical layout) is base64-encoded and split across multiple app messages — see the [`daily-integration.md` spec](../../../docs/) for the wire shape. `app.js` keeps a small `pending` map keyed on `stream_id`:

1. A `turn_ready` (or `interjection`) envelope arrives with `total_chunks` + `byte_len` — start an empty buffer.
2. Each subsequent `turn_chunk` carries a base64 slice + `index`; store it.
3. Once all chunks are gathered, concat them, call `parseTurnPayload(buf)`, and log the result.

The reassembly map is capped at 10 in-flight streams; oldest is dropped on overflow.

This demo only **logs** the decoded PCM + frame count — that's where you'd forward the captured turn audio to your own STT or recorder. For an actual voice agent that responds, see [`../voice_agent_cascaded`](../voice_agent_cascaded).

## What this demo does NOT do

- **No voice agent.** It only renders SAA's gating signal. If you want the agent to actually talk back, run `voice_agent_cascaded` alongside or in place of this.
- **No upstream actions.** `mute` / `responding_start` / `set_threshold` aren't wired here — they belong on the voice-agent side, gated by your TTS state.

## Production warning

`token_server.py` is **dev-only**: it has open CORS, creates a new billed Daily room on every `/session` hit, and starts a billed SAA session at the same time. For production you need auth on `/session`, rate limiting, a real room/identity policy, and your customers should mint the hidden-bot token using *their* Daily API key — not ours. The SAA API key must always stay server-side; the browser only ever receives the Daily room URL, user token, and the agent identity.
