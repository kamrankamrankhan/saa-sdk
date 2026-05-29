# web — SAA + LiveKit browser demo

A vanilla HTML/JS client that joins a LiveKit room, publishes cam+mic, and renders SAA's prediction stream as a live overlay — no build step, no framework. The address-decision flips the pill green only when you're talking *to* the device.

## Files

| File | What it is |
|---|---|
| [`token_server.py`](./token_server.py) | tiny FastAPI endpoint — mints a browser join token and summons the hidden SAA agent. Serves the static files too. |
| [`index.html`](./index.html) | UI shell (prediction pill, confidence bar, VAD/faces, video). |
| [`app.js`](./app.js) | connects to the room, publishes tracks, consumes the `"saa"` data topic. |
| [`turn-parser.js`](./turn-parser.js) | decodes the binary turn payload (PCM16 + JPEGs). |
| [`styles.css`](./styles.css) | minimal styling. |

## Run

```bash
cd examples/livekit/web
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
pip install -e ../../../packages/saa-livekit-client   # local dev against this repo

cp .env.example .env   # then fill in the keys
set -a && source .env && set +a

uvicorn token_server:app --port 8000
# open http://localhost:8000 and click Start
```

## How the overlay is wired

SAA events arrive as JSON on the `"saa"` data topic. The integration surface is two functions in `app.js`:

- `renderPrediction(msg)` — reads `msg.aligned_class` (0/1/2), `msg.confidence`, `msg.num_faces`
- `renderVAD(msg)` — reads `msg.is_speech`

`turn_ready` / `interjection` arrive as a JSON envelope plus a binary byte stream on the same topic; `onByteStream` reassembles it and `parseTurnPayload` decodes the PCM + frames. This demo just logs them — that's where you'd forward the captured turn audio to your own backend.

## Production warning

`token_server.py` is **dev-only**: it has open CORS, mints a token for any `room`/`identity`, and starts a billed SAA session on every `/token` hit. For production you need auth on `/token`, rate limiting, and a real room/identity policy. The SAA API key must always stay server-side — the browser only ever receives the LiveKit join token and the agent identity.
