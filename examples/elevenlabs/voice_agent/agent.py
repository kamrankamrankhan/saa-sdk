# SAA-gated ElevenLabs Conversational AI agent — debug/logging build.
#
# SAA decides which speech is addressed to the agent; only that audio is
# forwarded to ElevenLabs. This build logs prediction + responding + forward
# state so the gating can be observed (no TUI).
import argparse
import logging
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import AudioInterface, Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

from saa import AttentionClient

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

log = logging.getLogger("elevenlabs-saa")

# DefaultAudioInterface plays agent TTS at 16 kHz mono PCM16 → 32000 bytes/sec.
OUTPUT_BYTES_PER_SEC = 16000 * 2


class SAAFeedAudioInterface(AudioInterface):
    """Tees the mic into SAA (feed_audio) and forwards to ElevenLabs only when
    SAA says device-directed AND the agent isn't speaking.

    `responding` is tracked by agent-TTS *playback* duration, not by output()
    calls: DefaultAudioInterface queues output() instantly but plays on a
    background thread, so an output()-idle watchdog flips responding off while
    the agent is still audible — and its echo (no AEC) leaks back as user audio.
    """

    def __init__(self, base: AudioInterface, saa: AttentionClient, *,
                 gate: bool = True, responding_tail_ms: int = 300):
        self._base = base
        self._saa = saa
        self._gate_open = not gate
        self._user_cb = None
        self._primed = False
        self._tail_s = responding_tail_ms / 1000.0
        self._responding = False
        self._play_until = 0.0          # monotonic deadline: agent audio plays until here
        self._fwd_on = False            # last effective-forward state (for transition logs)
        self._lock = threading.Lock()
        self._wd_stop = threading.Event()
        self._wd: threading.Thread | None = None

    @property
    def responding(self) -> bool:
        return self._responding

    def set_gate_open(self, is_open: bool) -> None:
        self._gate_open = is_open

    # ── lifecycle ─────────────────────────────────────────────────────────
    def prime(self) -> None:
        # start the mic feeding SAA before the ElevenLabs session connects
        if self._primed:
            return
        self._primed = True
        self._start_watchdog()
        self._base.start(self._tee)

    def _start_watchdog(self) -> None:
        self._wd_stop.clear()
        self._wd = threading.Thread(target=self._watchdog, name="saa-responding", daemon=True)
        self._wd.start()

    def start(self, input_callback):
        self._user_cb = input_callback
        if self._primed:
            return
        self._start_watchdog()
        self._base.start(self._tee)

    def stop(self):
        self._wd_stop.set()
        if self._wd is not None:
            self._wd.join(timeout=1.0)
            self._wd = None
        self._base.stop()

    # ── agent TTS ─────────────────────────────────────────────────────────
    def output(self, audio: bytes):
        now = time.monotonic()
        first = False
        with self._lock:
            # extend playback deadline by this chunk's duration (queue draining)
            self._play_until = max(self._play_until, now) + len(audio) / OUTPUT_BYTES_PER_SEC
            if not self._responding:
                self._responding = True
                first = True
        if first:
            self._saa.mark_responding(True)
            log.info("RESP -> SPEAKING")
        self._base.output(audio)

    def interrupt(self):
        self._base.interrupt()          # clears the playback queue immediately
        with self._lock:
            self._play_until = time.monotonic()
        self._set_responding(False, "interrupt")

    # ── internals ─────────────────────────────────────────────────────────
    def _tee(self, audio: bytes):
        # always feed SAA; forward to the agent only when device-directed AND
        # the agent isn't speaking (else its own playback echoes back).
        try:
            self._saa.feed_audio(audio)
        except Exception:
            log.exception("feed_audio failed")
        forward = self._gate_open and not self._responding and self._user_cb is not None
        if forward != self._fwd_on:
            self._fwd_on = forward
            log.info("FWD %s  (gate=%s resp=%s)",
                     "ON " if forward else "OFF", self._gate_open, self._responding)
        if forward:
            self._user_cb(audio)

    def _set_responding(self, value: bool, reason: str):
        fire = False
        with self._lock:
            if self._responding != value:
                self._responding = value
                fire = True
        if fire:
            self._saa.mark_responding(value)
            if not value:
                log.info("RESP -> idle  (%s)", reason)

    def _watchdog(self):
        while not self._wd_stop.wait(0.1):
            with self._lock:
                done = (self._responding and time.monotonic() >= self._play_until + self._tail_s)
            if done:
                self._set_responding(False, "playback-done")


def main() -> int:
    saa_api_key = os.environ.get("SAA_API_KEY")
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    agent_id = os.environ.get("ELEVENLABS_AGENT_ID")
    if not (saa_api_key and api_key and agent_id):
        print("set SAA_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_AGENT_ID")
        return 2

    # class-2 (device-directed) confidence threshold: higher = stricter gate.
    # Set at startup here; change live anytime with saa.set_threshold(v).
    threshold = float(os.environ.get("SAA_CLASS2_THRESHOLD", "0.7"))

    saa = AttentionClient(token=saa_api_key, enable_audio=False, enable_video=False,
                          initial_threshold=threshold)
    attn = SAAFeedAudioInterface(DefaultAudioInterface(), saa, gate=True)

    warmed = threading.Event()

    @saa.on_warmup_complete
    def _():
        log.info("WARMUP complete — SAA predicting")
        warmed.set()

    @saa.on_config
    def _(ev):
        log.info("CONFIG class2_threshold=%.2f", ev.model_class2_threshold)

    @saa.on_prediction
    def _(ev):
        open_gate = ev.cls == 2
        attn.set_gate_open(open_gate)
        resp = attn.responding
        log.info("PRED cls=%d conf=%.2f src=%s | resp=%s gate=%s fwd=%s",
                 ev.cls, ev.confidence or 0.0, ev.source,
                 resp, "open" if open_gate else "shut",
                 "Y" if (open_gate and not resp) else "N")
        if open_gate and resp:
            log.info("  ^ OVERLAP: cls=2 while agent SPEAKING (echo?) — held back")

    @saa.on_interrupt
    def _(ev):
        log.info("SAA interrupt conf=%.2f", ev.confidence)

    @saa.on_error
    def _(ev):
        log.warning("SAA error: %s — %s", ev.title, ev.message)

    conversation = Conversation(
        client=ElevenLabs(api_key=api_key),
        agent_id=agent_id,
        requires_auth=True,
        audio_interface=attn,
        callback_agent_response=lambda r: log.info("AGENT: %s", r),
        callback_user_transcript=lambda t: log.info("USER:  %s", t),
    )

    saa.start()
    attn.prime()
    log.info("warming up SAA model... (~10-15s)")
    if not warmed.wait(timeout=20.0):
        log.warning("SAA warmup did not complete within 20s — greeting anyway")
    conversation.start_session()
    log.info("session started — Ctrl+C to end")
    try:
        conversation.wait_for_session_end()
    except KeyboardInterrupt:
        pass
    finally:
        conversation.end_session()
        saa.stop()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("elevenlabs").setLevel(logging.WARNING)
    logging.getLogger("saa").setLevel(logging.WARNING)
    raise SystemExit(main())
