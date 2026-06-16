# SAA-gated ElevenLabs Conversational AI agent.
#
# ElevenLabs owns its own (sealed) WebRTC room, so SAA cannot join as a hidden
# participant the way it does on LiveKit / Daily. Instead this taps ElevenLabs'
# AudioInterface — the clean both-directions PCM seam in their Python SDK —
# feeds the user mic to the SAA cloud via attenlabs-saa's feed_audio(), and
# gates the agent by only forwarding device-directed audio onward. The SAA
# classifier runs on Attention Labs' infra; no model ships here.
#
# Warmup: SAA's native on_warmup_complete fires once its model is classifying
# for real (inference buffer filled from real audio). We prime() the mic so SAA warms on real
# audio, then hold the agent's greeting until that signal
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

from tui import TerminalUI

# auto-load the shared examples/elevenlabs/.env
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger("elevenlabs-saa")


class SAAFeedAudioInterface(AudioInterface):
    """Wraps an ElevenLabs AudioInterface and wires SAA in two ways:

    - tees the user mic into AttentionClient.feed_audio (so SAA classifies it),
      then forwards to ElevenLabs *only when the gate is open*, only audio
      SAA judged device-directed (class 2) reaches the agent.
    - watches agent TTS on output() to drive saa.mark_responding, so SAA knows
      when the agent itself is speaking (arms barge-in, suppresses noise).

    The SDK's own mic is disabled (enable_audio=False); this is the only audio
    source, fed frame-by-frame.
    """

    def __init__(self, base: AudioInterface, saa: AttentionClient, *,
                 gate: bool = True, responding_idle_ms: int = 500,
                 on_responding=None):
        self._base = base
        self._saa = saa
        self._gate = gate
        # when gating, start closed and open on the first device-directed tick
        self._gate_open = not gate
        self._user_cb = None
        self._primed = False
        # optional fn(bool) notified on responding True/False transitions (TUI)
        self._on_responding = on_responding
        self._responding_idle_s = responding_idle_ms / 1000.0
        self._responding = False
        self._last_output_ts = 0.0
        self._lock = threading.Lock()
        self._wd_stop = threading.Event()
        self._wd: threading.Thread | None = None

    # ── gate control (driven by SAA predictions) ───────────────────────
    def set_gate_open(self, is_open: bool) -> None:
        self._gate_open = is_open

    # ── warmup priming ───────────────────────────────────────────────────
    def prime(self) -> None:
        """Start the mic feeding SAA *before* the ElevenLabs session connects.

        SAA's native warmup (on_warmup_complete) only fires once the model's
        inference buffer has filled from real audio which needs frames
        flowing.
        """
        if self._primed:
            return
        self._primed = True
        self._start_watchdog()
        self._base.start(self._tee)

    def _start_watchdog(self) -> None:
        self._wd_stop.clear()
        self._wd = threading.Thread(target=self._watchdog, name="saa-responding", daemon=True)
        self._wd.start()

    # ── AudioInterface contract ─────────────────────────────────────────
    def start(self, input_callback):
        self._user_cb = input_callback
        if self._primed:
            return  # mic already running from prime(); just attach the callback
        self._start_watchdog()
        self._base.start(self._tee)

    def stop(self):
        self._wd_stop.set()
        if self._wd is not None:
            self._wd.join(timeout=1.0)
            self._wd = None
        self._base.stop()

    def output(self, audio: bytes):
        fire = False
        with self._lock:
            self._last_output_ts = time.monotonic()
            if not self._responding:
                self._responding = True
                fire = True
        if fire:  # notify outside the lock (callbacks may take their own locks)
            self._saa.mark_responding(True)
            self._notify_responding(True)
        self._base.output(audio)

    def interrupt(self):
        self._base.interrupt()
        self._set_responding(False)

    # ── internals ───────────────────────────────────────────────────────
    def _tee(self, audio: bytes):
        # feed SAA continuously (it must see every frame to classify), but only
        # forward to ElevenLabs when SAA says the speech is device-directed
        try:
            self._saa.feed_audio(audio)
        except Exception:
            logger.exception("feed_audio failed")
        if self._gate_open and self._user_cb is not None:
            self._user_cb(audio)

    def _set_responding(self, value: bool):
        fire = False
        with self._lock:
            if not value:
                self._last_output_ts = 0.0
            if self._responding != value:
                self._responding = value
                fire = True
        if fire:
            self._saa.mark_responding(value)
            self._notify_responding(value)

    def _notify_responding(self, value: bool):
        if self._on_responding is not None:
            try:
                self._on_responding(value)
            except Exception:
                logger.exception("on_responding callback raised")

    def _watchdog(self):
        # ElevenLabs has no clean end-of-turn callback in output(); flip
        # responding back off after a short gap with no TTS chunks
        while not self._wd_stop.is_set():
            self._wd_stop.wait(0.1)
            if self._wd_stop.is_set():
                break
            with self._lock:
                idle = (self._responding and self._last_output_ts > 0
                        and time.monotonic() - self._last_output_ts > self._responding_idle_s)
            if idle:
                self._set_responding(False)


def main() -> int:
    saa_api_key = os.environ.get("SAA_API_KEY")
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    agent_id = os.environ.get("ELEVENLABS_AGENT_ID")
    if not (saa_api_key and api_key and agent_id):
        print("set SAA_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_AGENT_ID")
        return 2

    ui = TerminalUI()

    # streaming SDK in feed mode — no self-capture; we feed the ElevenLabs tap.
    # audio-only: ElevenLabs gives no video, so SAA runs on audio alone.
    saa = AttentionClient(token=saa_api_key, enable_audio=False, enable_video=False)
    attn = SAAFeedAudioInterface(DefaultAudioInterface(), saa, gate=True,
                                 on_responding=ui.set_responding)

    warmed = threading.Event()
    warm_ticks = {"n": 0}

    @saa.on_warmup_complete
    def _():
        # model has filled its inference buffer and is producing real
        # predictions. Gate the greeting on this so the agent only speaks once
        # SAA is actually classifying. main() activates the TUI on this signal.
        warmed.set()

    @saa.on_prediction
    def _(ev):
        # the gate: only device-directed speech (class 2) reaches the agent.
        # cls is the server's smoothed/aligned class.
        attn.set_gate_open(ev.cls == 2)
        if warmed.is_set():
            ui.update_prediction(ev.cls, ev.confidence)
            ui.update_gate(ev.cls == 2)
        else:
            warm_ticks["n"] += 1
            if warm_ticks["n"] % 10 == 0:
                ui.log(f"warming up SAA... ({warm_ticks['n']} ticks)")

    @saa.on_interrupt
    def _(ev):
        # informational — ElevenLabs runs its own barge-in detection
        ui.log(f"SAA interrupt (fade={ev.fade_ms}ms conf={ev.confidence:.2f})")

    @saa.on_error
    def _(ev):
        ui.log(f"SAA error: {ev.title}: {ev.message}")

    conversation = Conversation(
        client=ElevenLabs(api_key=api_key),
        agent_id=agent_id,
        requires_auth=True,
        audio_interface=attn,
    )

    # 1) open SAA, 2) prime the mic so SAA warms on real audio, 3) bring up the
    #    dashboard and greet — only once SAA's native warmup fires.
    saa.start()
    attn.prime()
    ui.log("warming up SAA model... (this takes ~10-15s)")
    if not warmed.wait(timeout=20.0):
        ui.log("SAA warmup did not complete within 20s — greeting anyway")
    ui.activate()
    conversation.start_session()
    try:
        conversation.wait_for_session_end()
    except KeyboardInterrupt:
        pass
    finally:
        ui.deactivate()
        print("stopping...")
        conversation.end_session()
        saa.stop()
    return 0


if __name__ == "__main__":
    # logs go to stderr, which the TUI can't suppress, so keep
    # them quiet enough not to tear the frame
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("elevenlabs").setLevel(logging.WARNING)
    logging.getLogger("saa").setLevel(logging.WARNING)
    raise SystemExit(main())
