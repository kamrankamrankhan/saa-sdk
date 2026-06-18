# SAA-gated ElevenLabs Conversational AI agent — debug/logging build.
#
# ElevenLabs always gets a continuous 16 kHz stream: the user's real mic audio
# while SAA says it's device-directed (held open briefly so a pause doesn't chop
# it and so ElevenLabs can endpoint), and silence otherwise — so the agent never
# answers side conversations, while its own VAD still endpoints and replies.
# While muted it also pings ElevenLabs' reset-turn-timeout so the agent doesn't re-prompt
# Logs prediction / responding / send state so the gating can be observed (no TUI).
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
    """Tees the mic into SAA and feeds ElevenLabs a continuous stream: real audio
    while device-directed, silence otherwise.

    - Gate opens on class-2 (device-directed) and closes at once on class-1
      (human-directed). A class-0 (silence) dip is debounced for
      `close_debounce_ticks` ticks so a pause mid-utterance doesn't chop the turn.

    - `responding` is tracked by agent-TTS *playback* duration, not output() calls:
      DefaultAudioInterface queues output() instantly but plays on a background
      thread, so output()-idle would flip responding off mid-playback and the
      agent's echo (no AEC) would leak back as user audio.

    - While the gate is shut, a keepalive ping (`bind_keepalive`) resets ElevenLabs'
      turn timer so its no-input timeout never re-prompts during silence/side-talk.
      turn_timeout caps at 30s and isn't per-session overridable, so the reset
      event is the only way to hold the turn open indefinitely. 
    """

    def __init__(self, base: AudioInterface, saa: AttentionClient, *,
                 close_debounce_ticks: int = 4, responding_tail_ms: int = 300,
                 keepalive_s: float = 5.0):
        self._base = base
        self._saa = saa
        self._close_debounce = close_debounce_ticks
        self._gate_open = False         # start closed; opens on first device-directed tick
        self._misses = 0                # consecutive non-device-directed ticks
        self._user_cb = None
        self._primed = False
        self._tail_s = responding_tail_ms / 1000.0
        self._responding = False
        self._play_until = 0.0          # monotonic deadline: agent audio plays until here
        self._sending_real = False      # last send state (real vs silence) for transition logs
        self._keepalive_s = keepalive_s # ping period to reset ElevenLabs' turn timer
        self._keepalive_cb = None       # set by bind_keepalive once the session is live
        self._last_keepalive = 0.0
        self._lock = threading.Lock()
        self._wd_stop = threading.Event()
        self._wd: threading.Thread | None = None

    @property
    def responding(self) -> bool:
        return self._responding

    @property
    def gate_open(self) -> bool:
        return self._gate_open

    def bind_keepalive(self, cb) -> None:
        # call once the ElevenLabs session is live: cb resets its turn timer so the
        # no-input timeout never re-prompts while the gate is shut.
        self._keepalive_cb = cb

    def update_gate(self, cls: int) -> None:
        # open on device-directed (2); human-directed (1) closes at once; a
        # class-0 (silence) dip is debounced so a pause mid-utterance doesn't chop.
        if cls == 2:
            self._misses = 0
            self._gate_open = True
        elif cls == 1:
            self._gate_open = False          # human-directed — never the device
        elif self._gate_open:                # cls == 0: silence, maybe a pause
            self._misses += 1
            if self._misses >= self._close_debounce:
                self._gate_open = False

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
        # always feed SAA so it classifies every frame
        try:
            self._saa.feed_audio(audio)
        except Exception:
            log.exception("feed_audio failed")
        if self._user_cb is None:
            return  # priming: SAA only, ElevenLabs not connected yet
        # continuous stream to ElevenLabs: real audio only when device-directed
        # and the agent isn't speaking; silence otherwise (so it never hears side
        # talk or its own echo, but its VAD still gets a clean speech→silence edge).
        send_real = self._gate_open and not self._responding
        if send_real != self._sending_real:
            self._sending_real = send_real
            log.info("SEND %s  (gate=%s resp=%s)",
                     "real " if send_real else "muted", self._gate_open, self._responding)
        if send_real:
            # defer the keepalive to send
            self._last_keepalive = time.monotonic()
        self._user_cb(audio if send_real else bytes(len(audio)))

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
            now = time.monotonic()
            with self._lock:
                done = (self._responding and now >= self._play_until + self._tail_s)
            if done:
                self._set_responding(False, "playback-done")
            # hold ElevenLabs' turn open while muted so it doesn't re-prompt on silence
            if (self._keepalive_cb and not self._gate_open and not self._responding
                    and now - self._last_keepalive >= self._keepalive_s):
                self._last_keepalive = now
                try:
                    self._keepalive_cb()
                except Exception:
                    pass


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
    attn = SAAFeedAudioInterface(DefaultAudioInterface(), saa)

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
        attn.update_gate(ev.cls)               # 2 opens, 1 closes now, 0 debounced
        resp, gate = attn.responding, attn.gate_open
        log.info("PRED cls=%d conf=%.2f src=%s | resp=%s gate=%s send=%s",
                 ev.cls, ev.confidence or 0.0, ev.source,
                 resp, "open" if gate else "shut",
                 "real" if (gate and not resp) else "muted")
        if ev.cls == 2 and resp:
            log.info("  ^ cls=2 while agent SPEAKING (echo?) — muted")

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
    # Hold ElevenLabs' turn open during silence/side-talk: its turn_timeout caps at
    # 30s and isn't overridable, so we reset it
    conversation.start_session()
    attn.bind_keepalive(conversation.register_user_activity)
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
