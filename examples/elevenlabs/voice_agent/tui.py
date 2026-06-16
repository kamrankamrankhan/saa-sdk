# Minimal terminal dashboard for the SAA-gated ElevenLabs demo.
from __future__ import annotations

import collections
import shutil
import sys
import threading
import time


class _SuppressableStdout:
    # While suppress is True, only direct_write() reaches the terminal, so other
    # threads' prints can't tear the frame.
    def __init__(self, original):
        self._original = original
        self._lock = threading.Lock()
        self.suppress = False

    def write(self, text):
        if self.suppress:
            return len(text)
        with self._lock:
            return self._original.write(text)

    def flush(self):
        self._original.flush()

    def direct_write(self, text):
        with self._lock:
            self._original.write(text)
            self._original.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()

    def __getattr__(self, name):
        return getattr(self._original, name)


class TerminalUI:
    """In-place MODE / BUFFER / GATE / AGENT status frame. No-ops without a TTY."""

    _PRED = {0: "NOT TALKING", 1: "TALKING TO HUMAN", 2: "TALKING TO COMPUTER"}
    _MODE, _BUFFER, _GATE, _AGENT = range(4)

    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._cols = 80
        self._row0 = 6
        self._lines: list[str] = []
        self._buf: collections.deque = collections.deque(maxlen=12)
        self._responding = False
        self._last_cls = 0
        self._tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        if self._tty and not isinstance(sys.stdout, _SuppressableStdout):
            self._out = _SuppressableStdout(sys.stdout)
            sys.stdout = self._out
        else:
            self._out = sys.stdout

    @property
    def active(self) -> bool:
        return self._active

    def activate(self) -> None:
        if self._active or not self._tty:
            return
        with self._lock:
            self._lines = [""] * 4
            self._active = True
            self._out.suppress = True
            self._draw()
        self.update_gate(False)
        self._render_agent()

    def deactivate(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._out.suppress = False
            self._out.direct_write(f"\033[{self._row0 + len(self._lines) + 2};1H\n")

    def update_prediction(self, cls: int, conf: float | None) -> None:
        self._last_cls = cls
        self._buf.append(cls)
        self._set(self._MODE, f"MODE   : {self._PRED.get(cls, '?')} ({(conf or 0) * 100:.0f}%)")
        self._set(self._BUFFER, f"BUFFER : {list(self._buf)}")
        self._render_agent()

    def update_gate(self, is_open: bool) -> None:
        self._set(self._GATE, f"GATE   : {'OPEN' if is_open else 'CLOSED'}")

    def set_responding(self, responding: bool) -> None:
        self._responding = responding
        self._render_agent()

    def log(self, msg: str) -> None:
        # Prints before the frame is up (or headless); dropped once active.
        if not self._active:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # ── internals ──────────────────────────────────────────────────────────

    def _render_agent(self) -> None:
        state = "speaking" if self._responding else ("listening" if self._last_cls == 2 else "idle")
        self._set(self._AGENT, f"AGENT  : {state}")

    def _bordered(self, t: str) -> str:
        w = self._cols - 6
        t = t[:w]
        return "║  " + t + " " * (w - len(t)) + "  ║"

    def _draw(self) -> None:
        try:
            self._cols = max(shutil.get_terminal_size().columns, 60)
        except (OSError, ValueError):
            self._cols = 80
        bar = "═" * (self._cols - 2)
        head = [
            "\033[2J\033[H",
            f"╔{bar}╗",
            self._bordered("SAA × ElevenLabs — addressee-gated voice agent"),
            self._bordered("Ctrl+C to stop"),
            f"╠{bar}╣",
        ]
        self._row0 = len(head) + 1
        head += [self._bordered("")] * len(self._lines)
        head.append(f"╚{bar}╝")
        self._out.direct_write("\n".join(head))

    def _set(self, i: int, msg: str) -> None:
        with self._lock:
            if not self._active:
                return
            self._lines[i] = msg
            w = self._cols - 6
            t = msg[:w]
            row = self._row0 + i
            bottom = self._row0 + len(self._lines) + 1
            self._out.direct_write(f"\033[{row};2H  {t}{' ' * (w - len(t))}  \033[{bottom};1H")
