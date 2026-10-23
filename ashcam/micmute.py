"""Mute the Mac's microphone while you're vanished, for every app at once.

AshCam can't reach into WhatsApp or FaceTime to press their mute button, but
every app records through the system input, so turning the input volume to
0 silences you everywhere. The level you had is saved to a small file first,
so if AshCam is killed while you're gone the next start puts it back.
"""

import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

STATE_FILE = os.path.expanduser("~/.ashcam-mic-level")
MIC_ERROR = ("Couldn't mute the microphone. Mute in your call app, or turn off "
             "“Mute my mic while I’m gone”.")


def _osascript(script: str) -> str:
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True,
                         timeout=5, check=True)
    return out.stdout.strip()


class MicMuter:
    def __init__(self, run: Callable[[str], str] = _osascript, state_file: str = STATE_FILE,
                 sync: bool = False):
        self._run = run
        self._state_file = state_file
        # osascript takes ~100 ms; run it off the video thread, in order.
        self._pool = None if sync else ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self.enabled = True
        self.muted = False
        self.error: str | None = None
        self._saved: int | None = None

    def _do(self, fn) -> None:
        if self._pool is None:
            fn()
        else:
            self._pool.submit(fn)

    def _safe(self, fn, clears_error: bool = False) -> None:
        try:
            fn()
            if clears_error:
                self.error = None
        except Exception:
            self.error = MIC_ERROR

    def mute(self) -> None:
        with self._lock:
            if not self.enabled or self.muted:
                return
            self.muted = True
        self._do(lambda: self._safe(self._mute_now, clears_error=True))

    def unmute(self) -> None:
        with self._lock:
            if not self.muted:
                return
            self.muted = False
        self._do(lambda: self._safe(self._unmute_now))

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.unmute()

    def restore_leftover(self) -> None:
        """Put back a level saved by a run that ended while muted."""
        if not os.path.exists(self._state_file):
            return

        def restore():
            with open(self._state_file) as f:
                level = int(f.read().strip() or 0)
            if level > 0:
                self._run(f"set volume input volume {level}")
            os.remove(self._state_file)

        self._safe(restore)

    def close(self) -> None:
        self.unmute()
        if self._pool is not None:
            self._pool.shutdown(wait=True)

    def _mute_now(self) -> None:
        level = int(float(self._run("input volume of (get volume settings)") or 0))
        if level > 0:
            self._saved = level
            with open(self._state_file, "w") as f:
                f.write(str(level))
        self._run("set volume input volume 0")

    def _unmute_now(self) -> None:
        if self._saved:
            self._run(f"set volume input volume {self._saved}")
        if os.path.exists(self._state_file):
            os.remove(self._state_file)
