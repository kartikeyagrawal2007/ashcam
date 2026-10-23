"""Control panel: a native window (pywebview) around panel/panel.html."""

import base64
import os
import threading
import time

import cv2

from .away import AwayLoop
from .engine import Engine

PANEL = os.path.join(os.path.dirname(__file__), "panel", "panel.html")


class Api:
    """Methods the panel's JavaScript calls through window.pywebview.api."""

    def __init__(self, engine: Engine, snap, away: AwayLoop):
        self._engine = engine
        self._snap = snap
        self._away = away

    def frame(self) -> str | None:
        latest = self._engine.latest
        if latest is None:
            return None
        small = cv2.resize(latest, (640, 360), interpolation=cv2.INTER_AREA)
        # Mirror the preview like every call app's self-view; the call itself
        # still gets the true image.
        small = cv2.flip(small, 1)
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(jpg.tobytes()).decode("ascii") if ok else None

    def status(self) -> dict:
        s = self._engine.status()
        s["snap_on"] = self._snap is not None
        s["sensitivity"] = self._snap.sensitivity if self._snap else 0.5
        s["double"] = self._snap.double if self._snap else False
        s["threshold"] = self._snap.threshold if self._snap else 1.0
        s["flip"] = self._engine.flip_output
        return s

    def vanish(self) -> None:
        self._engine.toggle_vanish()

    def away(self) -> None:
        self._engine.toggle_away()

    def record(self) -> None:
        self._away.start_recording(countdown_s=3.0, now=time.monotonic())

    def set_sensitivity(self, value) -> None:
        if self._snap:
            self._snap.sensitivity = min(1.0, max(0.0, float(value)))

    def capture_room(self) -> None:
        self._engine.capture_room(countdown_s=5.0)

    def set_mute_mic(self, value) -> None:
        if self._engine.muter is not None:
            self._engine.muter.set_enabled(bool(value))

    def set_flip(self, value) -> None:
        self._engine.flip_output = bool(value)

    def set_double(self, value) -> None:
        if self._snap:
            self._snap.double = bool(value)


def run_panel(engine: Engine, snap, away: AwayLoop, seconds: float | None = None) -> None:
    import webview

    window = webview.create_window(
        "AshCam", url=PANEL, js_api=Api(engine, snap, away),
        width=440, height=820, min_size=(360, 600), background_color="#15171C")
    if seconds:
        threading.Timer(seconds, window.destroy).start()
    webview.start()
