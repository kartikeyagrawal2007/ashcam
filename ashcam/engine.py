"""Frame loop: camera -> person mask -> composite per state -> virtual camera."""

import threading
import time
from typing import Callable

import cv2
import numpy as np

from .away import AwayLoop
from .background import BackgroundPlate
from .dust import DustEffect
from .state import State, StateMachine

OUT_SIZE = (1280, 720)
XFADE_S = 0.2
_GROW = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # at 1/4 res: ~9 px each side

VCAM_HELP = ("OBS Virtual Camera isn't available. Open OBS, click Start Virtual Camera, "
             "then Stop, and quit OBS. If it still fails, turn it on in System Settings → "
             "General → Login Items & Extensions → Camera Extensions.")
CAMERA_HELP = ("Can't read the camera. Allow your terminal app in System Settings → "
               "Privacy & Security → Camera, then restart AshCam.")


def feather(mask: np.ndarray) -> np.ndarray:
    """Grow the person mask a little and soften it, so no outline is left behind."""
    h, w = mask.shape
    small = cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(cv2.dilate(small, _GROW), (5, 5), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def composite(frame: np.ndarray, plate: np.ndarray, hide: np.ndarray) -> np.ndarray:
    """Live frame everywhere, the remembered room where `hide` is 1."""
    hide = np.clip(hide, 0, 1).astype(np.float32)
    return cv2.blendLinear(frame, plate, 1 - hide, hide)


class Engine:
    def __init__(self, camera, segmenter, away: AwayLoop, sm: StateMachine,
                 sink: Callable[[np.ndarray], None] | None = None):
        self.camera = camera
        self.segmenter = segmenter
        self.away = away
        self.sm = sm
        self.sink = sink
        self.snap = None          # SnapDetector, set by main for the level meter
        self.latest: np.ndarray | None = None
        self.vcam_ok = False
        self.error: str | None = None
        self._plate: BackgroundPlate | None = None
        self._fx: DustEffect | None = None
        self._fx_mask: np.ndarray | None = None
        self._xfade_from: np.ndarray | None = None
        self._xfade_start = 0.0
        self._was_away = False
        self._pending: list[str] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    # Called from hotkey, audio and UI threads; applied on the next frame.
    def toggle_vanish(self) -> None:
        with self._lock:
            self._pending.append("vanish")

    def toggle_away(self) -> None:
        with self._lock:
            self._pending.append("away")

    def status(self) -> dict:
        rec, left = self.away.status(time.monotonic())
        return {
            "state": self.sm.state.value,
            "away": self.sm.away,
            "away_rec": rec,
            "away_rec_left": round(left, 1),
            "has_clip": self.away.has_clip,
            "snap_level": float(self.snap.level) if self.snap else 0.0,
            "vcam_ok": self.vcam_ok,
            "error": self.error,
        }

    def _apply_pending(self, frame: np.ndarray, mask: np.ndarray, now: float) -> None:
        with self._lock:
            pending, self._pending = self._pending, []
        for action in pending:
            if action == "away":
                self.sm.toggle_away()
                self.away.reset_playhead()
            elif self.sm.toggle_vanish(now):
                # Vanishing crumbles the person as they are now; returning
                # reassembles the person who is sitting there now.
                self._fx = DustEffect(frame, mask, seed=int(now * 1000) % 2**32)
                self._fx_mask = mask

    def process(self, frame: np.ndarray | None, now: float) -> np.ndarray:
        if frame is None:
            if self.latest is None:
                self.latest = np.zeros((OUT_SIZE[1], OUT_SIZE[0], 3), np.uint8)
            return self.latest

        mask = self.segmenter.mask(frame)
        self._apply_pending(frame, mask, now)
        before = self.sm.state
        state = self.sm.tick(now)

        if self.sm.away:
            self._was_away = True
            out = self.away.next_frame()
            if out is None:
                out = frame
            elif out.shape != frame.shape:
                out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
            self.latest = out
            return out
        if self._was_away:
            self._was_away = False
            self.away.reset_playhead()

        # Never record the away fallback while away, or it fills with you leaving.
        self.away.push(frame, now)
        if self._plate is None or self._plate.plate.shape != frame.shape:
            self._plate = BackgroundPlate(frame.shape)
        self._plate.update(frame, mask)

        if state is State.PRESENT:
            out = frame
            if before is State.RETURNING:
                self._xfade_from, self._xfade_start = self.latest, now
            if self._xfade_from is not None:
                k = (now - self._xfade_start) / XFADE_S
                if k >= 1 or self._xfade_from.shape != frame.shape:
                    self._xfade_from = None
                else:
                    out = cv2.addWeighted(self._xfade_from, 1 - k, frame, k, 0)
        elif state is State.VANISHED:
            out = composite(frame, self._plate.filled(), feather(mask))
        else:
            hide = feather(np.maximum(self._fx_mask, mask))
            base = composite(frame, self._plate.filled(), hide)
            out = self._fx.render(base, now - self.sm.started_at,
                                  reverse=state is State.RETURNING)

        self.latest = out
        return out

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ashcam-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        import pyvirtualcam

        vcam = None
        try:
            vcam = pyvirtualcam.Camera(OUT_SIZE[0], OUT_SIZE[1], 30,
                                       fmt=pyvirtualcam.PixelFormat.RGB, backend="obs")
            self.vcam_ok = True
        except Exception:
            self.error = VCAM_HELP
        misses = 0
        try:
            while self._running:
                frame = self.camera.read()
                misses = misses + 1 if frame is None else 0
                if misses > 30:
                    self.error = CAMERA_HELP
                    time.sleep(0.1)
                elif frame is not None and self.error == CAMERA_HELP:
                    self.error = None
                out = self.process(frame, time.monotonic())
                if self.sink:
                    self.sink(out)
                if vcam is not None:
                    if (out.shape[1], out.shape[0]) != OUT_SIZE:
                        out = cv2.resize(out, OUT_SIZE)
                    vcam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        finally:
            if vcam is not None:
                vcam.close()
