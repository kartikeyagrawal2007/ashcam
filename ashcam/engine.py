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
             "General → Login Items & Extensions → Camera Extensions. AshCam keeps "
             "retrying, so there's no need to restart it.")
CAMERA_HELP = ("Can't read the camera. Allow your terminal app in System Settings → "
               "Privacy & Security → Camera. AshCam picks it up automatically.")


def open_obs_vcam():
    import pyvirtualcam

    return pyvirtualcam.Camera(OUT_SIZE[0], OUT_SIZE[1], 30,
                               fmt=pyvirtualcam.PixelFormat.RGB, backend="obs")


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
    VCAM_RETRY_S = 3.0

    def __init__(self, camera, segmenter, away: AwayLoop, sm: StateMachine,
                 sink: Callable[[np.ndarray], None] | None = None,
                 vcam_factory: Callable[[], object] = open_obs_vcam):
        self.camera = camera
        self.segmenter = segmenter
        self.away = away
        self.sm = sm
        self.sink = sink
        self.vcam_factory = vcam_factory
        self.snap = None          # SnapDetector, set by main for the level meter
        self.latest: np.ndarray | None = None
        self.vcam_ok = False
        # Off: the call gets the true image (text behind you reads correctly).
        self.flip_output = False
        self._composed: np.ndarray | None = None   # last output before any flip
        # Kept apart so one recovering can't hide another that hasn't.
        self.loop_error: str | None = None
        self.camera_error: str | None = None
        self.vcam_error: str | None = None
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

    @property
    def error(self) -> str | None:
        mic = getattr(self.snap, "mic_error", None) if self.snap else None
        return self.loop_error or self.camera_error or self.vcam_error or mic

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
                if self.sm.away:
                    # Recording while away would capture nothing (or the empty chair).
                    self.away.cancel_recording()
            elif self.sm.away:
                continue  # vanish is off while away; coming back must show you
            elif self.sm.toggle_vanish(now):
                # Vanishing crumbles the person as they are now; returning
                # reassembles the person who is sitting there now.
                self._fx = DustEffect(frame, mask, assemble_s=self.sm.return_s,
                                      seed=int(now * 1000) % 2**32)
                self._fx_mask = mask

    def process(self, frame: np.ndarray | None, now: float) -> np.ndarray:
        if frame is None:
            if self.latest is None:
                self.latest = np.zeros((OUT_SIZE[1], OUT_SIZE[0], 3), np.uint8)
            return self.latest
        out = self._compose(frame, now)
        self._composed = out
        # Flip only on the way out, so the room memory, effects and away clip
        # all stay in the camera's own orientation.
        self.latest = cv2.flip(out, 1) if self.flip_output else out
        return self.latest

    def _compose(self, frame: np.ndarray, now: float) -> np.ndarray:

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
                self._xfade_from, self._xfade_start = self._composed, now
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
            t = now - self.sm.started_at
            if state is State.RETURNING:
                out = self._fx.render_assemble(base, t)
            else:
                out = self._fx.render(base, t)

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
        import traceback

        vcam = None
        next_try = 0.0
        misses = 0
        try:
            while self._running:
                now = time.monotonic()
                if vcam is None and now >= next_try:
                    try:
                        vcam = self.vcam_factory()
                        self.vcam_ok, self.vcam_error = True, None
                    except Exception:
                        self.vcam_ok, self.vcam_error = False, VCAM_HELP
                        next_try = now + self.VCAM_RETRY_S
                try:
                    frame = self.camera.read()
                    misses = misses + 1 if frame is None else 0
                    if misses > 30:
                        self.camera_error = CAMERA_HELP
                        time.sleep(0.1)
                    elif frame is not None:
                        self.camera_error = None
                    out = self.process(frame, time.monotonic())
                    if self.sink:
                        self.sink(out)
                    if vcam is not None:
                        if (out.shape[1], out.shape[0]) != OUT_SIZE:
                            out = cv2.resize(out, OUT_SIZE)
                        vcam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                    self.loop_error = None
                except Exception as e:
                    # Keep the camera alive: one bad frame must not freeze the call.
                    traceback.print_exc()
                    self.loop_error = f"AshCam hit an error and is retrying: {e}"
                    time.sleep(0.05)
        finally:
            if vcam is not None:
                vcam.close()
