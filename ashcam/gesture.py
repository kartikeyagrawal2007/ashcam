"""Hand-tracked snaps: the camera sees the snap, the mic confirms the click.

A finger snap seen on camera is the thumb and middle fingertip pressed
together, then flying apart in a fraction of a second. Waving, typing and
talking never look like that, and music or claps never look like a hand,
so needing both kills almost every false trigger.
"""

import os
import threading
import time
from typing import Callable

import cv2
import numpy as np

from .segmenter import ensure_model

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
HAND_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models",
                          "hand_landmarker.task")
THUMB_TIP, MIDDLE_TIP, WRIST, MIDDLE_KNUCKLE = 4, 12, 0, 9


class HandSnap:
    """Spots pinch-then-fast-release in a stream of hand landmarks (pixels)."""

    def __init__(self, pinch_ratio: float = 0.35, open_ratio: float = 0.6,
                 max_release_s: float = 0.25, max_hold_s: float = 2.0):
        self.pinch_ratio = pinch_ratio
        self.open_ratio = open_ratio
        self.max_release_s = max_release_s
        self.max_hold_s = max_hold_s
        self._pinch_start: float | None = None
        self._last_pinch = 0.0

    def _reset(self) -> None:
        self._pinch_start = None

    def process(self, landmarks: np.ndarray | None, now: float) -> bool:
        if landmarks is None:
            self._reset()
            return False
        size = np.linalg.norm(landmarks[WRIST] - landmarks[MIDDLE_KNUCKLE])
        if size < 1e-6:
            return False
        gap = np.linalg.norm(landmarks[THUMB_TIP] - landmarks[MIDDLE_TIP]) / size
        if gap < self.pinch_ratio:
            if self._pinch_start is None:
                self._pinch_start = now
            self._last_pinch = now
            return False
        if self._pinch_start is None:
            return False
        if gap > self.open_ratio:
            snapped = (now - self._last_pinch <= self.max_release_s
                       and self._last_pinch - self._pinch_start <= self.max_hold_s)
            self._reset()
            return snapped
        if now - self._last_pinch > self.max_release_s:
            self._reset()  # opening slowly: a relaxing hand, not a snap
        return False


class SnapFusion:
    """Fires on_snap when the camera and the mic agree on a snap.

    With the mic muted (by AshCam while you're gone) the hand alone counts;
    without hand tracking the mic alone counts, as before.
    """

    def __init__(self, on_snap: Callable[[], None], window_s: float = 0.5,
                 refractory_s: float = 0.8):
        self.on_snap = on_snap
        self.window_s = window_s
        self.refractory_s = refractory_s
        self.hands_ok = True
        self.mic_live = True
        self._seen: float | None = None
        self._heard: float | None = None
        self._last_fire = -1e9
        self._lock = threading.Lock()

    def visual(self, now: float) -> None:
        self._event("seen", now)

    def audio(self, now: float) -> None:
        self._event("heard", now)

    def _event(self, kind: str, now: float) -> None:
        with self._lock:
            if now - self._last_fire < self.refractory_s:
                return
            setattr(self, "_" + kind, now)
            if not self.hands_ok:
                fire = self._heard is not None
            elif not self.mic_live:
                fire = self._seen is not None
            else:
                fire = (self._seen is not None and self._heard is not None
                        and abs(self._seen - self._heard) <= self.window_s)
            if fire:
                self._seen = self._heard = None
                self._last_fire = now
        if fire:
            self.on_snap()


class HandTracker:
    """MediaPipe hand landmarks for one hand, in pixels of the given frame."""

    def __init__(self, model_path: str = HAND_MODEL):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model(model_path, HAND_MODEL_URL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._det = vision.HandLandmarker.create_from_options(options)
        self._ts = 0

    def landmarks(self, frame_bgr: np.ndarray, now: float) -> np.ndarray | None:
        ts = max(self._ts + 1, int(now * 1000))
        self._ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._det.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)
        if not result.hand_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return np.array([(p.x * w, p.y * h) for p in result.hand_landmarks[0]], np.float32)

    def close(self) -> None:
        self._det.close()


class GestureWorker:
    """Runs hand tracking on its own thread, on the newest frame only, so it
    never slows the video. Reports snaps it sees to the fusion."""

    WIDTH = 640  # tracking resolution; plenty for a hand at desk distance

    def __init__(self, tracker, fusion: SnapFusion, snap: HandSnap | None = None):
        self.tracker = tracker
        self.fusion = fusion
        self.snap = snap or HandSnap()
        self.hand_seen = False
        self._frame: tuple[np.ndarray, float] | None = None
        self._ready = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

    def submit(self, frame: np.ndarray, now: float) -> None:
        self._frame = (frame, now)
        self._ready.set()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ashcam-hands", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._ready.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def step(self) -> None:
        """Process the newest submitted frame (the thread's loop body)."""
        item, self._frame = self._frame, None
        if item is None:
            return
        frame, now = item
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.WIDTH, int(h * self.WIDTH / w)),
                           interpolation=cv2.INTER_AREA)
        lm = self.tracker.landmarks(small, now)
        self.hand_seen = lm is not None
        if self.snap.process(lm, now):
            self.fusion.visual(now)

    def _loop(self) -> None:
        while self._running:
            self._ready.wait(timeout=0.5)
            self._ready.clear()
            try:
                self.step()
            except Exception:
                time.sleep(0.1)  # a bad frame must not end hand tracking
