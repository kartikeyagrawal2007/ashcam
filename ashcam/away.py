"""Go away mode: a recorded clip of you sitting, played forward then back."""

import threading
from collections import deque

import cv2
import numpy as np

# A finished recording shorter than this share of clip_s (camera stalled, or
# away was on) would loop a few frozen frames, so the old clip is kept.
MIN_CLIP_SHARE = 0.4


def pingpong_index(i: int, n: int) -> int:
    if n <= 1:
        return 0
    period = 2 * (n - 1)
    k = i % period
    return k if k < n else period - k


def _encode(frame: np.ndarray) -> bytes:
    # Raw 720p frames for 15 s would be ~1.2 GB, so keep them as JPEG.
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def _decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class AwayLoop:
    def __init__(self, fps: int = 30, rolling_s: float = 5, clip_s: float = 15):
        self.fps = fps
        self.clip_s = clip_s
        self._rolling: deque[bytes] = deque(maxlen=int(fps * rolling_s))
        self._clip: list[bytes] = []
        self._recording: list[bytes] | None = None
        self._rec_start = 0.0  # when capture begins (after the countdown)
        self._playhead = 0
        self._lock = threading.Lock()

    @property
    def has_clip(self) -> bool:
        return bool(self._clip)

    def start_recording(self, countdown_s: float = 3.0, now: float = 0.0) -> None:
        with self._lock:
            self._recording = []
            self._rec_start = now + countdown_s

    def cancel_recording(self) -> None:
        with self._lock:
            self._recording = None

    def status(self, now: float) -> tuple[str, float]:
        if self._recording is None:
            return "idle", 0.0
        if now < self._rec_start:
            return "countdown", self._rec_start - now
        return "recording", max(0.0, self._rec_start + self.clip_s - now)

    def push(self, frame: np.ndarray, now: float) -> None:
        data = _encode(frame)
        with self._lock:
            self._rolling.append(data)
            if self._recording is None or now < self._rec_start:
                return
            self._recording.append(data)
            full = len(self._recording) >= self.fps * self.clip_s
            if full or now >= self._rec_start + self.clip_s:
                if len(self._recording) >= MIN_CLIP_SHARE * self.fps * self.clip_s:
                    self._clip = self._recording
                    self._playhead = 0
                self._recording = None

    def reset_playhead(self) -> None:
        self._playhead = 0

    def next_frame(self) -> np.ndarray | None:
        with self._lock:
            frames = self._clip or list(self._rolling)
            if not frames:
                return None
            data = frames[pingpong_index(self._playhead, len(frames))]
            self._playhead += 1
        return _decode(data)
