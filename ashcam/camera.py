"""Real webcam (or a video file, for testing) as 1280x720 BGR frames."""

import time

import cv2
import numpy as np

SIZE = (1280, 720)


def _open_avfoundation(source: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


class Camera:
    def __init__(self, source: int | str = 0, width: int = SIZE[0], height: int = SIZE[1],
                 fps: int = 30, opener=None, retry_s: float = 2.0):
        self.size = (width, height)
        self.fps = fps
        self.source = source
        self.is_file = isinstance(source, str)
        self.retry_s = retry_s
        if opener is None:
            opener = (cv2.VideoCapture if self.is_file
                      else lambda src: _open_avfoundation(src, width, height, fps))
        self._opener = opener
        self._cap = opener(source)
        self._next = time.monotonic()
        self._next_try = time.monotonic() + retry_s

    @property
    def ok(self) -> bool:
        return self._cap.isOpened()

    def _reopen_if_blocked(self) -> bool:
        """Until macOS camera access is allowed, try again every retry_s seconds."""
        if self.is_file or self._cap.isOpened():
            return True
        now = time.monotonic()
        if now < self._next_try:
            return False
        self._next_try = now + self.retry_s
        self._cap.release()
        self._cap = self._opener(self.source)
        return self._cap.isOpened()

    def read(self) -> np.ndarray | None:
        if not self._reopen_if_blocked():
            return None
        if self.is_file:
            # Pace files to real time and loop them.
            time.sleep(max(0.0, self._next - time.monotonic()))
            self._next = max(self._next + 1 / self.fps, time.monotonic())
        ok, frame = self._cap.read()
        if not ok and self.is_file:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return frame

    def close(self) -> None:
        self._cap.release()
