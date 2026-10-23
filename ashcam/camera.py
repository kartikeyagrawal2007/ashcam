"""Real webcam (or a video file, for testing) as 1280x720 BGR frames."""

import time

import cv2
import numpy as np

SIZE = (1280, 720)


class Camera:
    def __init__(self, source: int | str = 0, width: int = SIZE[0], height: int = SIZE[1],
                 fps: int = 30):
        self.size = (width, height)
        self.fps = fps
        self.is_file = isinstance(source, str)
        if self.is_file:
            self._cap = cv2.VideoCapture(source)
        else:
            self._cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._next = time.monotonic()

    @property
    def ok(self) -> bool:
        return self._cap.isOpened()

    def read(self) -> np.ndarray | None:
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
