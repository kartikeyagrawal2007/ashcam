"""Person mask per frame, using MediaPipe's selfie segmenter."""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_segmenter/float16/latest/selfie_segmenter.tflite")
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models",
                             "selfie_segmenter.tflite")


def ensure_model(path: str = DEFAULT_MODEL) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class Segmenter:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=ensure_model(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
        )
        self._seg = vision.ImageSegmenter.create_from_options(options)
        self._ts = 0
        self._prev: np.ndarray | None = None

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._ts += 33  # VIDEO mode needs strictly increasing timestamps
        result = self._seg.segment_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._ts)
        m = result.confidence_masks[-1].numpy_view().astype(np.float32)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != frame_bgr.shape[:2]:
            m = cv2.resize(m, (frame_bgr.shape[1], frame_bgr.shape[0]))
        m = cv2.GaussianBlur(m, (7, 7), 0)
        # Smooth over time so the edge doesn't flicker.
        if self._prev is not None and self._prev.shape == m.shape:
            m = 0.6 * m + 0.4 * self._prev
        self._prev = m
        return m

    def close(self) -> None:
        self._seg.close()
