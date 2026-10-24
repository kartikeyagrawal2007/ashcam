"""Person mask per frame, using MediaPipe's selfie segmenter."""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# The landscape model is trained on 16:9 frames like a webcam's; the square
# one squashes them and gets sloppier edges (arms, hair, shoulders).
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite")
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models",
                             "selfie_segmenter_landscape.tflite")


def ensure_model(path: str = DEFAULT_MODEL, url: str = MODEL_URL) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Download beside it and move into place, so a cut-off download
        # never leaves a broken model that fails every later start.
        part = path + ".part"
        try:
            urllib.request.urlretrieve(url, part)
            os.replace(part, path)
        finally:
            if os.path.exists(part):
                os.remove(part)
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
        # No smoothing over time: it made the outline trail behind moving hands.
        # The engine steadies the hidden area by combining recent frames instead.
        return cv2.GaussianBlur(m, (7, 7), 0)

    def close(self) -> None:
        self._seg.close()
