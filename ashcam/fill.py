"""AI fill for the part of the room AshCam has never seen.

Uses LaMa (Suvorov et al., WACV 2022; Apache-2.0), the strongest open-source
model for filling large holes: it continues walls, door frames, shelves and
furniture edges instead of smearing colours inward like classic inpainting.
It takes about 1.4 s per fill on a MacBook's CPU, far too slow for every
frame, so it runs in the background. The area behind you hardly changes while
you're gone, so one fill (refreshed now and then) is all that's needed; the
quick classic fill shows until it's ready.
"""

import os
import threading

import cv2
import numpy as np

from .segmenter import ensure_model

LAMA_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
LAMA_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models",
                          "lama_fp32.onnx")
SIZE = 512            # the model's fixed input size
CONTEXT = 1.8         # crop this much wider than the hole, so LaMa sees the room
MIN_CROP = 256
_SEAM = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


class LamaFiller:
    def __init__(self, model_path: str = LAMA_MODEL, threads: int = 4, session=None):
        self.model_path = model_path
        self.threads = threads
        self._session = session

    def _model(self):
        if self._session is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            # Leave the other cores to the live video.
            opts.intra_op_num_threads = self.threads
            self._session = ort.InferenceSession(
                ensure_model(self.model_path, LAMA_URL), opts,
                providers=["CPUExecutionProvider"])
        return self._session

    def fill(self, img_bgr: np.ndarray, hole: np.ndarray) -> np.ndarray:
        """`img_bgr` with the pixels in `hole` filled in. Nothing else changes."""
        if not hole.any():
            return img_bgr
        h, w = hole.shape
        ys, xs = np.nonzero(hole)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        side = max(ys.max() - ys.min(), xs.max() - xs.min()) * CONTEXT
        side = max(MIN_CROP, side)
        cw, ch = int(min(side, w)), int(min(side, h))
        x0 = int(np.clip(cx - cw / 2, 0, w - cw))
        y0 = int(np.clip(cy - ch / 2, 0, h - ch))
        crop = img_bgr[y0:y0 + ch, x0:x0 + cw]
        hole_crop = hole[y0:y0 + ch, x0:x0 + cw]

        rgb = cv2.cvtColor(cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        # A few pixels of slack so the seam lands outside the hole's soft edge.
        m = cv2.dilate(hole_crop.astype(np.uint8), _SEAM)
        m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        feed = {
            "image": (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None],
            "mask": (m > 0).astype(np.float32)[None, None],
        }
        out = self._model().run(None, feed)[0][0]          # 3x512x512 RGB, 0..255
        out = np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8)
        out = cv2.resize(cv2.cvtColor(out, cv2.COLOR_RGB2BGR), (cw, ch),
                         interpolation=cv2.INTER_LINEAR)

        result = img_bgr.copy()
        region = result[y0:y0 + ch, x0:x0 + cw]
        region[hole_crop] = out[hole_crop]
        return result


class BackgroundFiller:
    """Runs a filler on its own thread. `request` never blocks; the newest
    request wins. `latest` is (filled image, tag) once one finishes."""

    def __init__(self, filler: LamaFiller | None = None):
        self.filler = filler or LamaFiller()
        self.state = "idle"          # loading | idle | working | ready | error
        self.latest: tuple[np.ndarray, object] | None = None
        self._job: tuple[np.ndarray, np.ndarray, object] | None = None
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def preload(self) -> None:
        """Load (and on first run, download) the model now, off the video
        thread, so the first snap doesn't wait for it."""
        def load():
            try:
                self.filler._model()
                if self.state == "loading":
                    self.state = "idle"
            except Exception:
                self.state = "error"

        self.state = "loading"
        threading.Thread(target=load, name="ashcam-fill-load", daemon=True).start()

    def request(self, img_bgr: np.ndarray, hole: np.ndarray, tag=None) -> None:
        with self._lock:
            self._job = (img_bgr.copy(), hole.copy(), tag)
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="ashcam-fill",
                                                daemon=True)
                self._thread.start()
        self._wake.set()

    def run_pending(self) -> None:
        """Do the newest pending job now (the thread's loop body)."""
        with self._lock:
            job, self._job = self._job, None
        if job is None:
            return
        img, hole, tag = job
        self.state = "working"
        try:
            self.latest = (self.filler.fill(img, hole), tag)
            self.state = "ready"
        except Exception:
            self.state = "error"   # the quick classic fill keeps working

    def _loop(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            self.run_pending()
