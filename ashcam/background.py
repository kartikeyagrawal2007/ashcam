"""Live clean plate: remembers every part of the room the person isn't covering.

Cameras keep re-adjusting exposure and white balance, so a part of the room
last seen a minute ago is often darker or bluer than the live picture. Each
block of the plate is stamped with the camera's exposure at the moment it was
written; when the plate is used, every block is scaled by "exposure now ÷
exposure then". The exposure is measured fresh each frame against a fixed
baseline, so nothing is multiplied into the plate and errors can't pile up.
"""

import cv2
import numpy as np

# Square kernel: OpenCV dilates rectangles far faster than ellipses.
_HALO = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
_GROW_HOLE = np.ones((3, 3), np.uint8)
# A pixel must look like background this many frames in a row before it is
# learnt, so a segmenter dropout or two can't burn the person into the room.
LEARN_FRAMES = 3
EXPOSURE_RANGE = (0.25, 4.0)
MIN_BLOCKS = 4              # visible room blocks needed to trust a measurement
RELIGHT_EPS = 0.01          # skip the full-size multiply below 1% difference


def _block_mean(img: np.ndarray, valid: np.ndarray):
    """Per-channel mean of `img` over `valid` pixels in 1/8-size blocks, and
    the share of each block that was valid. Never mixes in invalid pixels."""
    h, w = valid.shape
    sw, sh = max(1, w // 8), max(1, h // 8)
    v8 = valid.astype(np.uint8)
    masked = cv2.bitwise_and(img, img, mask=v8)
    cover = cv2.resize(v8 * 255, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    total = cv2.resize(masked, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    return total / np.maximum(cover, 1e-6)[..., None], cover


def _scale(img: np.ndarray, factor_small: np.ndarray) -> np.ndarray:
    """Multiply `img` by a per-block, per-channel factor, smoothly upsampled."""
    h, w = img.shape[:2]
    f = cv2.resize(factor_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.multiply(img.astype(np.float32), f).clip(0, 255).astype(np.uint8)


class BackgroundPlate:
    def __init__(self, shape: tuple[int, int, int]):
        h, w = shape[:2]
        sw, sh = max(1, w // 8), max(1, h // 8)
        self.plate = np.zeros(shape, np.uint8)
        self.known = np.zeros(shape[:2], bool)
        self._streak = np.zeros(shape[:2], np.uint8)
        self._last: np.ndarray | None = None
        # Brightness bookkeeping (per 8x8 block, per channel).
        self._base = np.zeros((sh, sw, 3), np.float32)   # block brightness at exposure 1
        self._base_ok = np.zeros((sh, sw), bool)
        self._stamp = np.ones((sh, sw, 3), np.float32)   # exposure when last written
        self.exposure = np.ones(3, np.float32)           # camera exposure now
        self._guess: np.ndarray | None = None
        self._guess_exposure = np.ones(3, np.float32)
        self._guess_for = -1  # known-pixel count the guess was made for
        self.ai = None        # BackgroundFiller: LaMa fill, computed in the background
        self.room_ref: np.ndarray | None = None   # the empty room, at exposure 1

    def _measure(self, frame: np.ndarray, visible: np.ndarray) -> np.ndarray | None:
        """Camera exposure of `frame`, per channel, against the fixed baseline.

        The median over every visible block that has a baseline: robust to a
        few changed objects and to noise, and computed fresh every time."""
        live, cover = _block_mean(frame, visible)
        ok = (cover > 0.5) & self._base_ok
        if ok.sum() < MIN_BLOCKS:
            return None
        ratios = live[ok] / np.maximum(self._base[ok], 8.0)
        return np.clip(np.median(ratios, axis=0), *EXPOSURE_RANGE).astype(np.float32)

    def update(self, frame: np.ndarray, mask: np.ndarray) -> None:
        person = cv2.dilate((mask > 0.2).astype(np.uint8), _HALO) > 0
        self._streak = np.where(person, 0, np.minimum(self._streak + 1, LEARN_FRAMES))
        self._streak = self._streak.astype(np.uint8)
        learn = self._streak >= LEARN_FRAMES
        measured = self._measure(frame, learn & self.known)
        if measured is not None:
            self.exposure = measured
        np.copyto(self.plate, frame, where=learn[..., None])
        self.known |= learn
        self._last = frame
        # Blocks just written carry today's exposure.
        _, written = _block_mean(frame, learn)
        self._stamp[written > 0.5] = self.exposure
        # Blocks seen for the first time get a baseline, in exposure-1 units.
        mean, cover = _block_mean(self.plate, self.known)
        new = (cover > 0.5) & ~self._base_ok
        if new.any():
            self._base[new] = mean[new] / self.exposure
            self._base_ok[new] = True

    def capture(self, frame: np.ndarray) -> None:
        """Learn the whole room at once (taken while nobody is in view)."""
        self.plate = frame.copy()
        self.room_ref = frame.copy()
        self.known[:] = True
        self._streak[:] = LEARN_FRAMES
        self._last = frame
        self.exposure = np.ones(3, np.float32)
        self._stamp[:] = 1.0
        self._base, cover = _block_mean(frame, self.known)
        self._base_ok = cover > 0.5
        self._guess, self._guess_for = None, -1
        if self.ai is not None:
            self.ai.latest = None   # nothing left to fill

    def request_ai_fill(self) -> None:
        """Ask the AI filler to fill the never-seen area, in the background."""
        if self.ai is None or self.known.all() or not self.known.any():
            return
        now = self.exposure.copy()
        self.ai.request(self._lit(self.plate, now / self._stamp), ~self.known,
                        tag=now)

    @staticmethod
    def _lit(img: np.ndarray, factor: np.ndarray) -> np.ndarray:
        return img if np.abs(factor - 1).max() < RELIGHT_EPS else _scale(img, factor)

    def matched(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """The remembered room, lit like `frame` (measured where the room is
        visible in it right now)."""
        person = cv2.dilate((mask > 0.2).astype(np.uint8), _HALO) > 0
        measured = self._measure(frame, self.known & ~person)
        if self.room_ref is not None:
            # A saved empty room beats anything remembered while you sat there:
            # no shadow of you in it, nothing guessed.
            return self.room_now(measured)
        return self.filled(exposure=measured)

    def room_now(self, exposure: np.ndarray | None = None) -> np.ndarray:
        """The saved empty room, lit like the camera is right now."""
        now = self.exposure if exposure is None else exposure
        if np.abs(now - 1).max() < RELIGHT_EPS:
            return self.room_ref
        return (self.room_ref * now).clip(0, 255).astype(np.uint8)

    def _inpaint_guess(self, plate_now: np.ndarray) -> np.ndarray:
        h, w = self.known.shape
        small = cv2.resize(plate_now, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        hole = cv2.resize((~self.known).astype(np.uint8), (w // 4, h // 4),
                          interpolation=cv2.INTER_NEAREST)
        # Downscaling blends never-seen (black) pixels into the hole's border;
        # grow the hole so those darkened pixels get filled, not copied inward.
        hole = cv2.dilate(hole, _GROW_HOLE)
        guess = cv2.inpaint(small, hole, 5, cv2.INPAINT_TELEA)
        return cv2.resize(guess, (w, h), interpolation=cv2.INTER_LINEAR)

    def filled(self, exposure: np.ndarray | None = None) -> np.ndarray:
        """Plate lit for `exposure` (default: the latest measured), with
        never-seen areas inpainted, so there's never a hole."""
        count = int(self.known.sum())
        if count == 0:
            # Nothing seen yet: a heavy blur of the frame beats a black screen.
            if self._last is None:
                return self.plate
            h, w = self.known.shape
            tiny = cv2.resize(self._last, (max(1, w // 32), max(1, h // 32)),
                              interpolation=cv2.INTER_AREA)
            return cv2.resize(tiny, (w, h), interpolation=cv2.INTER_LINEAR)
        now = self.exposure if exposure is None else exposure
        plate_now = self._lit(self.plate, now / self._stamp)
        if count == self.known.size:
            return plate_now
        # The AI fill, once ready: still valid as the room is learnt, since
        # the never-seen area only ever shrinks.
        ai = self.ai.latest if self.ai is not None else None
        if ai is not None and ai[0].shape == self.plate.shape:
            filled_img, made_at = ai
            g = now / made_at
            ai_now = filled_img if np.abs(g - 1).max() < RELIGHT_EPS else \
                (filled_img * g).clip(0, 255).astype(np.uint8)
            return np.where(self.known[..., None], plate_now, ai_now)
        # Inpainting is the slow part; only redo it when new pixels got seen.
        if count != self._guess_for:
            self._guess = self._inpaint_guess(plate_now)
            self._guess_exposure = now.copy()
            self._guess_for = count
        g = now / self._guess_exposure
        guess = self._guess if np.abs(g - 1).max() < RELIGHT_EPS else \
            (self._guess * g).clip(0, 255).astype(np.uint8)
        return np.where(self.known[..., None], plate_now, guess)
