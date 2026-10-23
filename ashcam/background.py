"""Live clean plate: remembers every part of the room the person isn't covering."""

import cv2
import numpy as np

# Square kernel: OpenCV dilates rectangles far faster than ellipses.
_HALO = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
_GROW_HOLE = np.ones((3, 3), np.uint8)
# A pixel must look like background this many frames in a row before it is
# learnt, so a segmenter dropout or two can't burn the person into the room.
LEARN_FRAMES = 3
GAIN_RANGE = (0.5, 2.0)
RELIGHT_THRESHOLD = 0.02   # ignore brightness changes smaller than 2%


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


def _spread(ratio: np.ndarray, ok: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Smooth a block-level gain across the frame from the blocks in `ok`
    (normalised convolution), at full size."""
    valid = ok.astype(np.float32)
    sigma = max(1.0, ratio.shape[1] / 10)
    weight = cv2.GaussianBlur(valid, (0, 0), sigma)
    spread = cv2.GaussianBlur(ratio * valid[..., None], (0, 0), sigma)
    overall = (ratio * valid[..., None]).sum(axis=(0, 1)) / max(valid.sum(), 1e-6)
    g = np.where(weight[..., None] > 1e-3, spread / np.maximum(weight[..., None], 1e-6), overall)
    return cv2.resize(g.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _scale(img: np.ndarray, g: np.ndarray) -> np.ndarray:
    return cv2.multiply(img.astype(np.float32), g).clip(0, 255).astype(np.uint8)


class BackgroundPlate:
    def __init__(self, shape: tuple[int, int, int]):
        self.plate = np.zeros(shape, np.uint8)
        self.known = np.zeros(shape[:2], bool)
        self._streak = np.zeros(shape[:2], np.uint8)
        self._last: np.ndarray | None = None
        self._guess: np.ndarray | None = None
        self._guess_for = -1  # known-pixel count the guess was made for
        self._ref: np.ndarray | None = None        # block means at the last re-light
        self._ref_cover: np.ndarray | None = None

    def update(self, frame: np.ndarray, mask: np.ndarray) -> None:
        person = cv2.dilate((mask > 0.2).astype(np.uint8), _HALO) > 0
        self._streak = np.where(person, 0, np.minimum(self._streak + 1, LEARN_FRAMES))
        self._streak = self._streak.astype(np.uint8)
        learn = self._streak >= LEARN_FRAMES
        self._relight_hidden(frame, self.known & learn)
        np.copyto(self.plate, frame, where=learn[..., None])
        self.known |= learn
        self._last = frame
        self._adopt_new_blocks()

    def _adopt_new_blocks(self) -> None:
        """Give newly learnt parts of the room a brightness baseline."""
        if self._ref is not None and (self._ref_cover > 0.5).all():
            return
        mean, cover = _block_mean(self.plate, self.known)
        if self._ref is None:
            self._ref, self._ref_cover = mean, cover
            return
        new = (cover > 0.5) & ~(self._ref_cover > 0.5)
        self._ref[new], self._ref_cover[new] = mean[new], cover[new]

    def _relight_hidden(self, frame: np.ndarray, visible: np.ndarray) -> None:
        """Re-light what can't be seen right now (behind you, and the guess).

        Compares the live room with how it looked at the last re-light, so
        slow exposure drift adds up instead of slipping by a frame at a time.
        Only does the full-size multiply when the light has really moved.
        """
        if self._ref is None or not visible.any():
            return
        live, cover = _block_mean(frame, visible)
        both = (cover > 0.5) & (self._ref_cover > 0.5)
        if not both.any():
            return
        ratio = np.clip(live / np.maximum(self._ref, 8.0), *GAIN_RANGE)
        if np.abs(ratio[both] - 1).max() < RELIGHT_THRESHOLD:
            return
        g = _spread(ratio, both, self.known.shape)
        self.plate = _scale(self.plate, g)
        if self._guess is not None:
            self._guess = _scale(self._guess, g)
        self._ref[both] = live[both]

    def capture(self, frame: np.ndarray) -> None:
        """Learn the whole room at once (taken while nobody is in view)."""
        self.plate = frame.copy()
        self.known[:] = True
        self._ref = None
        self._guess = None
        self._guess_for = -1
        self._streak[:] = LEARN_FRAMES
        self._last = frame
        self._adopt_new_blocks()

    def matched(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """The remembered room, re-lit to match the live frame.

        Cameras keep re-adjusting exposure and white balance, so a room
        remembered a minute ago is often a little darker or bluer than the
        live picture, and the hidden area shows as a patch. Compare the two
        wherever the room is visible right now, spread that per-channel gain
        smoothly across the hidden area, and apply it.
        """
        base = self.filled()
        person = cv2.dilate((mask > 0.2).astype(np.uint8), _HALO) > 0
        valid = self.known & ~person
        live, cover = _block_mean(frame, valid)
        old, _ = _block_mean(base, valid)
        ok = cover > 0.5
        if not ok.any():
            return base
        ratio = np.clip(live / np.maximum(old, 8.0), *GAIN_RANGE)
        if np.abs(ratio[ok] - 1).max() < RELIGHT_THRESHOLD:
            return base
        return _scale(base, _spread(ratio, ok, self.known.shape))

    def _inpaint_guess(self) -> np.ndarray:
        h, w = self.known.shape
        small = cv2.resize(self.plate, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        hole = cv2.resize((~self.known).astype(np.uint8), (w // 4, h // 4),
                          interpolation=cv2.INTER_NEAREST)
        # Downscaling blends never-seen (black) pixels into the hole's border;
        # grow the hole so those darkened pixels get filled, not copied inward.
        hole = cv2.dilate(hole, _GROW_HOLE)
        guess = cv2.inpaint(small, hole, 5, cv2.INPAINT_TELEA)
        return cv2.resize(guess, (w, h), interpolation=cv2.INTER_LINEAR)

    def filled(self) -> np.ndarray:
        """Plate with never-seen areas inpainted, so there's never a hole."""
        count = int(self.known.sum())
        if count == self.known.size:
            return self.plate
        if count == 0:
            # Nothing seen yet: a heavy blur of the frame beats a black screen.
            if self._last is None:
                return self.plate
            h, w = self.known.shape
            tiny = cv2.resize(self._last, (max(1, w // 32), max(1, h // 32)),
                              interpolation=cv2.INTER_AREA)
            return cv2.resize(tiny, (w, h), interpolation=cv2.INTER_LINEAR)
        # Inpainting is the slow part; only redo it when new pixels got seen.
        if count != self._guess_for:
            self._guess = self._inpaint_guess()
            self._guess_for = count
        return np.where(self.known[..., None], self.plate, self._guess)
