"""Live clean plate: remembers every part of the room the person isn't covering."""

import cv2
import numpy as np

# Square kernel: OpenCV dilates rectangles far faster than ellipses.
_HALO = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
_GROW_HOLE = np.ones((3, 3), np.uint8)
# A pixel must look like background this many frames in a row before it is
# learnt, so a segmenter dropout or two can't burn the person into the room.
LEARN_FRAMES = 3


class BackgroundPlate:
    def __init__(self, shape: tuple[int, int, int]):
        self.plate = np.zeros(shape, np.uint8)
        self.known = np.zeros(shape[:2], bool)
        self._streak = np.zeros(shape[:2], np.uint8)
        self._last: np.ndarray | None = None
        self._guess: np.ndarray | None = None
        self._guess_for = -1  # known-pixel count the guess was made for

    def update(self, frame: np.ndarray, mask: np.ndarray) -> None:
        person = cv2.dilate((mask > 0.2).astype(np.uint8), _HALO) > 0
        self._streak = np.where(person, 0, np.minimum(self._streak + 1, LEARN_FRAMES))
        self._streak = self._streak.astype(np.uint8)
        learn = self._streak >= LEARN_FRAMES
        np.copyto(self.plate, frame, where=learn[..., None])
        self.known |= learn
        self._last = frame

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
