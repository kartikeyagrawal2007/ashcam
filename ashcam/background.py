"""Live clean plate: remembers every part of the room the person isn't covering."""

import cv2
import numpy as np

# Square kernel: OpenCV dilates rectangles far faster than ellipses.
_HALO = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))


class BackgroundPlate:
    def __init__(self, shape: tuple[int, int, int]):
        self.plate = np.zeros(shape, np.uint8)
        self.known = np.zeros(shape[:2], bool)
        self._guess: np.ndarray | None = None
        self._guess_for = -1  # known-pixel count the guess was made for

    def update(self, frame: np.ndarray, mask: np.ndarray) -> None:
        person = cv2.dilate((mask > 0.2).astype(np.uint8), _HALO) > 0
        seen = ~person
        np.copyto(self.plate, frame, where=seen[..., None])
        self.known |= seen

    def _inpaint_guess(self) -> np.ndarray:
        h, w = self.known.shape
        small = cv2.resize(self.plate, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        hole = cv2.resize((~self.known).astype(np.uint8), (w // 4, h // 4),
                          interpolation=cv2.INTER_NEAREST)
        guess = cv2.inpaint(small, hole, 5, cv2.INPAINT_TELEA)
        return cv2.resize(guess, (w, h), interpolation=cv2.INTER_LINEAR)

    def filled(self) -> np.ndarray:
        """Plate with never-seen areas inpainted, so there's never a hole."""
        count = int(self.known.sum())
        if count == 0 or count == self.known.size:
            return self.plate
        # Inpainting is the slow part; only redo it when new pixels got seen.
        if count != self._guess_for:
            self._guess = self._inpaint_guess()
            self._guess_for = count
        return np.where(self.known[..., None], self.plate, self._guess)
