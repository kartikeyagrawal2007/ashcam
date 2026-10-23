"""Finger-snap detector: a short, sharp burst of high-frequency sound."""

import threading
from typing import Callable

import numpy as np

BAND = (1500.0, 6000.0)   # where a finger snap's energy sits
DECAY_RATIO = 0.3         # burst must fall below this share of its peak...
DECAY_BLOCKS = 2          # ...for this many blocks in a row
MAX_LOUD_BLOCKS = 4       # longer than this is a sound, not a snap
REFRACTORY_S = 0.5
DOUBLE_GAP = (0.12, 0.6)
MIN_PEAK = 0.035          # band-passed peak a snap must reach at sensitivity 0.5


class SnapDetector:
    def __init__(self, on_snap: Callable[[], None], sensitivity: float = 0.5,
                 double: bool = False, samplerate: int = 48000, block: int = 512):
        self.on_snap = on_snap
        self.sensitivity = sensitivity
        self.double = double
        self.samplerate = samplerate
        self.block = block
        self.level = 0.0
        # 50% overlapped Hann windows so a snap on a block edge isn't lost.
        self._prev = np.zeros(block, np.float32)
        self._window = np.hanning(2 * block).astype(np.float32)
        freqs = np.fft.rfftfreq(2 * block, 1 / samplerate)
        self._band = (freqs >= BAND[0]) & (freqs <= BAND[1])
        self._floor: float | None = None
        self._peak = 0.0          # >0 while a candidate burst is being tracked
        self._loud = 0
        self._quiet = 0
        self._last_fire = -1e9
        self._first_snap: float | None = None
        self._stream = None
        self._lock = threading.Lock()

    def _band_peak(self, block: np.ndarray) -> float:
        """Peak of the band-passed signal over the middle of a 2-block window.

        A peak, not an average: a snap lasts ~5 ms and averaging over the
        window would water it down. The window middles tile the timeline,
        so every sample is looked at exactly once.
        """
        frame = np.concatenate([self._prev, block]) * self._window
        self._prev = block
        spec = np.fft.rfft(frame)
        spec[~self._band] = 0
        n = self.block
        mid = slice(n // 2, n + n // 2)
        filtered = np.fft.irfft(spec, 2 * n)[mid] / self._window[mid]
        return float(np.abs(filtered).max())

    def _score(self, hf: float) -> float:
        # ratio 31.6x the room's normal peak -> 1.0
        ratio = hf / (self._floor + 1e-7)
        return float(min(1.0, max(0.0, np.log10(max(ratio, 1.0)) / 1.5)))

    def _update_floor(self, hf: float) -> None:
        # Drops fast, rises slowly: tracks the room, ignores short bursts.
        alpha = 0.1 if hf < self._floor else 0.02
        self._floor += alpha * (hf - self._floor)

    def process(self, block: np.ndarray, now: float) -> bool:
        with self._lock:
            hf = self._band_peak(np.asarray(block, np.float32))
            if self._floor is None:
                self._floor = hf
                return False
            score = self._score(hf)
            self.level = score
            threshold = 0.8 - 0.6 * self.sensitivity
            # Loud enough in absolute terms too: in a silent room a faint
            # keyboard click has a big ratio but is nowhere near a snap.
            min_peak = MIN_PEAK * (1.5 - self.sensitivity)
            fired = False

            if self._peak == 0.0:
                if score > threshold and hf >= min_peak:
                    self._peak, self._loud, self._quiet = hf, 1, 0
            elif hf > self._peak:
                self._peak, self._loud, self._quiet = hf, self._loud + 1, 0
            elif hf < DECAY_RATIO * self._peak:
                self._quiet += 1
                if self._quiet >= DECAY_BLOCKS:
                    self._peak = 0.0
                    fired = self._on_transient(now)
            else:
                self._loud += 1
                self._quiet = 0

            if self._peak and self._loud > MAX_LOUD_BLOCKS:
                self._peak = 0.0  # sustained sound, not a snap

            self._update_floor(hf)

        if fired:
            self.on_snap()
        return fired

    def _on_transient(self, now: float) -> bool:
        if now - self._last_fire < REFRACTORY_S:
            return False
        if self.double:
            first = self._first_snap
            if first is not None and DOUBLE_GAP[0] <= now - first <= DOUBLE_GAP[1]:
                self._first_snap = None
            else:
                self._first_snap = now
                return False
        self._last_fire = now
        return True

    def start(self) -> None:
        import sounddevice as sd
        import time

        def callback(indata, frames, t, status):
            self.process(indata[:, 0].copy(), time.monotonic())

        self._stream = sd.InputStream(samplerate=self.samplerate, blocksize=self.block,
                                      channels=1, dtype="float32", callback=callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
