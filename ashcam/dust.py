"""Endgame-style disintegration: the person crumbles into drifting ash.

Every particle's position is a closed-form function of time, so the effect
can be played backwards (to reassemble) by rendering at `duration - t`.
"""

import cv2
import numpy as np

BLOCK = 3                 # one particle per 3x3 block of the person
MAX_PARTICLES = 12000
ASH_BGR = np.array([70, 85, 100], np.float32)   # brownish grey
EMBER_BGR = np.array([0, 25, 55], np.float32)   # hot edge just before release
EMBER_S = 0.05
WIND = np.array([-90.0, -50.0], np.float32)     # px/s^2 at 720p: blow back and up,
                                                # away from the side still crumbling


class DustEffect:
    def __init__(self, snapshot: np.ndarray, mask: np.ndarray, sweep_s: float = 1.5,
                 life_s: float = 1.0, seed: int = 0):
        self.sweep_s = sweep_s
        self.life_s = life_s
        self.duration = sweep_s + life_s
        self._snapshot = snapshot
        h, w = mask.shape
        rng = np.random.default_rng(seed)
        self._body = mask > 0.5
        self._scale = h / 720.0

        # Release time per pixel: a ragged front sweeping left to right.
        ys, xs = np.nonzero(self._body)
        x0, x1 = (xs.min(), xs.max()) if len(xs) else (0, 1)
        span = max(1, x1 - x0)
        y = np.arange(h, dtype=np.float32)
        p = rng.uniform(0, 2 * np.pi, 3)
        ragged = 0.5 * np.sin(y / 23 + p[0]) + 0.3 * np.sin(y / 11 + p[1]) + 0.2 * np.sin(y / 5.3 + p[2])
        # Blotchy noise so the body breaks off in uneven flakes, not a clean line.
        blotch = rng.uniform(0, 1, (h // BLOCK + 1, w // BLOCK + 1)).astype(np.float32)
        blotch = cv2.GaussianBlur(blotch, (0, 0), 6)
        blotch = (blotch - blotch.min()) / max(1e-6, float(np.ptp(blotch)))
        jitter = 0.25 * blotch + rng.uniform(0, 0.04, blotch.shape).astype(np.float32)
        jit_px = np.repeat(np.repeat(jitter, BLOCK, 0), BLOCK, 1)[:h, :w]
        xn = (np.arange(w, dtype=np.float32) - x0) / span
        self._release = sweep_s * np.clip(xn[None, :] + 0.08 * ragged[:, None] + jit_px, 0, 1)

        # Particles from blocks that are mostly person.
        hb, wb = h // BLOCK, w // BLOCK
        crop = (slice(0, hb * BLOCK), slice(0, wb * BLOCK))
        cover = mask[crop].reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3))
        by, bx = np.nonzero(cover > 0.5)
        if len(by) > MAX_PARTICLES:
            keep = rng.choice(len(by), MAX_PARTICLES, replace=False)
            by, bx = by[keep], bx[keep]
        colours = snapshot[crop].reshape(hb, BLOCK, wb, BLOCK, 3).mean(axis=(1, 3))
        cy, cx = by * BLOCK + BLOCK // 2, bx * BLOCK + BLOCK // 2
        n = len(by)
        self.particle_count = n
        self._origin = np.stack([cx, cy], 1).astype(np.float32)
        self._colour = colours[by, bx].astype(np.float32)
        self._r = self._release[cy, cx]
        self._vel = np.stack([rng.uniform(-200, -60, n), rng.uniform(-100, -10, n)], 1)
        self._vel = self._vel.astype(np.float32)
        self._phase = rng.uniform(0, 2 * np.pi, (n, 2)).astype(np.float32)

    def render(self, base: np.ndarray, t: float, reverse: bool = False) -> np.ndarray:
        if reverse:
            t = self.duration - t
        if t >= self.duration or not self.particle_count and not self._body.any():
            return base.copy()
        h, w = base.shape[:2]
        out = base.astype(np.float32)

        # The part of the body that hasn't crumbled yet, with a glowing edge.
        intact = self._body & (self._release >= t)
        out[intact] = self._snapshot[intact]
        ember = intact & (self._release - t < EMBER_S)
        out[ember] = np.minimum(255, out[ember] + EMBER_BGR)

        # Released particles drift, wobble, turn to ash and fade.
        age = t - self._r
        live = (age >= 0) & (age < self.life_s)
        if live.any():
            a = age[live][:, None]
            wobble = 8 * np.sin(3 * a + self._phase[live])
            pos = self._origin[live] + self._scale * (
                self._vel[live] * a + 0.5 * WIND * a * a + wobble)
            fade = (1 - a / self.life_s)[:, 0]
            ashen = np.minimum(1.0, a * 1.5 / self.life_s)
            col = self._colour[live] * (1 - ashen) + ASH_BGR * ashen
            acc = np.zeros((h, w, 3), np.float32)
            alpha = np.zeros((h, w), np.float32)
            px = np.round(pos).astype(np.int32)
            # Fresh specks are 2x2, older ones shrink to a single pixel.
            big = fade > 0.5
            for dx, dy, sel in ((0, 0, slice(None)), (1, 0, big), (0, 1, big), (1, 1, big)):
                x = px[sel, 0] + dx
                y = px[sel, 1] + dy
                ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                f = fade[sel][ok]
                np.add.at(alpha, (y[ok], x[ok]), f)
                np.add.at(acc, (y[ok], x[ok]), col[sel][ok] * f[:, None])
            hit = alpha > 0
            a_px = np.minimum(1.0, alpha[hit])[:, None]
            out[hit] = out[hit] * (1 - a_px) + (acc[hit] / alpha[hit][:, None]) * a_px

        return np.clip(out, 0, 255).astype(np.uint8)
