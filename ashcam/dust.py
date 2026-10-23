"""Endgame-style disintegration: the person crumbles into drifting ash.

`render` crumbles the person away. `render_assemble` brings them back: a
cloud of tiny ash specks rushes in and snaps into place to rebuild the body.
Particle positions are closed-form functions of time.
"""

import cv2
import numpy as np

BLOCK = 3                 # one particle per 3x3 block of the person
MAX_PARTICLES = 12000
ASH_BGR = np.array([70, 85, 100], np.float32)   # brownish grey
EMBER_BGR = np.array([0, 25, 55], np.float32)   # hot edge just before release
EMBER_S = 0.05
GLOW_BGR = np.array([150, 205, 245], np.float32)  # incoming specks: bright ember
WIND = np.array([-90.0, -50.0], np.float32)     # px/s^2 at 720p: blow back and up,
                                                # away from the side still crumbling


class DustEffect:
    def __init__(self, snapshot: np.ndarray, mask: np.ndarray, sweep_s: float = 1.5,
                 life_s: float = 1.0, assemble_s: float = 1.2, seed: int = 0):
        self.sweep_s = sweep_s
        self.assemble_s = assemble_s
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

        # Reassembly: each block lands at its own time, in blotchy patches,
        # between 30% and 95% of the way through.
        grain = rng.uniform(0, 1, blotch.shape).astype(np.float32)
        land = 0.3 + 0.4 * blotch + 0.25 * grain
        land_px = np.repeat(np.repeat(land, BLOCK, 0), BLOCK, 1)[:h, :w]
        self._arrive_px = assemble_s * land_px
        self._arrive = self._arrive_px[cy, cx]
        # Specks start in a cloud mostly to the left (where the ash blew),
        # and fly in fast, easing out as they lock into place.
        span_px = float(max(1, x1 - x0, (ys.max() - ys.min()) if len(ys) else 1))
        angle = np.pi + rng.uniform(-1.8, 1.8, n)
        dist = (0.3 + rng.uniform(0, 0.6, n)) * span_px + 30 * self._scale
        self._start = self._origin + np.stack([np.cos(angle), np.sin(angle)], 1) * dist[:, None]
        self._start = self._start.astype(np.float32)
        # Every speck is in the air from the first frame; later landers fly longer.
        self._flight = np.maximum(self._arrive, 1e-3)
        self._swirl = rng.uniform(-1, 1, n).astype(np.float32) * 30 * self._scale

    def render_assemble(self, base: np.ndarray, t: float) -> np.ndarray:
        """Specks rush in from the ash cloud and lock together into the person."""
        if not self._body.any():
            return base.copy()
        out = base.astype(np.float32)
        if t >= self.assemble_s:
            out[self._body] = self._snapshot[self._body]
            return out.astype(np.uint8)
        h, w = base.shape[:2]

        # Pieces of the body that have landed, with a brief glow as they lock in.
        landed = self._body & (self._arrive_px <= t)
        out[landed] = self._snapshot[landed]
        spark = landed & (t - self._arrive_px < EMBER_S)
        out[spark] = np.minimum(255, out[spark] + EMBER_BGR)

        # Specks in flight: solid from the start, no fade-in.
        u = (t - (self._arrive - self._flight)) / np.maximum(self._flight, 1e-6)
        live = (u >= 0) & (u < 1)
        if live.any():
            uu = u[live][:, None]
            ease = 1 - (1 - uu) ** 3
            start, origin = self._start[live], self._origin[live]
            d = origin - start
            perp = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
            curve = perp * (self._swirl[live][:, None] * np.sin(np.pi * uu))
            pos = start + d * ease + curve
            cool = uu ** 2  # glowing ember cools into the person's real colour
            col = GLOW_BGR * (1 - cool) + self._colour[live] * cool
            self._splat(out, pos, col, np.ones(len(pos), np.float32),
                        big=np.ones(len(pos), bool))
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _splat(out: np.ndarray, pos: np.ndarray, col: np.ndarray, alpha: np.ndarray,
               big: np.ndarray) -> None:
        """Draw specks into `out` (float, in place): 1 px, or 2x2 where `big`."""
        h, w = out.shape[:2]
        px = np.round(pos).astype(np.int32)
        xs, ys, fs, cs = [], [], [], []
        for dx, dy, sel in ((0, 0, slice(None)), (1, 0, big), (0, 1, big), (1, 1, big)):
            xs.append(px[sel, 0] + dx)
            ys.append(px[sel, 1] + dy)
            fs.append(alpha[sel])
            cs.append(col[sel])
        x, y = np.concatenate(xs), np.concatenate(ys)
        f, c = np.concatenate(fs), np.concatenate(cs)
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        idx, f, c = y[ok] * w + x[ok], f[ok], c[ok]
        if not len(idx):
            return
        # Splat by flat pixel index; bincount is much faster than add.at.
        hit, inv = np.unique(idx, return_inverse=True)
        acc_a = np.bincount(inv, f)
        acc = np.stack([np.bincount(inv, c[:, k] * f) for k in range(3)], 1)
        flat = out.reshape(-1, 3)
        a_px = np.minimum(1.0, acc_a)[:, None]
        flat[hit] = flat[hit] * (1 - a_px) + (acc / acc_a[:, None]) * a_px

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
            # Fresh specks are 2x2, older ones shrink to a single pixel.
            self._splat(out, pos, col, fade, big=fade > 0.5)

        return np.clip(out, 0, 255).astype(np.uint8)
