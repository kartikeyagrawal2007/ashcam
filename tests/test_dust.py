import numpy as np

from ashcam.dust import DustEffect

S = 64


def scene():
    snapshot = np.zeros((S, S, 3), np.uint8)
    mask = np.zeros((S, S), np.float32)
    snapshot[16:48, 16:48] = 255
    mask[16:48, 16:48] = 1.0
    return snapshot, mask, np.zeros((S, S, 3), np.uint8)


def test_start_is_person():
    snap, mask, base = scene()
    out = DustEffect(snap, mask).render(base, 0.0)
    assert out[32, 32].min() >= 250


def test_end_is_base():
    snap, mask, base = scene()
    fx = DustEffect(snap, mask)
    assert np.array_equal(fx.render(base, fx.duration + 0.01), base)


def test_reverse_is_mirror():
    snap, mask, base = scene()
    fx = DustEffect(snap, mask)
    for t in (0.3, 1.2, 2.0):
        assert np.array_equal(fx.render(base, t, reverse=True),
                              fx.render(base, fx.duration - t))


def test_left_goes_first():
    snap, mask, base = scene()
    out = DustEffect(snap, mask).render(base, 0.6)
    body = out[16:48]
    left_gone = (body[:, 16:26].max(axis=2) < 50).sum()
    right_gone = (body[:, 38:48].max(axis=2) < 50).sum()
    assert left_gone > right_gone


def test_has_particles():
    snap, mask, _ = scene()
    fx = DustEffect(snap, mask)
    assert 0 < fx.particle_count <= 12000


def test_empty_mask_is_noop():
    snap, _, base = scene()
    fx = DustEffect(snap, np.zeros((S, S), np.float32))
    assert np.array_equal(fx.render(base, 1.0), base)
