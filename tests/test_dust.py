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


def test_assemble_starts_with_no_body():
    snap, mask, base = scene()
    out = DustEffect(snap, mask).render_assemble(base, 0.0)
    # Ash may already be flying through, but none of the person is formed yet.
    formed = (out[16:48, 16:48].min(axis=2) > 200).mean()
    assert formed < 0.05


def test_assemble_ends_as_person():
    snap, mask, base = scene()
    fx = DustEffect(snap, mask)
    out = fx.render_assemble(base, fx.assemble_s)
    assert np.array_equal(out[mask > 0.5], snap[mask > 0.5])
    assert np.array_equal(out[mask == 0], base[mask == 0])


def test_assemble_particles_are_solid_in_flight():
    # Particles fly in at full strength; nothing fades in.
    snap, mask, base = scene()
    fx = DustEffect(snap, mask)
    out = fx.render_assemble(base, fx.assemble_s * 0.4)
    outside = out[mask == 0]
    lit = outside[outside.max(axis=1) > 0]
    assert len(lit) > 20
    assert lit.max(axis=1).mean() >= 60


def test_assemble_body_fills_in_over_time():
    snap, mask, base = scene()
    fx = DustEffect(snap, mask)
    filled = [(fx.render_assemble(base, t)[16:48, 16:48].max(axis=2) > 200).sum()
              for t in np.linspace(0, fx.assemble_s, 6)]
    assert filled == sorted(filled) and filled[0] < filled[-1]


def test_assemble_is_quick():
    snap, mask, _ = scene()
    assert DustEffect(snap, mask).assemble_s <= 1.5
