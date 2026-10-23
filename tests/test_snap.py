import numpy as np

from ashcam.snap import SnapDetector

SR = 48000
BLOCK = 512
rng = np.random.default_rng(1)


def quiet(seconds, level=0.005):
    return rng.normal(0, level, int(SR * seconds)).astype(np.float32)


def band_noise(n, lo, hi):
    spec = np.fft.rfft(rng.normal(0, 1, n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return x / np.abs(x).max()


def snap_burst():
    n = int(SR * 0.015)
    t = np.arange(n) / SR
    return (0.5 * band_noise(n, 2000, 4000) * np.exp(-t / 0.004)).astype(np.float32)


def thump():
    n = int(SR * 0.05)
    t = np.arange(n) / SR
    env = np.minimum(1, t / 0.005) * np.exp(-t / 0.02)
    return (0.6 * np.sin(2 * np.pi * 100 * t) * env).astype(np.float32)


def with_events(seconds, events):
    """Quiet track with (time, burst) pairs mixed in."""
    x = quiet(seconds)
    for at, burst in events:
        i = int(at * SR)
        x[i : i + len(burst)] += burst
    return x


def feed(det, signal):
    fires = []
    for k in range(0, len(signal) - BLOCK + 1, BLOCK):
        now = k / SR
        if det.process(signal[k : k + BLOCK], now):
            fires.append(now)
    return fires


def detector(**kw):
    return SnapDetector(on_snap=lambda: None, samplerate=SR, block=BLOCK, **kw)


def test_snap_detected():
    calls = []
    det = SnapDetector(on_snap=lambda: calls.append(1), samplerate=SR, block=BLOCK)
    fires = feed(det, with_events(2.0, [(1.0, snap_burst())]))
    assert len(fires) == 1
    assert 1.0 <= fires[0] <= 1.1
    assert calls == [1]


def test_sustained_noise_no_snap():
    x = np.concatenate([quiet(1.0), rng.normal(0, 0.3, SR * 3).astype(np.float32)])
    fires = feed(detector(), x)
    assert [t for t in fires if t > 1.2] == []


def test_low_thump_no_snap():
    assert feed(detector(), with_events(2.0, [(1.0, thump())])) == []


def test_double_mode():
    assert feed(detector(double=True), with_events(2.0, [(1.0, snap_burst())])) == []
    two = with_events(2.5, [(1.0, snap_burst()), (1.3, snap_burst())])
    assert len(feed(detector(double=True), two)) == 1


def test_refractory():
    two = with_events(2.5, [(1.0, snap_burst()), (1.2, snap_burst())])
    assert len(feed(detector(), two)) == 1


def test_level_reported():
    det = detector()
    feed(det, quiet(0.5))
    assert 0.0 <= det.level < 0.3


def test_quieter_snap_detected():
    # Peak 0.15: a soft snap at arm's length from a laptop mic.
    for offset in range(0, BLOCK, 64):
        at = 1.0 + offset / SR
        fires = feed(detector(), with_events(2.0, [(at, snap_burst() * 0.3)]))
        assert len(fires) == 1, offset


def test_snap_at_any_block_offset():
    for offset in range(0, BLOCK, 37):
        at = 1.0 + offset / SR
        fires = feed(detector(), with_events(2.0, [(at, snap_burst())]))
        assert len(fires) == 1, offset


def test_faint_click_in_silent_room_ignored():
    # Very quiet room, faint click (keyboard, mouse): big ratio, but too quiet to be a snap.
    x = rng.normal(0, 0.0005, SR * 2).astype(np.float32)
    x[SR : SR + len(snap_burst())] += snap_burst() * 0.06
    assert feed(detector(), x) == []


def percussive_music(seconds=20, vol=0.3):
    t = np.arange(int(SR * seconds)) / SR
    pad = 0.2 * sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330)) / 3
    x = (pad + rng.normal(0, 0.003, len(t))).astype(np.float32)
    hat_n, snare_n = int(SR * 0.03), int(SR * 0.08)
    hat = band_noise(hat_n, 3000, 9000) * np.exp(-np.arange(hat_n) / SR / 0.008)
    snare = band_noise(snare_n, 1000, 5000) * np.exp(-np.arange(snare_n) / SR / 0.02)
    for k in range(int(seconds * 4)):
        i = int((0.5 + k * 0.25) * SR)
        burst = (snare if k % 4 == 2 else hat)[: len(x) - i]
        x[i : i + len(burst)] += (burst * 0.8).astype(np.float32)
    return x * (vol / 0.3)


def test_music_does_not_keep_snapping():
    for double in (False, True):
        fires = feed(detector(double=double), percussive_music())
        assert len(fires) <= 1, (double, fires)


def test_snap_after_music_stops_still_works():
    x = np.concatenate([percussive_music(seconds=4), quiet(1.5)])
    x[int(SR * 5.0) : int(SR * 5.0) + len(snap_burst())] += snap_burst()
    fires = feed(detector(), x)
    assert any(t >= 5.0 for t in fires)


def test_silent_mic_reports_permission():
    det = detector()
    feed(det, np.zeros(int(SR * 2.5), np.float32))
    assert det.mic_error and "Microphone" in det.mic_error
    feed(det, quiet(0.2))
    assert det.mic_error is None
