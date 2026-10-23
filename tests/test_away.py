import numpy as np

from ashcam.away import AwayLoop, pingpong_index


def frame(v):
    f = np.zeros((48, 64, 3), np.uint8)
    f[:] = (v, 255 - v, (v * 3) % 256)
    return f


def test_pingpong():
    assert [pingpong_index(i, 4) for i in range(8)] == [0, 1, 2, 3, 2, 1, 0, 1]
    assert all(pingpong_index(i, 1) == 0 for i in range(5))


def test_empty_returns_none():
    assert AwayLoop().next_frame() is None


def test_falls_back_to_rolling():
    loop = AwayLoop()
    pushed = [frame(i * 20) for i in range(10)]
    for i, f in enumerate(pushed):
        loop.push(f, now=i / 30)
    out = loop.next_frame()
    assert out is not None
    diffs = [np.abs(out.astype(int) - p.astype(int)).mean() for p in pushed]
    assert min(diffs) < 3


def test_recording_flow():
    loop = AwayLoop(fps=30, clip_s=15)
    loop.start_recording(countdown_s=3.0, now=0.0)
    assert loop.status(1.0)[0] == "countdown"
    loop.push(frame(10), now=1.0)
    assert not loop.has_clip
    for i in range(450):
        loop.push(frame(i % 256), now=3.0 + i / 30)
    assert loop.has_clip
    assert loop.status(3.0 + 450 / 30)[0] == "idle"
