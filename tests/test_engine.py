import numpy as np

from ashcam.away import AwayLoop
from ashcam.engine import Engine
from ashcam.state import State, StateMachine

H, W = 72, 128
BOX = (slice(20, 52), slice(48, 80))


class FakeSegmenter:
    def mask(self, frame):
        m = np.zeros(frame.shape[:2], np.float32)
        m[BOX] = 1.0
        return m


def scene(bg, person=250):
    f = np.full((H, W, 3), bg, np.uint8)
    f[BOX] = person
    return f


def make_engine():
    return Engine(camera=None, segmenter=FakeSegmenter(), away=AwayLoop(),
                  sm=StateMachine())


def run_until(engine, state, frame, t0, step=1 / 30, limit=5.0):
    t = t0
    while engine.sm.state is not state and t < t0 + limit:
        out = engine.process(frame, t)
        t += step
    return engine.process(frame, t), t


def test_present_passthrough():
    e = make_engine()
    f = scene(80)
    assert np.array_equal(e.process(f, 0.0), f)


def test_vanished_shows_plate():
    e = make_engine()
    for i in range(5):
        e.process(scene(80), i / 30)
    e.toggle_vanish()
    out, _ = run_until(e, State.VANISHED, scene(80), 0.2)
    assert e.sm.state is State.VANISHED
    assert abs(int(out[36, 64].mean()) - 80) <= 5


def test_live_background_when_vanished():
    e = make_engine()
    e.process(scene(80), 0.0)
    e.toggle_vanish()
    _, t = run_until(e, State.VANISHED, scene(80), 0.1)
    out = e.process(scene(140), t + 0.1)
    assert abs(int(out[2, 2].mean()) - 140) <= 2
    assert abs(int(out[H - 3, W - 3].mean()) - 140) <= 2


def test_returns_to_person():
    e = make_engine()
    e.process(scene(80), 0.0)
    e.toggle_vanish()
    _, t = run_until(e, State.VANISHED, scene(80), 0.1)
    e.toggle_vanish()
    out, t = run_until(e, State.PRESENT, scene(80), t)
    out = e.process(scene(80), t + 0.5)  # past the cross-fade
    assert np.array_equal(out, scene(80))


def test_away_plays_clip():
    e = make_engine()
    clip = [scene(30 + i) for i in range(10)]
    for i, f in enumerate(clip):
        e.process(f, i / 30)
    e.toggle_away()
    out = e.process(scene(200, person=0), 1.0)
    diffs = [np.abs(out.astype(int) - c.astype(int)).mean() for c in clip]
    assert min(diffs) < 3


def test_dropped_frame_repeats_last():
    e = make_engine()
    first = e.process(scene(80), 0.0)
    again = e.process(None, 0.033)
    assert np.array_equal(again, first)


def test_dropped_first_frame_is_black():
    e = make_engine()
    out = e.process(None, 0.0)
    assert out.shape == (720, 1280, 3) and not out.any()


def test_status_fields():
    e = make_engine()
    e.process(scene(80), 0.0)
    s = e.status()
    assert s["state"] == "present" and s["away"] is False
    assert {"away_rec", "snap_level", "vcam_ok", "error"} <= set(s)
