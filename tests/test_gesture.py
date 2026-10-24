import numpy as np

from ashcam.gesture import HandSnap, SnapFusion

FPS = 30


def hand(thumb, middle):
    """21 landmarks in pixels: wrist + middle knuckle fix the hand size (100 px)."""
    lm = np.zeros((21, 2), np.float32)
    lm[0] = (640, 500)     # wrist
    lm[9] = (640, 400)     # middle finger knuckle
    lm[4] = thumb
    lm[12] = middle
    return lm


PINCH = hand((650, 330), (660, 335))     # thumb and middle tip together
OPEN = hand((590, 360), (680, 260))      # snapped apart
HALF = hand((630, 340), (665, 315))      # partly open


def run(det, frames):
    events = []
    for i, lm in enumerate(frames):
        if det.process(lm, i / FPS):
            events.append(i)
    return events


def test_snap_is_pinch_then_fast_release():
    frames = [OPEN] * 5 + [PINCH] * 10 + [OPEN] * 5
    assert run(HandSnap(), frames) == [15]


def test_pinch_that_never_opens_is_not_a_snap():
    assert run(HandSnap(), [OPEN] * 5 + [PINCH] * 30) == []


def test_slow_opening_is_not_a_snap():
    frames = [PINCH] * 10 + [HALF] * 15 + [OPEN] * 5
    assert run(HandSnap(), frames) == []


def test_hand_lost_resets():
    frames = [PINCH] * 10 + [None] * 3 + [OPEN] * 3
    assert run(HandSnap(), frames) == []


def test_waving_open_hand_never_snaps():
    assert run(HandSnap(), [OPEN, HALF] * 20) == []


def fusion(**kw):
    fired = []
    f = SnapFusion(on_snap=lambda: fired.append(1), **kw)
    return f, fired


def test_needs_camera_and_mic_together():
    f, fired = fusion()
    f.visual(10.0)
    f.audio(10.2)
    assert fired == [1]


def test_mic_click_alone_does_nothing():
    f, fired = fusion()
    f.audio(10.0)
    f.audio(12.0)
    assert fired == []


def test_hand_alone_does_nothing_while_mic_is_live():
    f, fired = fusion()
    f.visual(10.0)
    assert fired == []


def test_too_far_apart_in_time():
    f, fired = fusion()
    f.visual(10.0)
    f.audio(11.0)
    assert fired == []


def test_hand_alone_counts_while_mic_is_muted():
    f, fired = fusion()
    f.mic_live = False
    f.visual(10.0)
    assert fired == [1]


def test_mic_alone_counts_without_hand_tracking():
    f, fired = fusion()
    f.hands_ok = False
    f.audio(10.0)
    assert fired == [1]


def test_one_snap_fires_once():
    f, fired = fusion()
    f.visual(10.0)
    f.audio(10.1)
    f.audio(10.15)
    f.visual(10.2)
    assert fired == [1]


def test_worker_reports_a_seen_snap():
    from ashcam.gesture import GestureWorker

    class ScriptedTracker:
        def __init__(self, frames):
            self.frames = list(frames)

        def landmarks(self, frame, now):
            assert frame.shape[1] == GestureWorker.WIDTH
            return self.frames.pop(0)

    f, fired = fusion()
    f.mic_live = False
    w = GestureWorker(ScriptedTracker([OPEN] * 3 + [PINCH] * 5 + [OPEN] * 2), f)
    img = np.zeros((720, 1280, 3), np.uint8)
    for i in range(10):
        w.submit(img, i / FPS)
        w.step()
    assert fired == [1] and w.hand_seen


def test_worker_shares_hand_points_in_frame_pixels():
    from ashcam.gesture import GestureWorker

    class OneHand:
        def landmarks(self, frame, now):
            lm = np.zeros((21, 2), np.float32)
            lm[:, 0], lm[:, 1] = 320, 180           # centre of the 640x360 tracking frame
            return lm

    f, _ = fusion()
    w = GestureWorker(OneHand(), f)
    w.submit(np.zeros((720, 1280, 3), np.uint8), 1.0)
    w.step()
    pts = w.hand_points(now=1.1, shape=(720, 1280))
    assert pts is not None and np.allclose(pts[0], (640, 360))
    assert w.hand_points(now=2.0, shape=(720, 1280)) is None   # stale: hand gone
