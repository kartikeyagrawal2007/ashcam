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


def test_vanish_ignored_while_away():
    e = make_engine()
    e.process(scene(80), 0.0)
    e.toggle_away()
    e.toggle_vanish()
    e.process(scene(80), 0.1)
    e.toggle_away()
    e.process(scene(80), 3.0)
    assert e.sm.state is State.PRESENT


def test_going_away_cancels_recording():
    e = make_engine()
    e.process(scene(80), 0.0)
    e.away.start_recording(countdown_s=3.0, now=0.0)
    e.toggle_away()
    e.process(scene(80), 0.1)
    assert e.away.status(0.2)[0] == "idle"


class FakeCamera:
    def read(self):
        import time
        time.sleep(0.01)
        return scene(80)


class FlakySegmenter(FakeSegmenter):
    def __init__(self):
        self.calls = 0

    def mask(self, frame):
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("boom")
        return super().mask(frame)


class FakeVcam:
    def __init__(self):
        self.sent = 0

    def send(self, frame):
        self.sent += 1

    def close(self):
        pass


def run_loop(engine, seconds):
    import time
    engine.start()
    time.sleep(seconds)
    alive = engine._thread.is_alive()
    engine.stop()
    return alive


def test_loop_survives_an_exception():
    vcam = FakeVcam()
    e = Engine(FakeCamera(), FlakySegmenter(), AwayLoop(), StateMachine(),
               vcam_factory=lambda: vcam)
    assert run_loop(e, 0.4)
    assert e.segmenter.calls > 5 and vcam.sent > 5
    assert e.status()["error"] is None  # cleared once frames flow again


def test_vcam_retried_until_available():
    vcam = FakeVcam()
    attempts = []

    def factory():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("not installed")
        return vcam

    e = Engine(FakeCamera(), FakeSegmenter(), AwayLoop(), StateMachine(),
               vcam_factory=factory)
    e.VCAM_RETRY_S = 0.05
    assert run_loop(e, 0.5)
    assert e.vcam_ok and vcam.sent > 0 and e.status()["error"] is None


def test_vcam_error_survives_camera_recovery():
    # Camera drops out long enough to raise its own error, then comes back:
    # the virtual camera problem must still be reported.
    class DroppingCamera(FakeCamera):
        n = 0

        def read(self):
            self.n += 1
            return None if self.n <= 40 else super().read()

    def factory():
        raise RuntimeError("not installed")

    e = Engine(DroppingCamera(), FakeSegmenter(), AwayLoop(), StateMachine(),
               vcam_factory=factory)
    e.VCAM_RETRY_S = 10
    run_loop(e, 2.0)
    assert e.camera.n > 45
    assert "OBS Virtual Camera" in (e.status()["error"] or "")


def test_mic_error_reaches_status():
    e = make_engine()

    class Snap:
        level = 0.0
        mic_error = "Allow Microphone"

    e.snap = Snap()
    e.process(scene(80), 0.0)
    assert e.status()["error"] == "Allow Microphone"


def test_coming_back_is_quick():
    e = make_engine()
    e.process(scene(80), 0.0)
    e.toggle_vanish()
    _, t = run_until(e, State.VANISHED, scene(80), 0.1)
    e.toggle_vanish()
    e.process(scene(80), t)
    assert e.sm.state is State.RETURNING
    _, t_end = run_until(e, State.PRESENT, scene(80), t)
    assert t_end - t <= 1.5


def lopsided():
    f = scene(80)
    f[:, : W // 4] = 10   # dark strip on the left only
    return f


def test_output_not_flipped_by_default():
    e = make_engine()
    f = lopsided()
    assert np.array_equal(e.process(f, 0.0), f)


def test_flip_output_mirrors_what_the_call_sees():
    import cv2
    e = make_engine()
    e.flip_output = True
    f = lopsided()
    assert np.array_equal(e.process(f, 0.0), cv2.flip(f, 1))


def test_flip_survives_coming_back():
    import cv2
    e = make_engine()
    e.flip_output = True
    e.process(lopsided(), 0.0)
    e.toggle_vanish()
    _, t = run_until(e, State.VANISHED, lopsided(), 0.1)
    e.toggle_vanish()
    _, t = run_until(e, State.PRESENT, lopsided(), t)
    out = e.process(lopsided(), t + 0.5)
    assert np.array_equal(out, cv2.flip(lopsided(), 1))


class ArmMissingSegmenter(FakeSegmenter):
    """Detection flickers: every other frame it misses the right part (an arm)."""
    def __init__(self):
        self.n = 0

    def mask(self, frame):
        self.n += 1
        m = super().mask(frame)
        if self.n % 2:
            m[BOX[0], 70:80] = 0
        return m


def test_flickering_arm_does_not_leak():
    e = Engine(None, ArmMissingSegmenter(), AwayLoop(), StateMachine())
    for i in range(5):
        e.process(scene(80), i / 30)
    e.toggle_vanish()
    _, t = run_until(e, State.VANISHED, scene(80), 0.2)
    for i in range(10):
        out = e.process(scene(80), t + i / 30)
        assert out[BOX].max() < 150, i


def test_vanished_follows_exposure_change():
    e = make_engine()
    for i in range(5):
        e.process(scene(120), i / 30)
    e.toggle_vanish()
    _, t = run_until(e, State.VANISHED, scene(120), 0.2)
    out = e.process(scene(90), t + 0.1)      # camera darkened the whole room
    assert abs(int(out[36, 64].mean()) - 90) <= 8


class SwitchableSegmenter(FakeSegmenter):
    present = True

    def mask(self, frame):
        m = super().mask(frame)
        return m if self.present else m * 0


def test_capture_room_waits_until_you_step_out():
    seg = SwitchableSegmenter()
    e = Engine(None, seg, AwayLoop(), StateMachine())
    e.capture_room(countdown_s=0.0)
    for i in range(10):
        e.process(scene(80), i / 30)
    assert e.status()["room"] == "waiting"
    seg.present = False
    for i in range(10, 20):
        e.process(scene(80, person=80), i / 30)
    s = e.status()
    assert s["room"] == "idle" and s["room_saved"] is True
    assert e._plate.known.all()


def test_capture_room_counts_down_first():
    e = make_engine()
    e.capture_room(countdown_s=5.0)
    e.process(scene(80), 0.0)
    assert e.status()["room"] == "countdown"


class FakeMuter:
    def __init__(self):
        self.calls, self.muted, self.enabled, self.error = [], False, True, None

    def mute(self):
        self.calls.append("mute")
        self.muted = True

    def unmute(self):
        self.calls.append("unmute")
        self.muted = False

    def close(self):
        self.unmute()


def test_vanishing_mutes_and_coming_back_unmutes():
    from ashcam.gesture import SnapFusion
    e = make_engine()
    from ashcam.snap import SnapDetector
    e.muter = FakeMuter()
    e.snap = SnapDetector(on_snap=lambda: None)
    e.fusion = SnapFusion(on_snap=e.toggle_vanish)
    e.process(scene(80), 0.0)
    e.toggle_vanish()
    e.process(scene(80), 0.1)
    assert e.muter.calls == ["mute"] and e.fusion.mic_live is False
    _, t = run_until(e, State.VANISHED, scene(80), 0.2)
    e.toggle_vanish()
    e.process(scene(80), t)
    assert e.muter.calls == ["mute", "unmute"] and e.fusion.mic_live is True


def test_frames_reach_the_hand_tracker():
    class FakeGesture:
        hand_seen = True

        def __init__(self):
            self.frames = 0

        def submit(self, frame, now):
            self.frames += 1

    e = make_engine()
    e.gesture = FakeGesture()
    for i in range(3):
        e.process(scene(80), i / 30)
    assert e.gesture.frames == 3
    s = e.status()
    assert s["hands_on"] is True and s["hand_seen"] is True


def test_status_reports_mute():
    e = make_engine()
    e.muter = FakeMuter()
    e.process(scene(80), 0.0)
    s = e.status()
    assert s["mute_mic"] is True and s["mic_muted"] is False


def test_snapping_asks_for_an_ai_fill():
    class FakeAI:
        state, latest = "idle", None

        def __init__(self):
            self.requests = 0

        def request(self, img, hole, tag=None):
            self.requests += 1

    e = make_engine()
    e.ai_filler = FakeAI()
    for i in range(5):
        e.process(scene(80), i / 30)
    assert e.ai_filler.requests == 0
    e.toggle_vanish()
    e.process(scene(80), 0.2)
    assert e.ai_filler.requests == 1
    assert e.status()["ai_fill"] == "idle"


# --- full invisibility ---------------------------------------------------

ROOM = 150


def room_frame(person=True, shadow=False, hair=False, blob_at=None):
    f = np.full((H, W, 3), ROOM, np.uint8)
    if shadow:
        f[20:60, 82:96] = 95                 # your shadow on the wall, right of you
    if hair:
        f[50:60, 8:24] = 250                 # an arm the detector is unsure about
    if person:
        f[BOX] = 250
    if blob_at is not None:                  # a moving hand just outside the outline
        x = blob_at
        f[30:40, x:x + 6] = 250
    return f


class RoomSegmenter(FakeSegmenter):
    present = True
    hair = False

    def mask(self, frame):
        m = super().mask(frame)
        if not self.present:
            return m * 0
        if self.hair:
            m[50:60, 8:24] = 0.3             # faint, but it's you
        return m


def vanished_engine(seg, frames_before, t=0.0):
    e = Engine(None, seg, AwayLoop(), StateMachine())
    for f in frames_before:
        e.process(f, t)
        t += 1 / 30
    e.toggle_vanish()
    _, t = run_until(e, State.VANISHED, frames_before[-1], t)
    return e, t


def test_room_is_saved_automatically_when_you_step_away():
    seg = RoomSegmenter()
    e = Engine(None, seg, AwayLoop(), StateMachine())
    for i in range(10):
        e.process(room_frame(), i / 30)
    assert not e.room_saved
    seg.present = False
    for i in range(10, 50):
        e.process(room_frame(person=False), i / 30)
    assert e.room_saved and e._plate.room_ref is not None


def test_room_not_saved_while_you_are_there():
    e = Engine(None, RoomSegmenter(), AwayLoop(), StateMachine())
    for i in range(60):
        e.process(room_frame(), i / 30)
    assert not e.room_saved


def stepped_away_then_back(seg, **kw):
    seg.present = False
    frames = [room_frame(person=False)] * 40
    seg.present = True
    return frames


def test_shadow_disappears_with_you():
    seg = RoomSegmenter()
    seg.present = False
    e = Engine(None, seg, AwayLoop(), StateMachine())
    t = 0.0
    for _ in range(40):                       # you step out once: room saved
        e.process(room_frame(person=False), t); t += 1 / 30
    seg.present = True
    for _ in range(10):                       # you sit back down, casting a shadow
        e.process(room_frame(shadow=True), t); t += 1 / 30
    e.toggle_vanish()
    _, t = run_until(e, State.VANISHED, room_frame(shadow=True), t)
    out = e.process(room_frame(shadow=True), t)
    assert abs(int(out[25:55, 85:94].mean()) - ROOM) <= 10


def test_faint_hair_is_hidden_completely():
    seg = RoomSegmenter()
    seg.hair = True
    e, t = vanished_engine(seg, [room_frame(hair=True)] * 10)
    out = e.process(room_frame(hair=True), t)
    assert out[50:60, 8:24].max() < 200


def test_moving_hand_next_to_you_is_hidden():
    seg = RoomSegmenter()
    e, t = vanished_engine(seg, [room_frame()] * 10)
    for i in range(6):
        x = 82 + 2 * i                        # hand moving right, outside the outline
        out = e.process(room_frame(blob_at=x), t + i / 30)
    assert out[30:40, x:x + 6].max() < 200


def test_tracked_hand_is_hidden():
    class FakeGesture:
        hand_seen = True

        def submit(self, frame, now):
            pass

        def hand_points(self, now, shape):
            return np.array([[100, 28], [112, 28], [112, 44], [100, 44]], np.float32)

    seg = RoomSegmenter()
    e = Engine(None, seg, AwayLoop(), StateMachine())
    e.gesture = FakeGesture()
    t = 0.0
    for _ in range(10):
        e.process(room_frame(), t); t += 1 / 30
    e.toggle_vanish()
    _, t = run_until(e, State.VANISHED, room_frame(), t)
    f = room_frame()
    f[28:44, 100:112] = 250                   # raised hand far from the body outline
    out = e.process(f, t)
    assert out[30:42, 102:110].max() < 200


def test_no_ai_guessing_once_the_room_is_saved():
    class FakeAI:
        state, latest = "idle", None
        requests = 0

        def request(self, img, hole, tag=None):
            FakeAI.requests += 1

    seg = RoomSegmenter()
    seg.present = False
    e = Engine(None, seg, AwayLoop(), StateMachine())
    e.ai_filler = FakeAI()
    t = 0.0
    for _ in range(40):
        e.process(room_frame(person=False), t); t += 1 / 30
    seg.present = True
    for _ in range(5):
        e.process(room_frame(), t); t += 1 / 30
    e.toggle_vanish()
    e.process(room_frame(), t)
    assert FakeAI.requests == 0
