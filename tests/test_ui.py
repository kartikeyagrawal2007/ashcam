import base64
import time

import cv2
import numpy as np

from ashcam.away import AwayLoop
from ashcam.engine import Engine
from ashcam.snap import SnapDetector
from ashcam.state import StateMachine
from ashcam.ui import Api


class FakeSegmenter:
    def mask(self, frame):
        return np.zeros(frame.shape[:2], np.float32)


def make():
    away = AwayLoop()
    engine = Engine(None, FakeSegmenter(), away, StateMachine())
    snap = SnapDetector(on_snap=engine.toggle_vanish)
    engine.snap = snap
    return Api(engine, snap, away), engine, snap, away


def test_frame_is_small_jpeg():
    api, engine, _, _ = make()
    assert api.frame() is None
    engine.process(np.full((720, 1280, 3), 90, np.uint8), 0.0)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(api.frame()), np.uint8), 1)
    assert img.shape == (360, 640, 3)


def test_status_has_snap_settings():
    api, engine, snap, _ = make()
    engine.process(np.zeros((720, 1280, 3), np.uint8), 0.0)
    s = api.status()
    assert s["state"] == "present"
    assert s["snap_on"] is True
    assert s["sensitivity"] == 0.5 and s["double"] is False
    assert abs(s["threshold"] - 0.5) < 1e-6


def test_settings_are_clamped():
    api, _, snap, _ = make()
    api.set_sensitivity(3)
    assert snap.sensitivity == 1.0
    api.set_sensitivity("0.25")
    assert snap.sensitivity == 0.25
    api.set_double(True)
    assert snap.double is True


def test_record_starts_countdown():
    api, _, _, away = make()
    api.record()
    assert away.status(time.monotonic())[0] == "countdown"


def test_buttons_queue_toggles():
    api, engine, _, _ = make()
    api.vanish()
    api.away()
    engine.process(np.zeros((720, 1280, 3), np.uint8), 0.0)
    assert engine.sm.state.value == "vanishing" and engine.sm.away is True


def test_works_without_mic():
    away = AwayLoop()
    engine = Engine(None, FakeSegmenter(), away, StateMachine())
    api = Api(engine, None, away)
    api.set_sensitivity(0.7)
    assert api.status()["snap_on"] is False


def test_preview_is_mirrored_like_a_selfie():
    api, engine, _, _ = make()
    f = np.full((720, 1280, 3), 200, np.uint8)
    f[:, :320] = 0                     # dark on the left of the real image
    engine.process(f, 0.0)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(api.frame()), np.uint8), 1)
    assert img[:, -80:].mean() < 30 and img[:, :80].mean() > 170


def test_flip_switch():
    api, engine, _, _ = make()
    assert api.status()["flip"] is False
    api.set_flip(True)
    assert engine.flip_output is True and api.status()["flip"] is True


def test_capture_room_button():
    api, engine, _, _ = make()
    engine.process(np.zeros((720, 1280, 3), np.uint8), 0.0)
    assert api.status()["room_saved"] is False
    api.capture_room()
    engine.process(np.zeros((720, 1280, 3), np.uint8), 0.1)
    assert api.status()["room"] == "countdown"
