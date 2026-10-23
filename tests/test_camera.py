import numpy as np

from ashcam.camera import Camera


class FakeCap:
    def __init__(self, opened):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, *args):
        return True

    def read(self):
        if not self.opened:
            return False, None
        return True, np.zeros((720, 1280, 3), np.uint8)

    def release(self):
        self.released = True


def test_reopens_once_access_is_allowed():
    caps = []

    def opener(source):
        # macOS blocks the first two attempts, then the user clicks Allow.
        cap = FakeCap(opened=len(caps) >= 2)
        caps.append(cap)
        return cap

    cam = Camera(0, opener=opener, retry_s=0.0)
    assert not cam.ok
    assert cam.read() is None          # second attempt, still blocked
    frame = cam.read()                 # third attempt succeeds
    assert frame is not None and frame.shape == (720, 1280, 3)
    assert cam.ok
    assert caps[0].released and caps[1].released


def test_waits_between_attempts():
    caps = []

    def opener(source):
        caps.append(FakeCap(opened=False))
        return caps[-1]

    cam = Camera(0, opener=opener, retry_s=60.0)
    for _ in range(5):
        assert cam.read() is None
    assert len(caps) == 1
