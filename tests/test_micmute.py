from ashcam.micmute import MicMuter


class FakeMac:
    """Stands in for osascript: holds the input volume."""
    def __init__(self, volume=72):
        self.volume = volume

    def run(self, script):
        if script.startswith("input volume of"):
            return str(self.volume)
        if script.startswith("set volume input volume "):
            self.volume = int(script.rsplit(" ", 1)[1])
            return ""
        raise AssertionError(script)


def make(tmp_path, volume=72):
    mac = FakeMac(volume)
    return MicMuter(run=mac.run, state_file=str(tmp_path / "mic"), sync=True), mac


def test_mute_then_restore(tmp_path):
    m, mac = make(tmp_path)
    m.mute()
    assert mac.volume == 0 and m.muted
    m.unmute()
    assert mac.volume == 72 and not m.muted


def test_mute_twice_keeps_original_level(tmp_path):
    m, mac = make(tmp_path)
    m.mute()
    m.mute()
    m.unmute()
    assert mac.volume == 72


def test_disabled_does_nothing(tmp_path):
    m, mac = make(tmp_path)
    m.enabled = False
    m.mute()
    assert mac.volume == 72 and not m.muted


def test_turning_off_while_muted_restores(tmp_path):
    m, mac = make(tmp_path)
    m.mute()
    m.set_enabled(False)
    assert mac.volume == 72 and not m.muted


def test_crash_while_muted_is_repaired_next_start(tmp_path):
    m, mac = make(tmp_path)
    m.mute()                       # app dies here, mic left at 0
    again = MicMuter(run=mac.run, state_file=str(tmp_path / "mic"), sync=True)
    again.restore_leftover()
    assert mac.volume == 72
    assert not (tmp_path / "mic").exists()


def test_failure_never_raises(tmp_path):
    def broken(script):
        raise OSError("osascript missing")
    m = MicMuter(run=broken, state_file=str(tmp_path / "mic"), sync=True)
    m.mute()
    m.unmute()
    assert m.error
