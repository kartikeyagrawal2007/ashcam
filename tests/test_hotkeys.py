from pynput.keyboard import Key, KeyCode

from ashcam.hotkeys import Hotkeys


def make():
    calls = []
    hk = Hotkeys(on_vanish=lambda: calls.append("v"), on_away=lambda: calls.append("a"))
    return hk, calls


def press(hk, *keys):
    for k in keys:
        hk._press(k)
    for k in reversed(keys):
        hk._release(k)


def test_option_shift_v_vanishes():
    hk, calls = make()
    press(hk, Key.alt, Key.shift, KeyCode(vk=9, char="◊"))
    assert calls == ["v"]


def test_option_shift_a_goes_away():
    hk, calls = make()
    press(hk, Key.alt_r, Key.shift_l, KeyCode(vk=0, char="Å"))
    assert calls == ["a"]


def test_needs_both_modifiers():
    hk, calls = make()
    press(hk, Key.alt, KeyCode(vk=9, char="√"))
    press(hk, Key.shift, KeyCode(vk=9, char="V"))
    press(hk, KeyCode(vk=9, char="v"))
    assert calls == []


def test_released_modifier_no_longer_counts():
    hk, calls = make()
    press(hk, Key.alt, Key.shift)
    hk._press(KeyCode(vk=9, char="v"))
    assert calls == []
