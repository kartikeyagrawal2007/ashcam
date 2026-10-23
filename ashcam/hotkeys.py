"""Global ⌥⇧V / ⌥⇧A, working while the call app has focus."""

from typing import Callable

from pynput import keyboard
from pynput.keyboard import Key

# macOS virtual keycodes. Matching on the character fails because ⌥⇧V types "◊".
VK_V = 9
VK_A = 0
_ALT = {Key.alt, Key.alt_l, Key.alt_r}
_SHIFT = {Key.shift, Key.shift_l, Key.shift_r}


class Hotkeys:
    def __init__(self, on_vanish: Callable[[], None], on_away: Callable[[], None]):
        self._actions = {VK_V: on_vanish, VK_A: on_away}
        self._alt = 0
        self._shift = 0
        self._listener: keyboard.Listener | None = None

    def _press(self, key) -> None:
        if key in _ALT:
            self._alt += 1
        elif key in _SHIFT:
            self._shift += 1
        elif self._alt and self._shift:
            action = self._actions.get(getattr(key, "vk", None))
            if action:
                action()

    def _release(self, key) -> None:
        if key in _ALT:
            self._alt = max(0, self._alt - 1)
        elif key in _SHIFT:
            self._shift = max(0, self._shift - 1)

    def start(self) -> bool:
        """Start listening. False means macOS hasn't trusted this app for input."""
        self._listener = keyboard.Listener(on_press=self._press, on_release=self._release)
        self._listener.start()
        self._listener.wait()
        return bool(getattr(self._listener, "IS_TRUSTED", True))

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
