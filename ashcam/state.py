"""Vanish state machine. Pure and time-driven so it is easy to test."""

from enum import Enum


class State(Enum):
    PRESENT = "present"
    VANISHING = "vanishing"
    VANISHED = "vanished"
    RETURNING = "returning"


_NEXT = {State.VANISHING: State.VANISHED, State.RETURNING: State.PRESENT}


class StateMachine:
    def __init__(self, vanish_s: float = 2.5, return_s: float = 2.5):
        self.vanish_s = vanish_s
        self.return_s = return_s
        self.state = State.PRESENT
        self.away = False
        self.started_at = 0.0

    def _duration(self) -> float:
        return self.vanish_s if self.state is State.VANISHING else self.return_s

    def toggle_vanish(self, now: float) -> bool:
        self.tick(now)
        if self.state is State.PRESENT:
            self.state = State.VANISHING
        elif self.state is State.VANISHED:
            self.state = State.RETURNING
        else:
            return False
        self.started_at = now
        return True

    def toggle_away(self) -> None:
        # Away is a flag on top of the vanish state, so turning it off
        # lands you back wherever you were.
        self.away = not self.away

    def tick(self, now: float) -> State:
        if self.state in _NEXT and now - self.started_at >= self._duration():
            self.state = _NEXT[self.state]
        return self.state

    def progress(self, now: float) -> float:
        if self.state not in _NEXT:
            return 1.0
        return min(1.0, max(0.0, (now - self.started_at) / self._duration()))
