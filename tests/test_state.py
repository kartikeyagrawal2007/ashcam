from ashcam.state import State, StateMachine


def test_vanish_cycle():
    sm = StateMachine()
    assert sm.toggle_vanish(0.0)
    assert sm.state is State.VANISHING
    assert sm.tick(2.4) is State.VANISHING
    assert sm.tick(2.6) is State.VANISHED
    assert sm.toggle_vanish(3.0)
    assert sm.state is State.RETURNING
    assert sm.tick(5.6) is State.PRESENT


def test_toggle_ignored_mid_transition():
    sm = StateMachine()
    sm.toggle_vanish(0.0)
    assert sm.toggle_vanish(1.0) is False
    assert sm.state is State.VANISHING


def test_away_restores_previous_state():
    sm = StateMachine()
    sm.toggle_vanish(0.0)
    sm.tick(3.0)
    assert sm.state is State.VANISHED
    sm.toggle_away()
    assert sm.away is True
    assert sm.state is State.VANISHED
    sm.toggle_away()
    assert sm.away is False
    assert sm.state is State.VANISHED


def test_progress():
    sm = StateMachine()
    sm.toggle_vanish(0.0)
    assert sm.progress(1.25) == 0.5
