"""Frame loop: camera -> person mask -> composite per state -> virtual camera."""

import threading
import time
from collections import deque
from typing import Callable

import cv2
import numpy as np

from .away import AwayLoop
from .background import BackgroundPlate
from .dust import DustEffect
from .state import State, StateMachine

OUT_SIZE = (1280, 720)
XFADE_S = 0.2
_GROW = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # at 1/4 res: ~9 px each side
_HIDE_CORE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))    # at 1/4 res: ~12 px
_HIDE_GROW = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))  # at 1/4 res: ~24 px
# Square kernels for the big reaches: OpenCV grows rectangles far faster.
_SHADOW_REACH = cv2.getStructuringElement(cv2.MORPH_RECT, (101, 101))  # ~200 px around you
_MOTION_REACH = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))    # ~100 px around you
_SMALL3 = np.ones((3, 3), np.uint8)
PERSON_THRESHOLD = 0.15  # anything the detector is even a little sure is you
AUTO_ROOM_FRAMES = 30    # ~1 s with nobody in view: save the empty room
AUTO_ROOM_EVERY_S = 30.0 # ...and refresh it at most this often
MASK_MEMORY = 8          # frames of person outline kept when hiding
ROOM_CLEAR_FRAMES = 5
AI_REFRESH_S = 5.0       # re-run the AI fill this often while gone, if the room grew    # frames with nobody in view before the room is saved

VCAM_HELP = ("OBS Virtual Camera isn't available. Open OBS, click Start Virtual Camera, "
             "then Stop, and quit OBS. If it still fails, turn it on in System Settings → "
             "General → Login Items & Extensions → Camera Extensions. AshCam keeps "
             "retrying, so there's no need to restart it.")
CAMERA_HELP = ("Can't read the camera. Allow your terminal app in System Settings → "
               "Privacy & Security → Camera. AshCam picks it up automatically.")


def open_obs_vcam():
    import pyvirtualcam

    return pyvirtualcam.Camera(OUT_SIZE[0], OUT_SIZE[1], 30,
                               fmt=pyvirtualcam.PixelFormat.RGB, backend="obs")


def feather(mask: np.ndarray) -> np.ndarray:
    """Grow the person mask a little and soften it, so no outline is left behind."""
    h, w = mask.shape
    small = cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(cv2.dilate(small, _GROW), (5, 5), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def composite(frame: np.ndarray, plate: np.ndarray, hide: np.ndarray) -> np.ndarray:
    """Live frame everywhere, the remembered room where `hide` is 1."""
    hide = np.clip(hide, 0, 1).astype(np.float32)
    return cv2.blendLinear(frame, plate, 1 - hide, hide)


class Engine:
    VCAM_RETRY_S = 3.0

    def __init__(self, camera, segmenter, away: AwayLoop, sm: StateMachine,
                 sink: Callable[[np.ndarray], None] | None = None,
                 vcam_factory: Callable[[], object] = open_obs_vcam):
        self.camera = camera
        self.segmenter = segmenter
        self.away = away
        self.sm = sm
        self.sink = sink
        self.vcam_factory = vcam_factory
        self.snap = None          # SnapDetector, set by main for the level meter
        self.fusion = None        # SnapFusion: camera + mic snap agreement
        self.gesture = None       # GestureWorker: hand tracking on its own thread
        self.muter = None         # MicMuter: silences the mic while you're gone
        self.ai_filler = None     # BackgroundFiller: LaMa fill of the never-seen room
        self._ai_asked_at = (0.0, -1)
        self.latest: np.ndarray | None = None
        self.vcam_ok = False
        # Off: the call gets the true image (text behind you reads correctly).
        self.flip_output = False
        self._composed: np.ndarray | None = None   # last output before any flip
        self._mask_hist: deque[np.ndarray] = deque(maxlen=MASK_MEMORY)
        self._prev_gray: np.ndarray | None = None
        self._empty_frames = 0
        self._auto_room_at = -1e9
        self.room = "idle"            # idle | countdown | waiting
        self.room_saved = False
        self._room_at = 0.0
        self._room_clear = 0
        # Kept apart so one recovering can't hide another that hasn't.
        self.loop_error: str | None = None
        self.camera_error: str | None = None
        self.vcam_error: str | None = None
        self._plate: BackgroundPlate | None = None
        self._fx: DustEffect | None = None
        self._fx_mask: np.ndarray | None = None
        self._xfade_from: np.ndarray | None = None
        self._xfade_start = 0.0
        self._was_away = False
        self._pending: list[str] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    # Called from hotkey, audio and UI threads; applied on the next frame.
    def toggle_vanish(self) -> None:
        with self._lock:
            self._pending.append("vanish")

    def toggle_away(self) -> None:
        with self._lock:
            self._pending.append("away")

    def capture_room(self, countdown_s: float = 5.0) -> None:
        """Count down, wait until nobody is in view, then save the whole room."""
        with self._lock:
            self._pending.append(("room", countdown_s))

    @property
    def error(self) -> str | None:
        mic = getattr(self.snap, "mic_error", None) if self.snap else None
        mute = getattr(self.muter, "error", None) if self.muter else None
        return self.loop_error or self.camera_error or self.vcam_error or mic or mute

    def status(self) -> dict:
        rec, left = self.away.status(time.monotonic())
        return {
            "state": self.sm.state.value,
            "away": self.sm.away,
            "away_rec": rec,
            "away_rec_left": round(left, 1),
            "has_clip": self.away.has_clip,
            "snap_level": float(self.snap.level) if self.snap else 0.0,
            "hands_on": self.gesture is not None,
            "hand_seen": bool(getattr(self.gesture, "hand_seen", False)),
            "mute_mic": bool(self.muter and self.muter.enabled),
            "mic_muted": bool(self.muter and self.muter.muted),
            "vcam_ok": self.vcam_ok,
            "ai_fill": getattr(self.ai_filler, "state", "off") if self.ai_filler else "off",
            "room": self.room,
            "room_left": round(max(0.0, self._room_at - time.monotonic()), 1),
            "room_saved": self.room_saved,
            "error": self.error,
        }

    def _apply_pending(self, frame: np.ndarray, mask: np.ndarray, now: float) -> None:
        with self._lock:
            pending, self._pending = self._pending, []
        for action in pending:
            if isinstance(action, tuple) and action[0] == "room":
                self.room, self._room_at, self._room_clear = "countdown", now + action[1], 0
            elif action == "away":
                self.sm.toggle_away()
                self.away.reset_playhead()
                if self.sm.away:
                    # Recording while away would capture nothing (or the empty chair).
                    self.away.cancel_recording()
            elif self.sm.away:
                continue  # vanish is off while away; coming back must show you
            elif self.sm.toggle_vanish(now):
                # Vanishing crumbles the person as they are now; returning
                # reassembles the person who is sitting there now.
                self._fx = DustEffect(frame, mask, assemble_s=self.sm.return_s,
                                      seed=int(now * 1000) % 2**32)
                self._fx_mask = mask
                if self.sm.state is State.VANISHING and self._plate is not None:
                    self._ask_ai_fill(now)
                if self.muter is not None:
                    if self.sm.state is State.VANISHING:
                        self.muter.mute()
                    else:
                        self.muter.unmute()
        self._sync_mic()

    def _sync_mic(self) -> None:
        """Tell the snap detectors whether the mic can be trusted right now."""
        muted = bool(self.muter and self.muter.muted)
        if self.snap is not None:
            self.snap.suspended = muted
        if self.fusion is not None:
            self.fusion.mic_live = (self.snap is not None and not muted
                                    and getattr(self.snap, "mic_error", None) is None)

    def process(self, frame: np.ndarray | None, now: float) -> np.ndarray:
        if frame is None:
            if self.latest is None:
                self.latest = np.zeros((OUT_SIZE[1], OUT_SIZE[0], 3), np.uint8)
            return self.latest
        out = self._compose(frame, now)
        self._composed = out
        # Flip only on the way out, so the room memory, effects and away clip
        # all stay in the camera's own orientation.
        self.latest = cv2.flip(out, 1) if self.flip_output else out
        return self.latest

    def _compose(self, frame: np.ndarray, now: float) -> np.ndarray:

        if self.gesture is not None:
            self.gesture.submit(frame, now)
        mask = self.segmenter.mask(frame)
        self._apply_pending(frame, mask, now)
        before = self.sm.state
        state = self.sm.tick(now)

        if self.sm.away:
            self._was_away = True
            out = self.away.next_frame()
            if out is None:
                out = frame
            elif out.shape != frame.shape:
                out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
            return out
        if self._was_away:
            self._was_away = False
            self.away.reset_playhead()

        # Never record the away fallback while away, or it fills with you leaving.
        self.away.push(frame, now)
        if self._plate is None or self._plate.plate.shape != frame.shape:
            self._plate = BackgroundPlate(frame.shape)
        self._update_room_capture(frame, mask, now)
        self._auto_save_room(frame, mask, now)
        you = self._everything_of_you(frame, mask, now)
        # The room memory must not learn your shadow, hand or motion either.
        h, w = mask.shape
        self._plate.update(frame, np.maximum(mask, cv2.resize(
            you, (w, h), interpolation=cv2.INTER_NEAREST)))
        self._mask_hist.append(you)

        if state is State.PRESENT:
            out = frame
            if before is State.RETURNING:
                self._xfade_from, self._xfade_start = self._composed, now
            if self._xfade_from is not None:
                k = (now - self._xfade_start) / XFADE_S
                if k >= 1 or self._xfade_from.shape != frame.shape:
                    self._xfade_from = None
                else:
                    out = cv2.addWeighted(self._xfade_from, 1 - k, frame, k, 0)
        elif state is State.VANISHED:
            asked, count = self._ai_asked_at
            if (self.ai_filler is not None and now - asked > AI_REFRESH_S
                    and abs(int(self._plate.known.sum()) - count) > 0.005 * self._plate.known.size):
                self._ask_ai_fill(now)
            out = composite(frame, self._plate.matched(frame, mask), self._hide_mask())
        else:
            hide = self._hide_mask(extra=self._fx_mask)
            base = composite(frame, self._plate.matched(frame, mask), hide)
            t = now - self.sm.started_at
            if state is State.RETURNING:
                out = self._fx.render_assemble(base, t)
            else:
                out = self._fx.render(base, t)

        return out

    def _ask_ai_fill(self, now: float) -> None:
        if self._plate.room_ref is not None:
            return  # the real empty room is saved: nothing to guess
        self._plate.ai = self.ai_filler
        self._plate.request_ai_fill()
        self._ai_asked_at = (now, int(self._plate.known.sum()))

    @staticmethod
    def _small(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        return cv2.resize(mask, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)

    def _everything_of_you(self, frame: np.ndarray, mask: np.ndarray, now: float) -> np.ndarray:
        """1/4-size map of everything that's you this frame: the detected
        outline (even where the detector is unsure), your tracked hand, things
        moving right next to you, and your shadow on the wall."""
        person = (self._small(mask) > PERSON_THRESHOLD).astype(np.float32)
        you = person.copy()
        sh, sw = person.shape
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        near = cv2.dilate(person, _MOTION_REACH) > 0

        # Motion next to you: a hand or hair the detector hasn't caught up with.
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape and person.any():
            moving = (cv2.absdiff(gray, self._prev_gray) > 18) & near
            you = np.maximum(you, cv2.dilate(moving.astype(np.float32), _SMALL3))
        self._prev_gray = gray

        # Your hand, from the hand tracker.
        pts = self.gesture.hand_points(now, (sh, sw)) if (
            self.gesture is not None and hasattr(self.gesture, "hand_points")) else None
        if pts is not None:
            hand = np.zeros_like(person)
            cv2.fillConvexPoly(hand, cv2.convexHull(pts.astype(np.int32)), 1.0)
            you = np.maximum(you, cv2.dilate(hand, _SMALL3, iterations=2))

        # Your shadow: darker than the saved empty room, same colour, close by.
        if self._plate is not None and self._plate.room_ref is not None and person.any():
            room = cv2.resize(self._plate.room_now(), (sw, sh), interpolation=cv2.INTER_AREA)
            live_f, room_f = small.astype(np.float32) + 1, room.astype(np.float32) + 1
            ratio = live_f.sum(axis=2) / room_f.sum(axis=2)
            chroma = np.abs(live_f / live_f.sum(axis=2, keepdims=True)
                            - room_f / room_f.sum(axis=2, keepdims=True)).max(axis=2)
            reach = cv2.dilate(person, _SHADOW_REACH) > 0
            shadow = (reach & (ratio > 0.3) & (ratio < 0.9) & (chroma < 0.05)).astype(np.uint8)
            shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, _SMALL3)
            you = np.maximum(you, cv2.dilate(shadow.astype(np.float32), _SMALL3))
        return you

    def _hide_mask(self, extra: np.ndarray | None = None) -> np.ndarray:
        """Where to show the room instead of the camera.

        Everything that was you in the last few frames, fully hidden with a
        margin, and the soft blend placed outside it, on the wall."""
        small = np.max(np.stack(self._mask_hist), axis=0)
        if extra is not None:
            small = np.maximum(small, (self._small(extra) > PERSON_THRESHOLD).astype(np.float32))
        core = cv2.dilate(small, _HIDE_CORE)
        soft = cv2.GaussianBlur(cv2.dilate(small, _HIDE_GROW), (9, 9), 0)
        small = np.maximum(core, soft)
        h, w = self._plate.known.shape
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def _auto_save_room(self, frame: np.ndarray, mask: np.ndarray, now: float) -> None:
        """Save the empty room by itself whenever nobody's been in view ~1 s."""
        if self.room != "idle" or self.sm.away:
            self._empty_frames = 0
            return
        self._empty_frames = self._empty_frames + 1 if mask.max() < 0.5 else 0
        if (self._empty_frames >= AUTO_ROOM_FRAMES
                and now - self._auto_room_at >= AUTO_ROOM_EVERY_S):
            self._plate.capture(frame)
            self.room_saved = True
            self._auto_room_at = now

    def _update_room_capture(self, frame: np.ndarray, mask: np.ndarray, now: float) -> None:
        if self.room == "countdown" and now >= self._room_at:
            self.room = "waiting"
        if self.room != "waiting":
            return
        self._room_clear = self._room_clear + 1 if mask.max() < 0.5 else 0
        if self._room_clear >= ROOM_CLEAR_FRAMES:
            self._plate.capture(frame)
            self.room, self.room_saved = "idle", True

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ashcam-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.muter is not None:
            self.muter.close()  # never leave the mic muted after quitting

    def _loop(self) -> None:
        import traceback

        vcam = None
        next_try = 0.0
        misses = 0
        try:
            while self._running:
                now = time.monotonic()
                if vcam is None and now >= next_try:
                    try:
                        vcam = self.vcam_factory()
                        self.vcam_ok, self.vcam_error = True, None
                    except Exception:
                        self.vcam_ok, self.vcam_error = False, VCAM_HELP
                        next_try = now + self.VCAM_RETRY_S
                try:
                    frame = self.camera.read()
                    misses = misses + 1 if frame is None else 0
                    if misses > 30:
                        self.camera_error = CAMERA_HELP
                        time.sleep(0.1)
                    elif frame is not None:
                        self.camera_error = None
                    out = self.process(frame, time.monotonic())
                    if self.sink:
                        self.sink(out)
                    if vcam is not None:
                        if (out.shape[1], out.shape[0]) != OUT_SIZE:
                            out = cv2.resize(out, OUT_SIZE)
                        vcam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                    self.loop_error = None
                except Exception as e:
                    # Keep the camera alive: one bad frame must not freeze the call.
                    traceback.print_exc()
                    self.loop_error = f"AshCam hit an error and is retrying: {e}"
                    time.sleep(0.05)
        finally:
            if vcam is not None:
                vcam.close()
