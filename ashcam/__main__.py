"""Run AshCam: `python -m ashcam` (panel) or `python -m ashcam --headless`."""

import argparse
import os
import sys
import time

import cv2

from .away import AwayLoop
from .camera import Camera
from .engine import Engine
from .fill import BackgroundFiller
from .gesture import GestureWorker, HandTracker, SnapFusion
from .hotkeys import Hotkeys
from .micmute import MicMuter
from .segmenter import Segmenter
from .snap import SnapDetector
from .state import StateMachine


def parse(argv):
    p = argparse.ArgumentParser(prog="ashcam", description="Snap to vanish on any video call.")
    p.add_argument("--camera", default="0",
                   help="camera index (default 0, the built-in camera) or a video file to loop")
    p.add_argument("--headless", action="store_true", help="no control panel")
    p.add_argument("--no-snap", action="store_true", help="don't listen to the mic")
    p.add_argument("--no-hands", action="store_true",
                   help="don't watch for the snap on camera (mic alone triggers it)")
    p.add_argument("--no-mute", action="store_true", help="don't mute the mic while vanished")
    p.add_argument("--no-ai-fill", action="store_true",
                   help="fill the unseen wall behind you with the quick classic fill only")
    p.add_argument("--seconds", type=float, help="quit after this many seconds")
    p.add_argument("--dump", metavar="DIR", help="save one output frame per second here")
    p.add_argument("--auto", metavar="T", type=float, nargs="*", default=[],
                   help="toggle vanish at these times (seconds), for testing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(sys.argv[1:] if argv is None else argv)
    source = int(args.camera) if args.camera.isdigit() else args.camera
    camera = Camera(source)
    if not camera.ok:
        print("Waiting for camera access. Click Allow, or turn on your terminal app in "
              "System Settings → Privacy & Security → Camera. AshCam picks it up "
              "automatically.", file=sys.stderr)

    dumped = {"last": -1.0}
    started = time.monotonic()

    def dump(frame):
        t = time.monotonic() - started
        if args.dump and t - dumped["last"] >= 1.0:
            dumped["last"] = t
            cv2.imwrite(os.path.join(args.dump, f"frame_{int(t):03d}.jpg"), frame)

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
    try:
        segmenter = Segmenter()
    except Exception as e:
        print(f"Couldn't load the person-detection model ({e}). The first run needs an "
              "internet connection to download it (about 250 KB).", file=sys.stderr)
        return 1
    away = AwayLoop()
    engine = Engine(camera, segmenter, away, StateMachine(), sink=dump if args.dump else None)

    # Camera sees the snap, mic confirms the click; either alone falls back.
    fusion = SnapFusion(on_snap=engine.toggle_vanish)
    engine.fusion = fusion

    snap = None
    if not args.no_snap:
        snap = SnapDetector(on_snap=lambda: fusion.audio(time.monotonic()))
        try:
            snap.start()
            engine.snap = snap
        except Exception as e:  # no mic or permission denied: hand + hotkeys still work
            print(f"Mic snap detection off ({e}).", file=sys.stderr)
            snap = None

    gesture = None
    if not args.no_hands:
        try:
            gesture = GestureWorker(HandTracker(), fusion)
            gesture.start()
            engine.gesture = gesture
        except Exception as e:
            print(f"Hand tracking off ({e}); the mic alone will trigger snaps.", file=sys.stderr)
            gesture = None
    fusion.hands_ok = gesture is not None
    fusion.mic_live = snap is not None
    if gesture is None and snap is None:
        print("No snap detection: use ⌥⇧V.", file=sys.stderr)

    if not args.no_ai_fill:
        # LaMa fills the part of the room AshCam never saw. ~200 MB, downloaded
        # once; loads in the background so startup stays instant.
        engine.ai_filler = BackgroundFiller()
        engine.ai_filler.preload()

    muter = MicMuter()
    muter.restore_leftover()   # a previous run that died while muted
    muter.enabled = not args.no_mute
    engine.muter = muter

    hotkeys = Hotkeys(on_vanish=engine.toggle_vanish, on_away=engine.toggle_away)
    if not hotkeys.start():
        print("For ⌥⇧V / ⌥⇧A to work in other apps, allow your terminal app in "
              "System Settings → Privacy & Security → Accessibility and Input Monitoring.",
              file=sys.stderr)

    engine.start()
    print("AshCam running. In your call app, pick “OBS Virtual Camera”. "
          "Raise your hand and snap (or ⌥⇧V) to vanish, ⌥⇧A for go away.")
    try:
        if args.headless:
            run_headless(engine, args, started)
        else:
            from .ui import run_panel
            run_panel(engine, snap, away, seconds=args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()          # also puts the mic back if it was muted
        hotkeys.stop()
        if gesture:
            gesture.stop()
        if snap:
            snap.stop()
        camera.close()
    return 0


def run_headless(engine: Engine, args, started: float) -> None:
    auto = sorted(args.auto)
    last = None
    while args.seconds is None or time.monotonic() - started < args.seconds:
        t = time.monotonic() - started
        while auto and t >= auto[0]:
            auto.pop(0)
            engine.toggle_vanish()
        status = engine.status()
        line = (status["state"], status["away"], status["error"])
        if line != last:
            last = line
            away = " +away" if status["away"] else ""
            print(f"[{t:5.1f}s] {status['state']}{away}"
                  + (f"  ! {status['error']}" if status["error"] else ""), flush=True)
        time.sleep(0.05)


if __name__ == "__main__":
    sys.exit(main())
