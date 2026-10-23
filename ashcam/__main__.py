"""Run AshCam: `python -m ashcam` (panel) or `python -m ashcam --headless`."""

import argparse
import os
import sys
import time

import cv2

from .away import AwayLoop
from .camera import Camera
from .engine import Engine
from .hotkeys import Hotkeys
from .segmenter import Segmenter
from .snap import SnapDetector
from .state import StateMachine


def parse(argv):
    p = argparse.ArgumentParser(prog="ashcam", description="Snap to vanish on any video call.")
    p.add_argument("--camera", default="0",
                   help="camera index (default 0, the built-in camera) or a video file to loop")
    p.add_argument("--headless", action="store_true", help="no control panel")
    p.add_argument("--no-snap", action="store_true", help="don't listen to the mic")
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
        print("Couldn't open the camera. Allow your terminal app in System Settings → "
              "Privacy & Security → Camera.", file=sys.stderr)

    dumped = {"last": -1.0}
    started = time.monotonic()

    def dump(frame):
        t = time.monotonic() - started
        if args.dump and t - dumped["last"] >= 1.0:
            dumped["last"] = t
            cv2.imwrite(os.path.join(args.dump, f"frame_{int(t):03d}.jpg"), frame)

    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
    away = AwayLoop()
    engine = Engine(camera, Segmenter(), away, StateMachine(), sink=dump if args.dump else None)

    snap = None
    if not args.no_snap:
        snap = SnapDetector(on_snap=engine.toggle_vanish)
        try:
            snap.start()
            engine.snap = snap
        except Exception as e:  # no mic or permission denied: hotkeys still work
            print(f"Snap detection off ({e}). Use ⌥⇧V instead.", file=sys.stderr)
            snap = None

    hotkeys = Hotkeys(on_vanish=engine.toggle_vanish, on_away=engine.toggle_away)
    if not hotkeys.start():
        print("For ⌥⇧V / ⌥⇧A to work in other apps, allow your terminal app in "
              "System Settings → Privacy & Security → Accessibility and Input Monitoring.",
              file=sys.stderr)

    engine.start()
    print("AshCam running. In your call app, pick “OBS Virtual Camera”. "
          "Snap or ⌥⇧V to vanish, ⌥⇧A for go away.")
    try:
        if args.headless:
            run_headless(engine, args, started)
        else:
            from .ui import run_panel
            run_panel(engine, snap, away, seconds=args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        hotkeys.stop()
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
