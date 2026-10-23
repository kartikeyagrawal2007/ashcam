# AshCam

Snap your fingers on any video call and turn to dust, Endgame-style. The room
behind you stays live. Snap again to come back. Go away mode plays a loop of
you sitting at your desk.

Works in WhatsApp, FaceTime, Google Meet, Zoom and any app that lets you pick
a camera, through the OBS Virtual Camera.

## One-time setup

1. Install OBS: `brew install --cask obs`
2. Open OBS, click **Start Virtual Camera**, then **Stop**, and quit OBS.
3. Install AshCam:
   ```bash
   python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
4. First run asks for permissions. Allow your terminal app in System Settings →
   Privacy & Security under **Camera**, **Microphone** (for snapping), and
   **Accessibility** + **Input Monitoring** (for ⌥⇧V / ⌥⇧A while another app
   has focus). Then run it again.

## Use it

Double-click `AshCam.command`, or:

```bash
.venv/bin/python -m ashcam
```

In your call app, choose **OBS Virtual Camera** as the camera.

| Do this | What happens |
| --- | --- |
| Snap, or ⌥⇧V | You turn to dust and vanish. Do it again to come back. |
| ⌥⇧A | Go away: the call sees a loop of you sitting there. Again to stop. |
| Record 15 s (panel) | Records the go away clip. Sit naturally and look at the screen. |

Tips:
- For a clean vanish, click **Capture room** in the panel once and step out of
  view until it says the room is saved. Without it, the wall behind you is
  filled with a smooth guess, which can blur posters or furniture. Capture
  again if you move the camera or the lighting changes a lot.
- If snaps don't register, raise **Sensitivity** and watch the meter cross the
  gold mark. If claps or typing set it off, turn on **Require a double snap**.
- AshCam doesn't mute your mic. Mute in the call app.
- If the call's chat box has focus, ⌥⇧V and ⌥⇧A also type ◊ or Å there.
  Snapping avoids that.
- Music with a beat is ignored, but the first beat after silence can count as
  a snap. Turn on **Require a double snap** if music is playing.

Options: `--camera 1` picks another camera, `--headless` runs without the
panel, `--no-snap` turns off the mic.

## Proctored tests and interviews

Quit AshCam and OBS before any proctored test or online assessment, and pick
your real camera. AshCam does not try to hide from proctoring checks.

## Development

```bash
.venv/bin/pytest -q
```
