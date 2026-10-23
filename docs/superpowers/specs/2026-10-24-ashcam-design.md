# AshCam — design spec

Date: 2026-10-24
Status: approved in chat, awaiting written-spec review

## Goal

A macOS app that feeds a virtual camera usable in every call app (WhatsApp,
FaceTime, Google Meet, Zoom). On a finger snap the user disintegrates into ash
(Endgame-style) and disappears, while the room behind them stays live. A second
mode, "Go away", plays a seamless loop of the user sitting at the desk.

## Constraints

- macOS (Apple silicon), Python 3.14 from Homebrew (3.11 as fallback).
- No full Xcode and no Apple signing identity, so no native camera extension.
  Output goes to the OBS Virtual Camera (OBS 32.2.2, extension enabled) via
  `pyvirtualcam` >= 0.14.
- Everything runs locally. No network calls.
- The app never disguises itself as a physical camera and has no proctoring
  bypass. Users quit it before proctored tests.

## Success criteria

1. "OBS Virtual Camera" shows AshCam output in FaceTime, WhatsApp and Meet.
2. A finger snap (or ⌥⇧V) triggers the ash effect; the person is gone within
   ~2.5 s and the background keeps moving where it is visible.
3. A second snap reassembles the person.
4. ⌥⇧A toggles Go away; the loop has no visible jump.
5. Steady 30 fps at 1280×720 on the user's Mac.

## Architecture

One process, one frame loop, small single-purpose modules:

| Module | Responsibility | Depends on |
|---|---|---|
| `camera.py` | Open the real webcam, yield BGR frames at 1280×720@30 | OpenCV |
| `segmenter.py` | Return a soft person mask (0..1 float) per frame | MediaPipe |
| `background.py` | Running clean plate: update pixels where mask < 0.2; expose `plate` | NumPy |
| `dust.py` | Particle disintegration and reverse reassembly over a person snapshot | NumPy, OpenCV |
| `away.py` | Record N seconds; serve frames ping-pong (forward then reverse) | NumPy |
| `snap.py` | Mic listener; detect sharp transients; sensitivity + double-snap | sounddevice, NumPy |
| `engine.py` | State machine + compositing; writes frames to the virtual cam | all above, pyvirtualcam |
| `ui.py` | Control panel window + global hotkeys | tkinter / pynput |
| `main.py` | Wire it together | |

### States

`PRESENT → VANISHING → VANISHED → RETURNING → PRESENT`.
`AWAY` overrides all of them while toggled on; leaving AWAY returns to the
prior state.

### Compositing per state

- PRESENT: raw frame. Background plate keeps updating.
- VANISHING: at trigger, snapshot person pixels + mask. Output = live frame
  with person area replaced by plate, plus the dust layer drawn on top.
- VANISHED: `frame * (1 - mask) + plate * mask` (feathered mask). Live room,
  no person.
- RETURNING: dust in reverse, converging onto the current person, then
  cross-fade to raw frame.
- AWAY: next frame from the ping-pong clip.

### Dust effect

- Person snapshot is split into ~6–10k particles (one per 3×3 px block inside
  the mask), each carrying its colour.
- A sweep front moves across the body (left to right with noise) over ~1.5 s;
  a particle starts moving when the front passes it.
- Moving particles drift right/up with per-particle velocity + curl-like
  noise, shrink and fade out over ~1 s.
- Pixels not yet released are drawn from the snapshot so the body crumbles
  progressively.
- Reverse: same simulation played backward onto the current person.

## Controls

- Snap or ⌥⇧V: vanish / return.
- ⌥⇧A: Go away on/off.
- Panel: live output preview, Vanish and Go away buttons, "Record away clip"
  (3 s countdown, 15 s capture), snap sensitivity slider with live meter,
  double-snap checkbox, status pill.

## Error handling

- Virtual camera unavailable: panel shows how to start OBS Virtual Camera once
  and where the Camera Extensions setting lives.
- Camera or mic permission denied: panel says which permission to grant to the
  terminal/app.
- Mic unavailable: snap detection off, hotkeys still work.
- No away clip: Go away uses the last 5 s rolling buffer of the camera.
- Plate not yet learnt for an area (person never moved): fill it with an
  inpainted guess (OpenCV `inpaint`) so there is no hole.

## Testing

- Unit tests: state machine transitions, plate updates only where unmasked,
  ping-pong index sequence, snap detector on synthetic transient vs. clap/noise.
- Dust: render to frames offline, check particle count drops to zero and
  final frame equals the plate.
- Manual: run against the real camera, check output frames, then pick
  "OBS Virtual Camera" in FaceTime.

## Out of scope (v1)

Muting the mic across all apps (needs a virtual audio device), hand-gesture
reactions, logo overlay, packaged .app bundle.
