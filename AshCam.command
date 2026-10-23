#!/bin/zsh
# Double-click to start AshCam in Terminal.
cd "$(dirname "$0")"
exec .venv/bin/python -m ashcam "$@"
