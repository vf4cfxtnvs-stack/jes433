#!/usr/bin/env bash

set -euo pipefail

mkdir -p models

curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx?download=true" \
  -o models/de_DE-thorsten-medium.onnx

curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json?download=true" \
  -o models/de_DE-thorsten-medium.onnx.json
