#!/usr/bin/env bash

set -euo pipefail

mkdir -p models

curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx?download=true" \
  -o models/de_DE-thorsten-high.onnx

curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json?download=true" \
  -o models/de_DE-thorsten-high.onnx.json
