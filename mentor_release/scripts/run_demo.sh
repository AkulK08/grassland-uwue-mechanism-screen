#!/usr/bin/env bash
set -euo pipefail
pip install -e .
wue run-all --config configs/demo.yaml --make-demo
