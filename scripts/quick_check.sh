#!/usr/bin/env bash
# Smoke test: tiny optimiser budget, results are NOT quotable.
set -euo pipefail

cd "$(dirname "$0")/.."
pytest -q
python -m qml_compare run --quick --output-dir results/scratch
