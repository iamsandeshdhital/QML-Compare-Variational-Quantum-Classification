#!/usr/bin/env bash
# Full reproduction: every study, figures and reports into results/.
# Takes roughly 7 minutes; the noise-aware training run dominates.
set -euo pipefail

cd "$(dirname "$0")/.."
python -m qml_compare run "$@"
