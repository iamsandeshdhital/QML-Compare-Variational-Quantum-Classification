# Full reproduction: every study, figures and reports into results\.
# Takes roughly 7 minutes; the noise-aware training run dominates.
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")
python -m qml_compare run @args
