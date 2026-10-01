#!/usr/bin/env bash
# Runs the project's test suite. This is the one command to run before every
# commit (the pre-commit hook in .githooks/ also runs this automatically once
# enabled — see README "Testing" section).
#
# Usage:
#   ./scripts/run_tests.sh              # fast suite (mocked, no network/ffmpeg) — a few seconds
#   ./scripts/run_tests.sh --integration  # also run the slower real-ffmpeg checks
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--integration" ]]; then
    python3 -m pytest -m integration
else
    python3 -m pytest
fi
