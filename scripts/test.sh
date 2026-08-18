#!/bin/bash
# Run the unit test suite.
#
#   ./scripts/test.sh                       # everything
#   ./scripts/test.sh test_length_policy     # one module (pattern is matched against filenames)
#
# The tests are unittest-based and stub every model, so they need no GPU, no weights, and no
# processed dataset — the whole suite is a few seconds. PYTHONPATH is set here because test/ has no
# __init__.py, so discovery has to be pointed at the repository root explicitly.
set -euo pipefail

cd "$(dirname "$0")/.."

PATTERN="${1:-test_*}"
[[ "$PATTERN" == *.py ]] || PATTERN="${PATTERN}.py"

PYTHONPATH=. python -m unittest discover -s test -p "$PATTERN" "${@:2}"
