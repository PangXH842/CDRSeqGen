#!/bin/bash
# Run the unit test suite.
#
#   ./scripts/unit_tests.sh                       # everything
#   ./scripts/unit_tests.sh test_length_policy     # one module (pattern is matched against filenames)
#
# Tests combine synthetic fixtures, stubs for external models, real tiny CPU scratch models,
# and isolated shell-runner checks. No GPU, downloaded weights, or research data is required.
# See unit_tests/README.md for coverage and limitations. PYTHONPATH is set because unit_tests/
# has no __init__.py, so discovery has to be pointed at the repository root explicitly.
set -euo pipefail

cd "$(dirname "$0")/.."

PATTERN="${1:-test_*}"
[[ "$PATTERN" == *.py ]] || PATTERN="${PATTERN}.py"

PYTHONPATH=. python -m unittest discover -s unit_tests -p "$PATTERN" "${@:2}"
