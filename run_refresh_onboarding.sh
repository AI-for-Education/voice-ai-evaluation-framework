#!/usr/bin/env bash
set -euo pipefail

# Prevent Git Bash/MSYS from rewriting Linux container paths.
export MSYS_NO_PATHCONV="${MSYS_NO_PATHCONV:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EVALUATIONS_ROOT="${1:-input_output_data/output/evaluations}"
LEADERBOARDS_ROOT="${2:-input_output_data/output/leaderboards}"
# Local-development package: docs/* is ignored except docs/vocabulary.md.
PACKAGE_ROOT="${3:-docs/shared-onboarding}"

./run_leaderboard.sh "$EVALUATIONS_ROOT" "$LEADERBOARDS_ROOT"

SNAPSHOT_PYTHON="${ONBOARDING_PYTHON:-}"
if [[ -z "$SNAPSHOT_PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys' >/dev/null 2>&1; then
    SNAPSHOT_PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1 && python -c 'import sys' >/dev/null 2>&1; then
    SNAPSHOT_PYTHON="$(command -v python)"
  elif command -v uv >/dev/null 2>&1; then
    SNAPSHOT_PYTHON="$(uv python find)"
  else
    echo "No host Python was found. Install Python or uv, or set ONBOARDING_PYTHON." >&2
    exit 1
  fi
fi

"$SNAPSHOT_PYTHON" tools/build_onboarding_leaderboard_snapshot.py \
  --leaderboards-root "$LEADERBOARDS_ROOT" \
  --package-root "$PACKAGE_ROOT"
