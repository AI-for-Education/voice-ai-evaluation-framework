#!/usr/bin/env bash
set -euo pipefail

export MSYS_NO_PATHCONV="${MSYS_NO_PATHCONV:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec docker compose run --rm \
  --entrypoint "" \
  omnilingual-prepare \
  python3 -m inference.omnilingual.prepare "$@"
