#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible alias for the explicit NeMo launcher.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_nemo_inference.sh" "$@"
