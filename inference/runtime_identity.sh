#!/usr/bin/env bash

# Deprecated compatibility shim. New launchers source
# execution_environment_identity.sh directly.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/execution_environment_identity.sh"

pipeline_runtime_docker_args() {
  pipeline_execution_environment_docker_args "$@"
  PIPELINE_RUNTIME_DOCKER_ARGS=(
    "${PIPELINE_EXECUTION_ENVIRONMENT_DOCKER_ARGS[@]}"
  )
}
