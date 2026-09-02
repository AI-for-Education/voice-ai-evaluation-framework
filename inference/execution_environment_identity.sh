#!/usr/bin/env bash

# Build the Docker environment arguments used by every inference launcher.
# Image inspection is best-effort: missing immutable identity must not block a
# run, and metadata records unavailable values explicitly instead of guessing.
pipeline_execution_environment_docker_args() {
  local compose_service="$1"
  local image_reference="$2"
  local launcher="$3"
  local image_id=""
  local repo_digests_json=""

  PIPELINE_EXECUTION_ENVIRONMENT_DOCKER_ARGS=(
    --env "PIPELINE_LAUNCH_ORCHESTRATOR=docker_compose"
    --env "PIPELINE_LAUNCHER=$launcher"
    --env "PIPELINE_COMPOSE_SERVICE=$compose_service"
    --env "PIPELINE_IMAGE_REFERENCE=$image_reference"
  )

  if command -v docker >/dev/null 2>&1; then
    image_id="$(docker image inspect --format '{{.Id}}' "$image_reference" 2>/dev/null || true)"
    repo_digests_json="$(docker image inspect --format '{{json .RepoDigests}}' "$image_reference" 2>/dev/null || true)"
  fi

  if [[ -n "$image_id" ]]; then
    PIPELINE_EXECUTION_ENVIRONMENT_DOCKER_ARGS+=(
      --env "PIPELINE_IMAGE_ID=$image_id"
    )
  fi
  if [[ -n "$repo_digests_json" && "$repo_digests_json" != "null" ]]; then
    PIPELINE_EXECUTION_ENVIRONMENT_DOCKER_ARGS+=(
      --env "PIPELINE_IMAGE_REPO_DIGESTS_JSON=$repo_digests_json"
    )
  fi
}
