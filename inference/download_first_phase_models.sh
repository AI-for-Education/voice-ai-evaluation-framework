#!/usr/bin/env bash

# Download the locally deployable ASR candidates selected in:
#   ASR_benchmark_candidate_models_with_runtime_guidance.xlsx
#
# The workbook contains repeated model families and a few rows without an exact
# checkpoint. This script downloads each exact, selected Hugging Face package
# with `uvx hf download` and reports unresolved choices without guessing.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMERS_ROOT="${SCRIPT_DIR}/transformers/models"
FRAMEWORK_SPEC_MODELS_ROOT="${SCRIPT_DIR}/framework_spec/models"
MULTIMODAL_MODELS_ROOT="${SCRIPT_DIR}/multimodal/models"

# Immutable revisions verified against the exact repositories selected by the
# workbook and the tracked inference profiles. A full snapshot means every
# file in the repository at this revision (not every revision in Git history).
# MMS is the documented exception: its repository provides language adapters,
# so the script downloads the complete Swahili (`swh`) package and shared base.
REV_WAV2VEC2_BASE_960H="22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
REV_BOOKBOT_ORTHOGRAPHIC="f71831f7a9ad8d0ab1fa16ff071985472a191007"
REV_BOOKBOT_PHONEME="b83c1c4f1da4eef4a3a7b8c8c32d7d1a91bc0011"
REV_W2V_BERT_SWAHILI="10e85418ae5978a084c06de2c47448acb1a4e0c8"
REV_HUBERT_LARGE="ece5fabbf034c1073acae96d5401b25be96709d8"
REV_PAZA="8a0bd24795b298b2f3678fcd8129c0a1c2b4ee22"
REV_WHISPER_LARGE="4ef9b41f0d4fe232daafdb5f76bb1dd8b23e01d7"
REV_WHISPER_LARGE_V2="ae4642769ce2ad8fc292556ccea8e901f1530655"
REV_MMS_1B_ALL="3d33597edbdaaba14a8e858e2c8caa76e3cec0cd"
REV_PHI4_MULTIMODAL="1e78f4f84fd8a92f132295a32e5cb71b5321dcab"
REV_GEMMA4_OMNI="3e22461f65e89153144f8adb70e3b8c2cc9845a7"
REV_QWEN25_OMNI="ae9e1690543ffd5c0221dc27f79834d0294cba00"
REV_KALDI="e02e35f0254bb033fab73d1df99fc34123e31d56"

MODE="download"
FAILED_JOBS=()
ACTIVE_JOBS=()
COMPLETE_JOBS=()
DOWNLOADED_JOBS=()
RETRIED_JOBS=()
PROCESS_INSPECTION_WARNED=0

usage() {
  cat <<'EOF'
Usage:
  bash inference/download_first_phase_models.sh [--list]

Options:
  --list      Show configured jobs plus live active/complete/partial status.
  --help      Show this help text.

Run the script again after an interruption. Hugging Face reconciles each pinned
package and avoids downloading unchanged completed files. Packages are full
repository snapshots unless that repository documents a separately loadable
benchmark-language adapter (MMS Swahili). Byte-level resume behavior is
controlled by the installed Hugging Face client and is not assumed by this
script. Completed jobs receive source, size, and completion manifests beside
their destination and are skipped on later runs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      MODE="list"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

print_unresolved_choices() {
  cat <<'EOF'

[UNRESOLVED - not downloaded]
  AI4Bharat IndicConformer
    The workbook points to a collection, not one exact checkpoint repository.

  CoLA
    The workbook provides a paper/reference, not an ASR checkpoint.

The generic Whisper choice is represented by both selected Transformers
checkpoints: openai/whisper-large and openai/whisper-large-v2. The local omni
choice is represented by both selected repositories: google/gemma-4-E2B-it and
Qwen/Qwen2.5-Omni-7B. Repeated wav2vec and Kaldi entries are downloaded once.
EOF
}

print_jobs() {
  cat <<EOF
[CURRENT IMAGE - ${TRANSFORMERS_ROOT}]
  facebook/wav2vec2-base-960h@${REV_WAV2VEC2_BASE_960H}
  bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm@${REV_BOOKBOT_ORTHOGRAPHIC}
  bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot@${REV_BOOKBOT_PHONEME}
  badrex/w2v-bert-2.0-swahili-asr@${REV_W2V_BERT_SWAHILI}
  facebook/hubert-large-ls960-ft@${REV_HUBERT_LARGE}
  microsoft/paza-whisper-large-v3-turbo@${REV_PAZA}
  openai/whisper-large@${REV_WHISPER_LARGE}
  openai/whisper-large-v2@${REV_WHISPER_LARGE_V2}
  facebook/mms-1b-all@${REV_MMS_1B_ALL} [language package: swh]

[SEPARATE TOOLKIT STAGING - ${FRAMEWORK_SPEC_MODELS_ROOT}]
  kaldi-asr/kaldi@${REV_KALDI}

[MULTIMODAL RUNTIMES - ${MULTIMODAL_MODELS_ROOT}]
  microsoft/paza-Phi-4-multimodal-instruct@${REV_PHI4_MULTIMODAL}
  google/gemma-4-E2B-it@${REV_GEMMA4_OMNI}
  Qwen/Qwen2.5-Omni-7B@${REV_QWEN25_OMNI}
EOF
  print_unresolved_choices
}

if command -v powershell.exe >/dev/null 2>&1; then
  PROCESS_INSPECTOR="powershell.exe"
elif command -v pwsh.exe >/dev/null 2>&1; then
  PROCESS_INSPECTOR="pwsh.exe"
elif command -v ps >/dev/null 2>&1; then
  PROCESS_INSPECTOR="ps"
else
  PROCESS_INSPECTOR="none"
fi

process_snapshot() {
  if [[ "$PROCESS_INSPECTOR" == "powershell.exe" ]] \
    || [[ "$PROCESS_INSPECTOR" == "pwsh.exe" ]]; then
    "$PROCESS_INSPECTOR" -NoProfile -NonInteractive -Command '
      $ErrorActionPreference = "SilentlyContinue"
      Get-CimInstance Win32_Process | ForEach-Object {
        $command = [string]$_.CommandLine
        $command = $command -replace "[\r\n|]", " "
        "{0}|{1}" -f $_.ProcessId, $command
      }
    ' 2>/dev/null
    return
  fi

  if [[ "$PROCESS_INSPECTOR" == "ps" ]]; then
    ps -eo pid=,args= 2>/dev/null \
      | while read -r process_id command_line; do
          printf '%s|%s\n' "$process_id" "$command_line"
        done
  fi
}

find_active_download() {
  local repository_identity="$1"
  local destination="$2"
  local identity_lower="${repository_identity,,}"
  local destination_lower="${destination,,}"
  local process_id
  local command_line
  local command_lower
  local identity_match
  local download_command
  local snapshot_rows=0

  ACTIVE_PROCESS_ID=""
  ACTIVE_PROCESS_COMMAND=""
  destination_lower="${destination_lower//\\//}"

  if [[ "$PROCESS_INSPECTOR" == "none" ]]; then
    if [[ $PROCESS_INSPECTION_WARNED -eq 0 ]]; then
      echo "[WARN] No supported process inspector was found; active-download checks are unavailable." >&2
      PROCESS_INSPECTION_WARNED=1
    fi
    return 1
  fi

  while IFS='|' read -r process_id command_line; do
    [[ -n "$process_id" ]] || continue
    [[ -n "$command_line" ]] || continue
    snapshot_rows=$((snapshot_rows + 1))

    command_lower="${command_line,,}"
    command_lower="${command_lower//\\//}"
    identity_match=0
    download_command=0

    if [[ "$command_lower" == *"$identity_lower"* ]] \
      || [[ "$command_lower" == *"$destination_lower"* ]]; then
      identity_match=1
    fi

    if [[ "$command_lower" == *"download"* ]] \
      && [[ "$command_lower" == *"uvx"* \
        || "$command_lower" == *"huggingface"* \
        || "$command_lower" == *"hf "* \
        || "$command_lower" == *"hf.exe"* ]]; then
      download_command=1
    elif [[ "$command_lower" == *"git"* ]] \
      && [[ "$command_lower" == *"clone"* ]]; then
      download_command=1
    fi

    if [[ $identity_match -eq 1 ]] && [[ $download_command -eq 1 ]]; then
      ACTIVE_PROCESS_ID="$process_id"
      ACTIVE_PROCESS_COMMAND="$command_line"
      return 0
    fi
  done < <(process_snapshot)

  if [[ $snapshot_rows -eq 0 ]] && [[ $PROCESS_INSPECTION_WARNED -eq 0 ]]; then
    echo "[WARN] Process inspection returned no readable command lines;" >&2
    echo "       active downloads in other windows may not be detectable." >&2
    PROCESS_INSPECTION_WARNED=1
  fi

  return 1
}

verify_download_directory() {
  local destination="$1"
  [[ -s "${destination}/config.json" ]] \
    && [[ -n "$(find "$destination" -type f \
      \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) \
      -size +1024c -print -quit 2>/dev/null)" ]]
}

verify_download() {
  local destination="$1"
  verify_download_directory "$destination"
}

write_size_manifest() {
  local destination="$1"
  local manifest_path="$2"
  local temporary_manifest="${manifest_path}.tmp.$$"
  local file_path
  local relative_path
  local file_size

  : > "$temporary_manifest"
  while IFS= read -r -d '' file_path; do
    relative_path="${file_path#"${destination}/"}"
    file_size="$(wc -c < "$file_path" | tr -d '[:space:]')"
    printf '%s\t%s\n' "$file_size" "$relative_path" >> "$temporary_manifest"
  done < <(
    find "$destination" -type f \
      -not -path "${destination}/.cache/huggingface/*" \
      -not -path "${destination}/.git/*" \
      -print0
  )

  if [[ ! -s "$temporary_manifest" ]]; then
    echo "[ERROR] No downloaded model files were found in ${destination}." >&2
    return 1
  fi

  LC_ALL=C sort -k2,2 "$temporary_manifest" -o "$temporary_manifest"
  mv "$temporary_manifest" "$manifest_path"
}

verify_size_manifest() {
  local destination="$1"
  local manifest_path="$2"
  local expected_size
  local relative_path
  local actual_size
  local checked_files=0

  [[ -s "$manifest_path" ]] || return 1

  while IFS=$'\t' read -r expected_size relative_path; do
    [[ -n "$expected_size" ]] || return 1
    [[ -n "$relative_path" ]] || return 1
    [[ -f "${destination}/${relative_path}" ]] || return 1

    actual_size="$(wc -c < "${destination}/${relative_path}" | tr -d '[:space:]')"
    [[ "$actual_size" == "$expected_size" ]] || return 1
    checked_files=$((checked_files + 1))
  done < "$manifest_path"

  [[ $checked_files -gt 0 ]]
}

write_source_manifest() {
  local repository_id="$1"
  local revision="$2"
  local scope="$3"
  local manifest_path="$4"
  local temporary_manifest="${manifest_path}.tmp.$$"

  {
    printf 'repository_id\t%s\n' "$repository_id"
    printf 'revision\t%s\n' "$revision"
    printf 'scope\t%s\n' "$scope"
  } > "$temporary_manifest"
  mv "$temporary_manifest" "$manifest_path"
}

verify_source_manifest() {
  local repository_id="$1"
  local revision="$2"
  local scope="$3"
  local manifest_path="$4"
  local key
  local value
  local recorded_repository=""
  local recorded_revision=""
  local recorded_scope=""

  [[ -s "$manifest_path" ]] || return 1
  while IFS=$'\t' read -r key value; do
    case "$key" in
      repository_id) recorded_repository="$value" ;;
      revision) recorded_revision="$value" ;;
      scope) recorded_scope="$value" ;;
    esac
  done < "$manifest_path"

  [[ "$recorded_repository" == "$repository_id" ]] \
    && [[ "$recorded_revision" == "$revision" ]] \
    && [[ "$recorded_scope" == "$scope" ]]
}

verify_required_files() {
  local destination="$1"
  shift
  local relative_path

  for relative_path in "$@"; do
    [[ -s "${destination}/${relative_path}" ]] || return 1
  done
}

verify_completed_state() {
  local destination="$1"
  local repository_id="$2"
  local revision="$3"
  local scope="$4"
  local completion_marker="$5"
  local size_manifest="$6"
  local source_manifest="$7"

  [[ -f "$completion_marker" ]] \
    && verify_download "$destination" \
    && verify_size_manifest "$destination" "$size_manifest" \
    && verify_source_manifest \
      "$repository_id" "$revision" "$scope" "$source_manifest"
}

verify_hf_local_directory() {
  local repository_id="$1"
  local revision="$2"
  local destination="$3"
  local scope="$4"
  shift 4
  local metadata_root="${destination}/.cache/huggingface/download"
  local metadata_path
  local metadata_revision
  local relative_path
  local downloaded_path
  local checked_metadata=0

  verify_download "$destination" || return 1
  [[ -d "$metadata_root" ]] || return 1

  # A completed local-directory download records one .metadata file beside
  # each downloaded file. Confirm that every record still has its real file
  # and belongs to the exact pinned revision. Stale .incomplete files are not
  # treated as missing when the authoritative repository verification passes.
  while IFS= read -r -d '' metadata_path; do
    relative_path="${metadata_path#"${metadata_root}/"}"
    relative_path="${relative_path%.metadata}"
    downloaded_path="${destination}/${relative_path}"
    [[ -f "$downloaded_path" ]] || return 1
    IFS= read -r metadata_revision < "$metadata_path" || return 1
    metadata_revision="${metadata_revision%$'\r'}"
    [[ "$metadata_revision" == "$revision" ]] || return 1
    checked_metadata=$((checked_metadata + 1))
  done < <(find "$metadata_root" -type f -name '*.metadata' -print0 2>/dev/null)

  [[ $checked_metadata -gt 0 ]] || return 1
  if [[ "$scope" == "full" ]]; then
    verify_hf_snapshot_with_cli "$repository_id" "$revision" "$destination"
  else
    verify_required_files "$destination" "$@" \
      && verify_hf_language_package_with_cli \
        "$repository_id" "$revision" "$destination"
  fi
}

verify_hf_git_clone() {
  local repository_id="$1"
  local revision="$2"
  local destination="$3"
  local expected_remote="https://huggingface.co/${repository_id}"
  local actual_remote
  local local_revision
  local lfs_files

  command -v git >/dev/null 2>&1 || return 1
  [[ -d "${destination}/.git" ]] || return 1
  actual_remote="$(git -c "safe.directory=${destination}" -C "$destination" \
    remote get-url origin 2>/dev/null)" || return 1
  actual_remote="${actual_remote%.git}"
  actual_remote="${actual_remote%/}"
  [[ "$actual_remote" == "$expected_remote" ]] || return 1

  local_revision="$(git -c "safe.directory=${destination}" -C "$destination" \
    rev-parse --verify HEAD 2>/dev/null)" || return 1
  [[ "$local_revision" == "$revision" ]] || return 1
  verify_download "$destination" || return 1

  # Missing tracked files mean the checkout is not a complete clone.
  [[ -z "$(git -c "safe.directory=${destination}" -C "$destination" \
    ls-files --deleted 2>/dev/null)" ]] || return 1

  # In `git lfs ls-files -l`, `*` means the real LFS object is present and
  # `-` means that only the small pointer file is present.
  if lfs_files="$(git -c "safe.directory=${destination}" -C "$destination" \
    lfs ls-files -l 2>/dev/null)" && [[ -n "$lfs_files" ]]; then
    if printf '%s\n' "$lfs_files" | grep -Eq '^[[:xdigit:]]+[[:space:]]+-[[:space:]]+'; then
      return 1
    fi
  fi

  git -c "safe.directory=${destination}" -C "$destination" \
    fsck --no-dangling >/dev/null 2>&1 || return 1
  return 0
}

verify_hf_snapshot_with_cli() {
  local repository_id="$1"
  local revision="$2"
  local destination="$3"

  HF_VERIFY_DETAILS=""
  if ! command -v uvx >/dev/null 2>&1; then
    HF_VERIFY_DETAILS="uvx is unavailable, so the full repository snapshot cannot be verified."
    return 1
  fi

  if ! HF_VERIFY_DETAILS="$(
    PYTHONIOENCODING=utf-8 uvx hf cache verify "$repository_id" \
      --revision "$revision" \
      --local-dir "$destination" \
      --fail-on-missing-files 2>&1
  )"; then
    return 1
  fi

  return 0
}

verify_hf_language_package_with_cli() {
  local repository_id="$1"
  local revision="$2"
  local destination="$3"

  HF_VERIFY_DETAILS=""
  if ! command -v uvx >/dev/null 2>&1; then
    HF_VERIFY_DETAILS="uvx is unavailable, so the language package cannot be verified."
    return 1
  fi

  # Missing files outside the selected language package are intentional. The
  # explicit required-file check above establishes completeness for Swahili;
  # this command verifies all present files against the pinned Hub revision.
  if ! HF_VERIFY_DETAILS="$(
    PYTHONIOENCODING=utf-8 uvx hf cache verify "$repository_id" \
      --revision "$revision" \
      --local-dir "$destination" 2>&1
  )"; then
    return 1
  fi

  return 0
}

report_hf_verification_failure() {
  if [[ -n "${HF_VERIFY_DETAILS:-}" ]]; then
    while IFS= read -r detail_line; do
      echo "                       ${detail_line}"
    done <<< "$HF_VERIFY_DETAILS"
  fi

  if [[ "${HF_VERIFY_DETAILS:-}" == *"vocabs/con.txt"* ]]; then
    echo "                       Windows reserves the name CON, so this exact"
    echo "                       repository path requires a Linux filesystem."
  fi
}

verify_external_completed_state() {
  local repository_id="$1"
  local revision="$2"
  local destination="$3"
  local scope="$4"
  shift 4

  EXTERNAL_COMPLETION_SOURCE=""
  HF_VERIFY_DETAILS=""
  if verify_hf_local_directory \
    "$repository_id" "$revision" "$destination" "$scope" "$@"; then
    EXTERNAL_COMPLETION_SOURCE="pinned Hugging Face ${scope} verification"
    return 0
  fi

  if verify_hf_git_clone "$repository_id" "$revision" "$destination"; then
    EXTERNAL_COMPLETION_SOURCE="exact Git/LFS repository checkout"
    return 0
  fi

  return 1
}

locate_hf_global_snapshot() {
  local repository_id="$1"
  local revision="$2"
  local cache_root
  local converted_root

  if [[ -n "${HF_HUB_CACHE:-}" ]]; then
    cache_root="$HF_HUB_CACHE"
  elif [[ -n "${HF_HOME:-}" ]]; then
    cache_root="${HF_HOME}/hub"
  else
    cache_root="${HOME}/.cache/huggingface/hub"
  fi

  if command -v cygpath >/dev/null 2>&1; then
    if converted_root="$(cygpath -u "$cache_root" 2>/dev/null)"; then
      cache_root="$converted_root"
    fi
  fi

  HF_GLOBAL_SNAPSHOT="${cache_root}/models--${repository_id//\//--}/snapshots/${revision}"
  [[ -d "$HF_GLOBAL_SNAPSHOT" ]]
}

verify_hf_global_cache() {
  local repository_id="$1"
  local revision="$2"

  locate_hf_global_snapshot "$repository_id" "$revision" || return 1
  if ! command -v uvx >/dev/null 2>&1; then
    return 1
  fi

  if ! PYTHONIOENCODING=utf-8 uvx hf cache verify "$repository_id" \
      --revision "$revision" \
      --fail-on-missing-files >/dev/null 2>&1; then
    return 1
  fi

  return 0
}

LIST_ACTIVE_COUNT=0
LIST_COMPLETE_COUNT=0
LIST_CACHED_COMPLETE_COUNT=0
LIST_PARTIAL_COUNT=0
LIST_NOT_PRESENT_COUNT=0
LIST_MODEL_NAMES=()
LIST_MODEL_STATUSES=()
LIST_MODEL_BYTES=()
LIST_TOTAL_BYTES=0
GLOBAL_ACTIVE_REPOSITORIES=()
GLOBAL_ACTIVE_PROCESS_IDS=()
GLOBAL_ACTIVE_COMMANDS=()
GLOBAL_ACTIVE_COUNT=0
GLOBAL_CONFIGURED_ACTIVE_COUNT=0
GLOBAL_OTHER_ACTIVE_COUNT=0

is_configured_hf_repository() {
  local repository_id="${1,,}"

  case "$repository_id" in
    facebook/wav2vec2-base-960h \
      |bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm \
      |bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot \
      |badrex/w2v-bert-2.0-swahili-asr \
      |facebook/hubert-large-ls960-ft \
      |microsoft/paza-whisper-large-v3-turbo \
      |openai/whisper-large \
      |openai/whisper-large-v2 \
      |facebook/mms-1b-all \
      |microsoft/phi-4-multimodal-instruct \
      |google/gemma-4-e2b-it \
      |qwen/qwen2.5-omni-7b)
      return 0
      ;;
  esac

  return 1
}

collect_all_active_hf_downloads() {
  local process_id
  local command_line
  local command_lower
  local repository_id
  local repository_lower
  local known_repository
  local index

  GLOBAL_ACTIVE_REPOSITORIES=()
  GLOBAL_ACTIVE_PROCESS_IDS=()
  GLOBAL_ACTIVE_COMMANDS=()
  GLOBAL_ACTIVE_COUNT=0
  GLOBAL_CONFIGURED_ACTIVE_COUNT=0
  GLOBAL_OTHER_ACTIVE_COUNT=0

  while IFS='|' read -r process_id command_line; do
    [[ -n "$process_id" ]] || continue
    [[ -n "$command_line" ]] || continue
    command_lower="${command_line,,}"

    if [[ "$command_lower" != *"uvx"* \
      && "$command_lower" != *"hf.exe"* \
      && "$command_lower" != *"huggingface"* ]]; then
      continue
    fi

    if [[ "$command_line" \
      =~ [[:space:]]download[[:space:]]+([[:alnum:]_.-]+/[[:alnum:]_.-]+) ]]; then
      repository_id="${BASH_REMATCH[1]}"
    else
      continue
    fi

    repository_lower="${repository_id,,}"
    known_repository=0
    for index in "${!GLOBAL_ACTIVE_REPOSITORIES[@]}"; do
      if [[ "${GLOBAL_ACTIVE_REPOSITORIES[$index],,}" == "$repository_lower" ]]; then
        known_repository=1
        break
      fi
    done
    [[ $known_repository -eq 0 ]] || continue

    GLOBAL_ACTIVE_REPOSITORIES+=("$repository_id")
    GLOBAL_ACTIVE_PROCESS_IDS+=("$process_id")
    GLOBAL_ACTIVE_COMMANDS+=("$command_line")
    if is_configured_hf_repository "$repository_id"; then
      GLOBAL_CONFIGURED_ACTIVE_COUNT=$((GLOBAL_CONFIGURED_ACTIVE_COUNT + 1))
    else
      GLOBAL_OTHER_ACTIVE_COUNT=$((GLOBAL_OTHER_ACTIVE_COUNT + 1))
    fi
  done < <(process_snapshot)

  GLOBAL_ACTIVE_COUNT=${#GLOBAL_ACTIVE_REPOSITORIES[@]}
}

print_all_active_hf_downloads() {
  local index
  local category

  collect_all_active_hf_downloads
  echo
  echo "[ALL ACTIVE HUGGING FACE DOWNLOADS]"
  if [[ $GLOBAL_ACTIVE_COUNT -eq 0 ]]; then
    echo "  None detected."
    return
  fi

  for index in "${!GLOBAL_ACTIVE_REPOSITORIES[@]}"; do
    if is_configured_hf_repository "${GLOBAL_ACTIVE_REPOSITORIES[$index]}"; then
      category="CONFIGURED"
    else
      category="OTHER"
    fi
    echo "  [${category}] ${GLOBAL_ACTIVE_REPOSITORIES[$index]}"
    echo "               representative PID ${GLOBAL_ACTIVE_PROCESS_IDS[$index]}"
  done
}

directory_size_bytes() {
  local destination="$1"
  local bytes

  if [[ ! -d "$destination" ]]; then
    printf '0\n'
    return
  fi

  # Count model/source artifacts, but not Git objects or Hugging Face's small
  # local-directory bookkeeping files. This also avoids double-counting LFS.
  bytes="$(find "$destination" -type f \
    -not -path "${destination}/.git/*" \
    -not -path "${destination}/.cache/huggingface/*" \
    -printf '%s\n' 2>/dev/null \
    | awk '{ total += $1 } END { printf "%.0f", total }')"

  if [[ ! "$bytes" =~ ^[0-9]+$ ]]; then
    bytes=0
  fi
  printf '%s\n' "$bytes"
}

format_bytes() {
  local bytes="$1"

  awk -v bytes="$bytes" 'BEGIN {
    split("B KiB MiB GiB TiB", units, " ")
    value = bytes + 0
    unit_index = 1
    while (value >= 1024 && unit_index < 5) {
      value /= 1024
      unit_index++
    }
    if (unit_index == 1) {
      printf "%.0f %s", value, units[unit_index]
    } else {
      printf "%.2f %s", value, units[unit_index]
    }
  }'
}

record_model_size() {
  local job_name="$1"
  local status="$2"
  local destination="$3"
  local bytes

  bytes="$(directory_size_bytes "$destination")"
  LIST_MODEL_NAMES+=("$job_name")
  LIST_MODEL_STATUSES+=("$status")
  LIST_MODEL_BYTES+=("$bytes")
  LIST_TOTAL_BYTES=$((LIST_TOTAL_BYTES + bytes))
}

print_model_size_summary() {
  local index

  echo "  Local artifact sizes (excludes .git and Hugging Face metadata):"
  for index in "${!LIST_MODEL_NAMES[@]}"; do
    printf '    %-58s %10s  [%s]\n' \
      "${LIST_MODEL_NAMES[$index]}" \
      "$(format_bytes "${LIST_MODEL_BYTES[$index]}")" \
      "${LIST_MODEL_STATUSES[$index]}"
  done
  printf '  Total local model/tool data: %s\n' "$(format_bytes "$LIST_TOTAL_BYTES")"
}

report_hf_status() {
  local job_name="$1"
  local repository_id="$2"
  local revision="$3"
  local destination_root="$4"
  local destination_name="$5"
  local scope="$6"
  shift 6
  local destination="${destination_root}/${destination_name}"
  local completion_marker="${destination_root}/.${destination_name}.download-complete"
  local size_manifest="${destination_root}/.${destination_name}.download-sizes.tsv"
  local source_manifest="${destination_root}/.${destination_name}.download-source.tsv"
  local status

  if find_active_download "$repository_id" "$destination"; then
    echo "  [ACTIVE] ${job_name}"
    echo "           PID ${ACTIVE_PROCESS_ID}: ${ACTIVE_PROCESS_COMMAND}"
    LIST_ACTIVE_COUNT=$((LIST_ACTIVE_COUNT + 1))
    status="ACTIVE"
  elif verify_completed_state \
    "$destination" "$repository_id" "$revision" "$scope" \
    "$completion_marker" "$size_manifest" "$source_manifest"; then
    echo "  [COMPLETE] ${job_name}"
    echo "             ${repository_id}@${revision}"
    LIST_COMPLETE_COUNT=$((LIST_COMPLETE_COUNT + 1))
    status="COMPLETE"
  elif verify_external_completed_state \
    "$repository_id" "$revision" "$destination" "$scope" "$@"; then
    echo "  [COMPLETE] ${job_name}"
    echo "             ${repository_id}@${revision}"
    echo "             Verified from ${EXTERNAL_COMPLETION_SOURCE}."
    LIST_COMPLETE_COUNT=$((LIST_COMPLETE_COUNT + 1))
    status="COMPLETE"
  elif [[ -e "$destination" ]]; then
    echo "  [PARTIAL/UNVERIFIED] ${job_name}"
    echo "                       ${destination}"
    report_hf_verification_failure
    LIST_PARTIAL_COUNT=$((LIST_PARTIAL_COUNT + 1))
    status="PARTIAL/UNVERIFIED"
  elif verify_hf_global_cache "$repository_id" "$revision"; then
    echo "  [CACHED COMPLETE - NOT STAGED] ${job_name}"
    echo "                                 ${repository_id}@${revision}"
    echo "                                 Cache: ${HF_GLOBAL_SNAPSHOT}"
    echo "                                 Expected: ${destination}"
    LIST_CACHED_COMPLETE_COUNT=$((LIST_CACHED_COMPLETE_COUNT + 1))
    status="CACHED COMPLETE / NOT STAGED"
  elif locate_hf_global_snapshot "$repository_id" "$revision"; then
    echo "  [PARTIAL/UNVERIFIED IN HF CACHE] ${job_name}"
    echo "                                   ${HF_GLOBAL_SNAPSHOT}"
    LIST_PARTIAL_COUNT=$((LIST_PARTIAL_COUNT + 1))
    status="PARTIAL/UNVERIFIED CACHE"
  else
    echo "  [NOT PRESENT] ${job_name}"
    LIST_NOT_PRESENT_COUNT=$((LIST_NOT_PRESENT_COUNT + 1))
    status="NOT PRESENT"
  fi

  record_model_size "$job_name" "$status" "$destination"
}

report_git_status() {
  local job_name="$1"
  local repository_identity="$2"
  local revision="$3"
  local destination_root="$4"
  local destination_name="$5"
  local destination="${destination_root}/${destination_name}"
  local repository_url="https://github.com/${repository_identity}"
  local status

  if find_active_download "$repository_identity" "$destination"; then
    echo "  [ACTIVE] ${job_name}"
    echo "           PID ${ACTIVE_PROCESS_ID}: ${ACTIVE_PROCESS_COMMAND}"
    LIST_ACTIVE_COUNT=$((LIST_ACTIVE_COUNT + 1))
    status="ACTIVE"
  elif verify_git_checkout "$repository_url" "$revision" "$destination"; then
    echo "  [COMPLETE] ${job_name}"
    echo "             ${repository_identity}@${revision}"
    LIST_COMPLETE_COUNT=$((LIST_COMPLETE_COUNT + 1))
    status="COMPLETE"
  elif [[ -e "$destination" ]]; then
    echo "  [PARTIAL/UNVERIFIED] ${job_name}"
    echo "                       ${destination}"
    LIST_PARTIAL_COUNT=$((LIST_PARTIAL_COUNT + 1))
    status="PARTIAL/UNVERIFIED"
  else
    echo "  [NOT PRESENT] ${job_name}"
    LIST_NOT_PRESENT_COUNT=$((LIST_NOT_PRESENT_COUNT + 1))
    status="NOT PRESENT"
  fi

  record_model_size "$job_name" "$status" "$destination"
}

verify_git_checkout() {
  local repository_url="$1"
  local revision="$2"
  local destination="$3"
  local actual_remote
  local expected_remote="${repository_url%.git}"
  local local_revision

  command -v git >/dev/null 2>&1 || return 1
  [[ -d "${destination}/.git" ]] || return 1
  actual_remote="$(git -c "safe.directory=${destination}" -C "$destination" \
    remote get-url origin 2>/dev/null)" || return 1
  actual_remote="${actual_remote%.git}"
  actual_remote="${actual_remote%/}"
  [[ "$actual_remote" == "$expected_remote" ]] || return 1

  local_revision="$(git -c "safe.directory=${destination}" -C "$destination" \
    rev-parse --verify HEAD 2>/dev/null)" || return 1
  [[ "$local_revision" == "$revision" ]] || return 1
  [[ -z "$(git -c "safe.directory=${destination}" -C "$destination" \
    ls-files --deleted 2>/dev/null)" ]] || return 1
  git -c "safe.directory=${destination}" -C "$destination" \
    fsck --no-dangling >/dev/null 2>&1
}

if [[ "$MODE" == "list" ]]; then
  print_jobs
  echo
  echo "[LIVE STATUS]"

  report_hf_status "wav2vec2-base-960h" \
    "facebook/wav2vec2-base-960h" "$REV_WAV2VEC2_BASE_960H" \
    "$TRANSFORMERS_ROOT" "wav2vec2-base-960h" "full"
  report_hf_status "BookBot orthographic CTC" \
    "bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm" \
    "$REV_BOOKBOT_ORTHOGRAPHIC" \
    "$TRANSFORMERS_ROOT" "wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm" \
    "full"
  report_hf_status "BookBot phoneme CTC" \
    "bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot" \
    "$REV_BOOKBOT_PHONEME" \
    "$TRANSFORMERS_ROOT" \
    "wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot" \
    "full"
  report_hf_status "w2v-bert-2.0-swahili-asr" \
    "badrex/w2v-bert-2.0-swahili-asr" \
    "$REV_W2V_BERT_SWAHILI" \
    "$TRANSFORMERS_ROOT" "w2v-bert-2.0-swahili-asr" "full"
  report_hf_status "hubert-large-ls960-ft" \
    "facebook/hubert-large-ls960-ft" "$REV_HUBERT_LARGE" \
    "$TRANSFORMERS_ROOT" "hubert-large-ls960-ft" "full"
  report_hf_status "paza-whisper-large-v3-turbo" \
    "microsoft/paza-whisper-large-v3-turbo" \
    "$REV_PAZA" \
    "$TRANSFORMERS_ROOT" "paza-whisper-large-v3-turbo" "full"
  report_hf_status "whisper-large" \
    "openai/whisper-large" "$REV_WHISPER_LARGE" \
    "$TRANSFORMERS_ROOT" "whisper-large" "full"
  report_hf_status "whisper-large-v2" \
    "openai/whisper-large-v2" "$REV_WHISPER_LARGE_V2" \
    "$TRANSFORMERS_ROOT" "whisper-large-v2" "full"
  report_hf_status "mms-1b-all" \
    "facebook/mms-1b-all" "$REV_MMS_1B_ALL" \
    "$TRANSFORMERS_ROOT" "mms-1b-all" "language:swh" \
    ".gitattributes" \
    "README.md" \
    "adapter.swh.bin" \
    "adapter.swh.safetensors" \
    "config.json" \
    "create_adapters.py" \
    "create_vocab.py" \
    "download_vocabs.sh" \
    "model.safetensors" \
    "preprocessor_config.json" \
    "pytorch_model.bin" \
    "special_tokens_map.json" \
    "tokenizer_config.json" \
    "vocab.json" \
    "vocabs/swh.txt"
  report_git_status "Kaldi toolkit" \
    "kaldi-asr/kaldi" "$REV_KALDI" "$FRAMEWORK_SPEC_MODELS_ROOT" "kaldi"
  report_hf_status "Paza Phi-4 multimodal" \
    "microsoft/paza-Phi-4-multimodal-instruct" \
    "$REV_PHI4_MULTIMODAL" \
    "$MULTIMODAL_MODELS_ROOT" "paza-Phi-4-multimodal-instruct" "full"
  report_hf_status "Gemma 4 E2B Omni" \
    "google/gemma-4-E2B-it" "$REV_GEMMA4_OMNI" \
    "$MULTIMODAL_MODELS_ROOT" "gemma-4-E2B-it" "full"
  report_hf_status "Qwen2.5 Omni 7B" \
    "Qwen/Qwen2.5-Omni-7B" "$REV_QWEN25_OMNI" \
    "$MULTIMODAL_MODELS_ROOT" "Qwen2.5-Omni-7B" "full"

  print_all_active_hf_downloads

  echo
  echo "[LIVE SUMMARY]"
  echo "  Process inspector: ${PROCESS_INSPECTOR}"
  echo "  Active configured jobs: ${LIST_ACTIVE_COUNT}"
  echo "  Active Hugging Face repositories (all): ${GLOBAL_ACTIVE_COUNT}"
  echo "  Active repositories outside this downloader: ${GLOBAL_OTHER_ACTIVE_COUNT}"
  echo "  Complete in configured destinations: ${LIST_COMPLETE_COUNT}"
  echo "  Complete in Hugging Face cache only: ${LIST_CACHED_COMPLETE_COUNT}"
  echo "  Partial/unverified: ${LIST_PARTIAL_COUNT}"
  echo "  Not present: ${LIST_NOT_PRESENT_COUNT}"
  print_model_size_summary
  if [[ $PROCESS_INSPECTION_WARNED -eq 1 ]]; then
    echo "  Warning: some process command lines were not readable"
  fi
  exit 0
fi

download_hf_model() {
  local job_name="$1"
  local repository_id="$2"
  local revision="$3"
  local destination_root="$4"
  local destination_name="$5"
  local scope="$6"
  shift 6
  local destination="${destination_root}/${destination_name}"
  local completion_marker="${destination_root}/.${destination_name}.download-complete"
  local size_manifest="${destination_root}/.${destination_name}.download-sizes.tsv"
  local source_manifest="${destination_root}/.${destination_name}.download-source.tsv"
  local hf_command=(uvx hf download "$repository_id" --revision "$revision")
  local had_existing_files=0

  echo
  echo "[JOB] ${job_name}"
  echo "      https://huggingface.co/${repository_id}"
  echo "      revision ${revision}"
  echo "      -> ${destination}"

  if find_active_download "$repository_id" "$destination"; then
    echo "[ACTIVE] Another process is downloading this model."
    echo "         PID ${ACTIVE_PROCESS_ID}: ${ACTIVE_PROCESS_COMMAND}"
    echo "[SKIP] Leaving the active download untouched."
    ACTIVE_JOBS+=("${job_name} (PID ${ACTIVE_PROCESS_ID})")
    return 0
  fi

  if verify_completed_state \
    "$destination" "$repository_id" "$revision" "$scope" \
    "$completion_marker" "$size_manifest" "$source_manifest"; then
    echo "[SKIP] Source and size manifests confirm the pinned full snapshot."
    COMPLETE_JOBS+=("$job_name")
    return 0
  fi

  if verify_external_completed_state \
    "$repository_id" "$revision" "$destination" "$scope" "$@"; then
    echo "[SKIP] Existing download is complete (${EXTERNAL_COMPLETION_SOURCE})."
    echo "[ADOPT] Recording source and size manifests for future checks."
    if ! write_size_manifest "$destination" "$size_manifest"; then
      echo "[ERROR] Could not record the completion manifest for ${job_name}." >&2
      return 1
    fi
    if ! write_source_manifest \
      "$repository_id" "$revision" "$scope" "$source_manifest"; then
      echo "[ERROR] Could not record the source manifest for ${job_name}." >&2
      return 1
    fi
    touch "$completion_marker"
    COMPLETE_JOBS+=("$job_name")
    return 0
  fi

  if [[ -e "$destination" ]] && [[ ! -d "$destination" ]]; then
    echo "[ERROR] Destination exists but is not a directory: ${destination}" >&2
    return 1
  fi

  if [[ -d "$destination" ]] \
    && [[ -n "$(find "$destination" -type f -not -path '*/.cache/huggingface/*' -print -quit 2>/dev/null)" ]]; then
    had_existing_files=1
    echo "[INCOMPLETE] The pinned full snapshot has not been verified."
    echo "[RETRY] Hugging Face will reconcile and fetch missing or outdated files."
  fi

  mkdir -p "$destination"
  if [[ "$scope" == "full" ]]; then
    echo "[DOWNLOAD] Every repository file at revision ${revision}."
  else
    echo "[DOWNLOAD] Complete ${scope} package at revision ${revision}."
    hf_command+=("$@")
  fi
  hf_command+=(--local-dir "$destination")

  if ! "${hf_command[@]}"; then
    echo "[ERROR] Hugging Face download failed for ${job_name}." >&2
    echo "        Rerun the script to retry; completed jobs will be skipped." >&2
    return 1
  fi

  if ! verify_download "$destination"; then
    echo "[ERROR] Core model-file verification failed for ${job_name}." >&2
    return 1
  fi

  if [[ "$scope" == "full" ]]; then
    if ! verify_hf_snapshot_with_cli "$repository_id" "$revision" "$destination"; then
      echo "[ERROR] Full repository verification failed for ${job_name}." >&2
      report_hf_verification_failure >&2
      return 1
    fi
  elif ! verify_required_files "$destination" "$@" \
    || ! verify_hf_language_package_with_cli \
      "$repository_id" "$revision" "$destination"; then
    echo "[ERROR] Language-package verification failed for ${job_name}." >&2
    report_hf_verification_failure >&2
    return 1
  fi

  if ! write_size_manifest "$destination" "$size_manifest"; then
    echo "[ERROR] Could not record the completed file-size manifest for ${job_name}." >&2
    return 1
  fi

  if ! write_source_manifest \
    "$repository_id" "$revision" "$scope" "$source_manifest"; then
    echo "[ERROR] Could not record the source manifest for ${job_name}." >&2
    return 1
  fi

  touch "$completion_marker"
  if [[ $had_existing_files -eq 1 ]]; then
    RETRIED_JOBS+=("$job_name")
  else
    DOWNLOADED_JOBS+=("$job_name")
  fi
  echo "[DONE] ${job_name}"
}

download_git_toolchain() {
  local job_name="$1"
  local repository_url="$2"
  local revision="$3"
  local destination_root="$4"
  local destination_name="$5"
  local destination="${destination_root}/${destination_name}"
  local completion_marker="${destination_root}/.${destination_name}.download-complete"

  echo
  echo "[JOB] ${job_name} (GitHub toolchain; not a Hugging Face model)"
  echo "      ${repository_url}"
  echo "      revision ${revision}"
  echo "      -> ${destination}"

  if find_active_download "kaldi-asr/kaldi" "$destination"; then
    echo "[ACTIVE] Another process is cloning this toolchain."
    echo "         PID ${ACTIVE_PROCESS_ID}: ${ACTIVE_PROCESS_COMMAND}"
    echo "[SKIP] Leaving the active clone untouched."
    ACTIVE_JOBS+=("${job_name} (PID ${ACTIVE_PROCESS_ID})")
    return 0
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] git is required for the Kaldi toolchain." >&2
    return 1
  fi

  if verify_git_checkout "$repository_url" "$revision" "$destination"; then
    touch "$completion_marker"
    echo "[SKIP] Existing Kaldi clone passed Git object verification."
    COMPLETE_JOBS+=("$job_name")
    return 0
  fi

  if [[ -e "$destination" ]]; then
    echo "[ERROR] Kaldi destination exists but is not a verifiably complete clone:" >&2
    echo "        ${destination}" >&2
    echo "        The script will not delete, reset, or overwrite it." >&2
    echo "        Repair or move it manually, then rerun this script." >&2
    return 1
  fi

  if ! git clone "$repository_url" "$destination"; then
    echo "[ERROR] Clone failed for ${job_name}. Rerun the script to retry." >&2
    return 1
  fi

  if ! git -c "safe.directory=${destination}" -C "$destination" \
    checkout --detach "$revision"; then
    echo "[ERROR] Could not check out pinned Kaldi revision ${revision}." >&2
    return 1
  fi

  if ! verify_git_checkout "$repository_url" "$revision" "$destination"; then
    echo "[ERROR] Pinned Kaldi checkout failed verification." >&2
    return 1
  fi

  touch "$completion_marker"
  DOWNLOADED_JOBS+=("$job_name")
  echo "[DONE] ${job_name}"
}

run_hf_job() {
  local job_name="$1"
  if ! download_hf_model "$@"; then
    FAILED_JOBS+=("$job_name")
  fi
}

run_git_job() {
  local job_name="$1"
  if ! download_git_toolchain "$@"; then
    FAILED_JOBS+=("$job_name")
  fi
}

if ! command -v uvx >/dev/null 2>&1; then
  echo "[ERROR] uvx is required but was not found on PATH." >&2
  echo "        Install uv, then rerun this script." >&2
  exit 1
fi

mkdir -p "$TRANSFORMERS_ROOT" "$FRAMEWORK_SPEC_MODELS_ROOT" "$MULTIMODAL_MODELS_ROOT"

# Current shared Transformers image. The order starts with smaller plumbing
# checks, keeps Paza before both Whisper variants, and leaves MMS-1B until last.
run_hf_job \
  "wav2vec2-base-960h" \
  "facebook/wav2vec2-base-960h" \
  "$REV_WAV2VEC2_BASE_960H" \
  "$TRANSFORMERS_ROOT" \
  "wav2vec2-base-960h" \
  "full"

run_hf_job \
  "BookBot orthographic CTC" \
  "bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm" \
  "$REV_BOOKBOT_ORTHOGRAPHIC" \
  "$TRANSFORMERS_ROOT" \
  "wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-word-lm" \
  "full"

run_hf_job \
  "BookBot phoneme CTC" \
  "bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot" \
  "$REV_BOOKBOT_PHONEME" \
  "$TRANSFORMERS_ROOT" \
  "wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot" \
  "full"

run_hf_job \
  "w2v-bert-2.0-swahili-asr" \
  "badrex/w2v-bert-2.0-swahili-asr" \
  "$REV_W2V_BERT_SWAHILI" \
  "$TRANSFORMERS_ROOT" \
  "w2v-bert-2.0-swahili-asr" \
  "full"

run_hf_job \
  "hubert-large-ls960-ft" \
  "facebook/hubert-large-ls960-ft" \
  "$REV_HUBERT_LARGE" \
  "$TRANSFORMERS_ROOT" \
  "hubert-large-ls960-ft" \
  "full"

run_hf_job \
  "paza-whisper-large-v3-turbo" \
  "microsoft/paza-whisper-large-v3-turbo" \
  "$REV_PAZA" \
  "$TRANSFORMERS_ROOT" \
  "paza-whisper-large-v3-turbo" \
  "full"

run_hf_job \
  "whisper-large" \
  "openai/whisper-large" \
  "$REV_WHISPER_LARGE" \
  "$TRANSFORMERS_ROOT" \
  "whisper-large" \
  "full"

run_hf_job \
  "whisper-large-v2" \
  "openai/whisper-large-v2" \
  "$REV_WHISPER_LARGE_V2" \
  "$TRANSFORMERS_ROOT" \
  "whisper-large-v2" \
  "full"

run_hf_job \
  "mms-1b-all" \
  "facebook/mms-1b-all" \
  "$REV_MMS_1B_ALL" \
  "$TRANSFORMERS_ROOT" \
  "mms-1b-all" \
  "language:swh" \
  ".gitattributes" \
  "README.md" \
  "adapter.swh.bin" \
  "adapter.swh.safetensors" \
  "config.json" \
  "create_adapters.py" \
  "create_vocab.py" \
  "download_vocabs.sh" \
  "model.safetensors" \
  "preprocessor_config.json" \
  "pytorch_model.bin" \
  "special_tokens_map.json" \
  "tokenizer_config.json" \
  "vocab.json" \
  "vocabs/swh.txt"

# Candidates that require a separate image, adapter, or toolchain.
run_git_job \
  "Kaldi toolkit" \
  "https://github.com/kaldi-asr/kaldi" \
  "$REV_KALDI" \
  "$FRAMEWORK_SPEC_MODELS_ROOT" \
  "kaldi"

run_hf_job \
  "Paza Phi-4 multimodal" \
  "microsoft/paza-Phi-4-multimodal-instruct" \
  "$REV_PHI4_MULTIMODAL" \
  "$MULTIMODAL_MODELS_ROOT" \
  "paza-Phi-4-multimodal-instruct" \
  "full"

run_hf_job \
  "Gemma 4 E2B Omni" \
  "google/gemma-4-E2B-it" \
  "$REV_GEMMA4_OMNI" \
  "$MULTIMODAL_MODELS_ROOT" \
  "gemma-4-E2B-it" \
  "full"

run_hf_job \
  "Qwen2.5 Omni 7B" \
  "Qwen/Qwen2.5-Omni-7B" \
  "$REV_QWEN25_OMNI" \
  "$MULTIMODAL_MODELS_ROOT" \
  "Qwen2.5-Omni-7B" \
  "full"

print_unresolved_choices

print_summary_group() {
  local label="$1"
  shift
  echo "  ${label}: $#"
  if [[ $# -gt 0 ]]; then
    printf '    - %s\n' "$@"
  fi
}

echo
echo "[SUMMARY]"
echo "  Process inspector: ${PROCESS_INSPECTOR}"
if [[ $PROCESS_INSPECTION_WARNED -eq 1 ]]; then
  echo "  Process-inspection warning: active jobs may not have been visible"
fi
print_summary_group "Active elsewhere; skipped" "${ACTIVE_JOBS[@]}"
print_summary_group "Already complete" "${COMPLETE_JOBS[@]}"
print_summary_group "Newly downloaded" "${DOWNLOADED_JOBS[@]}"
print_summary_group "Incomplete and successfully retried/reconciled" "${RETRIED_JOBS[@]}"
print_summary_group "Failed or needs manual repair" "${FAILED_JOBS[@]}"

if [[ ${#FAILED_JOBS[@]} -gt 0 ]]; then
  echo
  echo "One or more jobs could not be completed automatically." >&2
  echo "No existing model folder was deleted, reset, or overwritten." >&2
  echo "Review the errors above, then rerun; active and completed jobs will be skipped." >&2
  exit 1
fi

echo
echo "All resolved first-phase jobs are complete or currently running elsewhere."
