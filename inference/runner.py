"""Shared ordering, output, and provenance handling for ASR adapters."""

from __future__ import annotations

import json
import platform
import re
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm import tqdm

from inference.common import resolve_audio_paths
from inference.contracts import ASRBackend, TranscriptionResult
from inference.profile import InferenceProfile
from inference.pipeline_provenance import (
    PIPELINE_PROVENANCE_SCHEMA_VERSION,
    build_pipeline_provenance,
    execution_environment_from_environment,
    summarize_wav_headers,
)
from inference.provenance import (
    canonical_json_sha256,
    file_identity,
    git_identity,
    model_artifact_identity,
    package_versions,
)


LOW_NON_EMPTY_HYPOTHESIS_FRACTION = 0.5


def _chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _safe_run_component(value: str) -> str:
    """Return a readable directory component that cannot introduce subdirectories."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or "inference_setup"


def _warning_rows(
    items: Sequence[warnings.WarningMessage], *, phase: str
) -> list[dict[str, Any]]:
    return [
        {
            "phase": phase,
            "category": item.category.__name__,
            "message": str(item.message),
            "filename": item.filename,
            "lineno": item.lineno,
        }
        for item in items
    ]


def _hypothesis_output_diagnostics(
    *, total: int, non_empty: int, errors: int
) -> dict[str, Any]:
    """Summarize adapter output without changing or gating its hypotheses."""
    blank = total - non_empty
    non_empty_fraction = non_empty / total if total else None
    is_low = (
        non_empty_fraction is not None
        and non_empty_fraction < LOW_NON_EMPTY_HYPOTHESIS_FRACTION
    )
    return {
        "counts": {
            "total": total,
            "non_empty": non_empty,
            "blank": blank,
            "errors": errors,
        },
        "coverage": {
            "status": "warning" if is_low else ("ok" if total else "not_assessed"),
            "warning_code": (
                "low_non_empty_hypothesis_coverage" if is_low else None
            ),
            "non_empty_fraction": non_empty_fraction,
            "minimum_expected_non_empty_fraction": (
                LOW_NON_EMPTY_HYPOTHESIS_FRACTION
            ),
            "affects_evaluation": False,
        },
    }


def resolve_transcript_output_dir(
    *,
    output_root: str | Path,
    inference_setup_id: str,
    smoke_test: bool,
    started_at: datetime,
) -> Path:
    """Build the transcript directory for one inference run."""
    if not inference_setup_id:
        raise ValueError("inference_setup_id is required")
    timestamp = started_at.astimezone(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_UTC")
    run_id = f"{_safe_run_component(inference_setup_id)}_{timestamp}"
    base = Path(output_root)
    if smoke_test:
        base = base / "smoke_tests"
    return base / "transcripts" / run_id


def _reserve_transcript_output_dir(candidate: Path) -> tuple[Path, Path]:
    """Reserve a collision-free final run path through a dot-prefixed staging path."""
    candidate.parent.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        destination = (
            candidate
            if suffix == 1
            else candidate.with_name(f"{candidate.name}__{suffix}")
        )
        staging = destination.with_name(f".{destination.name}.in_progress")
        if destination.exists():
            suffix += 1
            continue
        try:
            staging.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            suffix += 1
            continue
        if destination.exists():
            shutil.rmtree(staging)
            suffix += 1
            continue
        return destination, staging


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            # Windows scanners can briefly hold the destination after a write.
            time.sleep(0.05 * (attempt + 1))


def _resume_state(
    *,
    resume_run: str,
    profile_hash: str,
    audio_paths_hash: str,
    audio_paths: Sequence[str],
) -> tuple[Path, Path, datetime, int, int, int, dict[str, Any]]:
    staging = Path(resume_run).resolve()
    if not (
        staging.is_dir()
        and staging.name.startswith(".")
        and staging.name.endswith(".in_progress")
    ):
        raise ValueError(
            "--resume_run must name an existing .<run>.in_progress directory"
        )
    state_path = staging / "run_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Resume checkpoint is missing or invalid") from exc
    if not isinstance(state, dict) or state.get("checkpoint_schema_version") != 1:
        raise ValueError("Resume checkpoint has an unsupported schema")
    if state.get("status") not in {"in_progress", "completed"}:
        raise ValueError("Resume checkpoint is not in progress")
    if state.get("profile_sha256") != profile_hash:
        raise ValueError("Resume profile does not match the checkpoint")
    if state.get("audio_paths_sha256") != audio_paths_hash:
        raise ValueError("Resume audio input does not match the checkpoint")

    try:
        started = datetime.fromisoformat(str(state["started_at"]))
        destination = Path(str(state["destination"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("Resume checkpoint is missing required run identity") from exc
    if destination.exists():
        raise ValueError("Resume destination already exists")
    if destination.parent.resolve() != staging.parent.resolve():
        raise ValueError("Resume destination is outside the checkpoint directory")
    expected_destination = staging.with_name(
        staging.name[1 : -len(".in_progress")]
    ).resolve()
    if destination.resolve() != expected_destination:
        raise ValueError("Resume destination does not match the checkpoint directory")

    manifest_path = staging / "transcriptions.jsonl"
    rows: list[dict[str, Any]] = []
    committed_items = state.get("completed_items")
    if type(committed_items) is not int or committed_items < 0:
        raise ValueError("Resume checkpoint has an invalid completed item count")
    valid_bytes = 0
    repair_tail = False
    lines: list[bytes] = []
    if manifest_path.exists():
        try:
            lines = manifest_path.read_bytes().splitlines(keepends=True)
            for index, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except ValueError:
                    # A killed writer may leave an uncommitted, partial last row.
                    if (
                        index == len(lines) - 1
                        and not line.endswith(b"\n")
                        and len(rows) >= committed_items
                    ):
                        repair_tail = True
                        break
                    raise
                if not isinstance(row, dict):
                    raise ValueError
                rows.append(row)
                valid_bytes += len(line)
        except (OSError, ValueError) as exc:
            raise ValueError("Resume transcript checkpoint is corrupt") from exc
    if len(rows) > len(audio_paths):
        raise ValueError("Resume checkpoint contains too many transcript rows")
    for expected_path, row in zip(audio_paths, rows):
        if row.get("audio_filepath") != expected_path:
            raise ValueError("Resume transcript order does not match the input")
    non_empty = sum(bool(str(row.get("pred_text", "")).strip()) for row in rows)
    errors = sum("error" in row for row in rows)
    if committed_items > len(rows):
        raise ValueError("Resume state and transcript checkpoint disagree")
    if state["status"] == "completed" and len(rows) != len(audio_paths):
        raise ValueError("Completed checkpoint is missing transcript rows")
    if repair_tail:
        with manifest_path.open("r+b") as output:
            output.truncate(valid_bytes)
    elif lines and not lines[-1].endswith(b"\n"):
        with manifest_path.open("ab") as output:
            output.write(b"\n")
    # Complete, ordered transcript rows survive even if the state write did not.
    attempts = state.get("attempts", [])
    if attempts and attempts[-1]["status"] == "in_progress":
        attempts[-1].update(
            status="interrupted",
            ended_at=None,
            observations_complete=False,
        )
    if attempts:
        attempts[-1]["completed_items"] = len(rows) - attempts[-1]["start_item"]
        attempts[-1]["errors"] = errors - attempts[-1]["start_errors"]
    state["completed_items"] = len(rows)
    state["errors"] = errors
    return staging, destination, started, len(rows), non_empty, errors, state


def run_backend(
    *,
    backend: ASRBackend,
    profile: InferenceProfile,
    profile_path: str,
    model_path: str | None,
    root_audio_dir: str | None,
    audio_manifest: str | None,
    output_root: str,
    batch_size: int,
    smoke_test: bool = False,
    fail_on_error: bool = False,
    startup_warnings: Sequence[warnings.WarningMessage] = (),
    checkpoint: bool = False,
    resume_run: str | None = None,
    model_identity: Mapping[str, Any] | None = None,
) -> Path:
    """Run an adapter and write the stable transcription/metadata handoff."""
    if batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")
    if resume_run is not None and not checkpoint:
        raise ValueError("--resume_run requires checkpointing")
    started = datetime.now(timezone.utc)
    attempt_started = started
    result_count = 0
    non_empty_result_count = 0
    error_count = 0
    checkpoint_result_count = 0
    checkpoint_error_count = 0
    backend_metadata: dict[str, Any] = {}
    observed_warnings = _warning_rows(startup_warnings, phase="initialization")
    audio_paths: list[str] = []
    staging_destination: Path | None = None
    attempt: dict[str, Any] | None = None

    def save_attempt(
        status: str = "in_progress", *, failure_type: str | None = None
    ) -> None:
        if not checkpoint or attempt is None or staging_destination is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        attempt.update(
            status=status,
            ended_at=None if status == "in_progress" else now,
            updated_at=now,
            completed_items=checkpoint_result_count - attempt["start_item"],
            errors=checkpoint_error_count - attempt["start_errors"],
            inference_adapter=backend.metadata(),
            warnings=list(observed_warnings),
        )
        if failure_type is not None:
            attempt["failure_type"] = failure_type
        checkpoint_state.update(
            status="in_progress",
            completed_items=checkpoint_result_count,
            errors=checkpoint_error_count,
            updated_at=now,
        )
        checkpoint_state.pop("completed_at", None)
        _write_json_atomic(staging_destination / "run_state.json", checkpoint_state)

    try:
        audio_paths = resolve_audio_paths(
            root_audio_dir=root_audio_dir,
            audio_manifest=audio_manifest,
        )
        profile_payload = profile.to_dict()
        profile_file = file_identity(profile_path)
        profile_hash = canonical_json_sha256(profile_payload)
        profile_file["canonical_content_sha256"] = profile_hash
        audio_paths_hash = canonical_json_sha256(list(audio_paths))
        if resume_run is not None:
            (
                staging_destination,
                destination,
                started,
                result_count,
                non_empty_result_count,
                error_count,
                checkpoint_state,
            ) = _resume_state(
                resume_run=resume_run,
                profile_hash=profile_hash,
                audio_paths_hash=audio_paths_hash,
                audio_paths=audio_paths,
            )
            output_mode = "a"
        else:
            candidate = resolve_transcript_output_dir(
                output_root=output_root,
                inference_setup_id=profile.inference_setup_id,
                smoke_test=smoke_test,
                started_at=started,
            )
            destination, staging_destination = _reserve_transcript_output_dir(candidate)
            checkpoint_state = {
                "checkpoint_schema_version": 1,
                "status": "in_progress",
                "inference_setup_id": profile.inference_setup_id,
                "destination": str(destination.resolve()),
                "started_at": started.isoformat(),
                "profile_sha256": profile_hash,
                "audio_paths_sha256": audio_paths_hash,
                "resolved_items": len(audio_paths),
                "completed_items": 0,
                "errors": 0,
            }
            output_mode = "w"
        checkpoint_result_count = result_count
        checkpoint_error_count = error_count
        output_manifest = staging_destination / "transcriptions.jsonl"
        metadata_path = staging_destination / "run_metadata.json"
        published_output_manifest = destination / output_manifest.name
        published_metadata_path = destination / metadata_path.name
        backend_metadata = backend.metadata()
        selected_artifact_files = backend_metadata.get("selected_artifact_files")
        if not (
            isinstance(selected_artifact_files, list)
            and all(isinstance(item, str) for item in selected_artifact_files)
        ):
            selected_artifact_files = None
        if model_identity is not None:
            selected_model_artifact_identity = dict(model_identity)
        else:
            if model_path is None or profile.artifact is None:
                raise ValueError("A local model path or remote model identity is required")
            selected_model_artifact_identity = model_artifact_identity(
                model_path,
                artifact=profile.artifact,
                selected_files=selected_artifact_files,
            )
        execution_stack_metadata = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "batch_size": batch_size,
            "argv": list(sys.argv),
            "packages": package_versions(),
            "launch_context": execution_environment_from_environment(),
        }
        repository_identity = git_identity()
        if checkpoint:
            checkpoint_state.setdefault(
                "attempt_history_complete",
                resume_run is None or "attempts" in checkpoint_state,
            )
            attempts = checkpoint_state.setdefault("attempts", [])
            attempt = {
                "attempt_id": len(attempts) + 1,
                "started_at": attempt_started.isoformat(),
                "start_item": result_count,
                "start_errors": error_count,
                "observations_complete": True,
                "execution_stack": execution_stack_metadata,
                "repository": repository_identity,
                "model_artifact": {
                    "path": model_path,
                    "identity": dict(selected_model_artifact_identity),
                },
            }
            attempts.append(attempt)
            save_attempt()
        print(f"[INFO] Inference run: {destination.name}")
        print(f"[INFO] Device: {backend_metadata.get('device', 'unknown')}")
        print(f"[INFO] Audio files: {len(audio_paths)} | batch_size={batch_size}")
        print(f"[INFO] Transcript output: {published_output_manifest}")
        if result_count:
            print(f"[INFO] Resuming after {result_count} completed file(s)")
        remaining_audio_paths = audio_paths[result_count:]
        with output_manifest.open(output_mode, encoding="utf-8") as output:
            with tqdm(
                total=len(audio_paths),
                initial=result_count,
                desc=f"Transcribing {profile.inference_setup_id}",
                unit="file",
                dynamic_ncols=True,
            ) as progress:
                for batch_paths in _chunks(remaining_audio_paths, batch_size):
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        warnings.simplefilter("always")
                        try:
                            results = backend.transcribe_batch(batch_paths)
                        finally:
                            observed_warnings.extend(
                                _warning_rows(caught_warnings, phase="transcription")
                            )
                    if len(results) != len(batch_paths):
                        raise RuntimeError(
                            "Backend returned an unexpected number of results: "
                            f"expected {len(batch_paths)}, got {len(results)}"
                        )
                    for expected_path, result in zip(batch_paths, results):
                        if not isinstance(result, TranscriptionResult):
                            raise TypeError(
                                "Backend returned a non-TranscriptionResult value"
                            )
                        if result.audio_filepath != expected_path:
                            raise RuntimeError(
                                "Backend changed result order: "
                                f"expected {expected_path}, got {result.audio_filepath}"
                            )
                        # pred_text remains the unmodified adapter hypothesis.
                        output.write(
                            json.dumps(result.to_row(), ensure_ascii=False) + "\n"
                        )
                        result_count += 1
                        non_empty_result_count += int(bool(result.pred_text.strip()))
                        error_count += int(result.error is not None)
                    output.flush()
                    checkpoint_result_count = result_count
                    checkpoint_error_count = error_count
                    save_attempt()
                    progress.update(len(batch_paths))
        backend_metadata = backend.metadata()
        if model_identity is not None:
            statistics = backend_metadata.get("statistics")
            if (
                selected_model_artifact_identity.get("kind") == "remote_api_model"
                and isinstance(statistics, dict)
                and isinstance(statistics.get("models_returned"), dict)
            ):
                selected_model_artifact_identity["models_returned"] = dict(
                    statistics["models_returned"]
                )
        input_audio_summary = summarize_wav_headers(audio_paths)
        pipeline_provenance = build_pipeline_provenance(
            profile=profile_payload,
            adapter_metadata=backend_metadata,
            execution_stack=execution_stack_metadata,
            model_artifact_identity=selected_model_artifact_identity,
            profile_link={
                "path": str(Path(profile_path)),
                "identity": profile_file,
            },
            run_metadata_path=published_metadata_path,
            transcriptions_path=output_manifest,
            published_transcriptions_path=published_output_manifest,
            input_audio_summary=input_audio_summary,
            recording_mode="run_time",
        )
        output_diagnostics = _hypothesis_output_diagnostics(
            total=result_count,
            non_empty=non_empty_result_count,
            errors=error_count,
        )
        metadata: dict[str, Any] = {
            "metadata_schema_version": 2,
            "provenance_schema_version": PIPELINE_PROVENANCE_SCHEMA_VERSION,
            "status": "completed",
            "inference_setup_id": profile.inference_setup_id,
            "run_id": destination.name,
            "inference_profile": profile_payload,
            "inference_profile_path": str(Path(profile_path)),
            "inference_profile_identity": profile_file,
            "model_artifact": {
                "path": model_path,
                "identity": selected_model_artifact_identity,
            },
            "inference_adapter": backend_metadata,
            "execution_stack": execution_stack_metadata,
            "repository": repository_identity,
            "input": {
                "root_audio_dir": root_audio_dir,
                "audio_manifest": audio_manifest,
                "resolved_items": len(audio_paths),
                "audio_header_summary": input_audio_summary,
            },
            "output": {
                "transcriptions": str(published_output_manifest),
                "directory": str(destination),
                "smoke_test": smoke_test,
                "results": result_count,
                "errors": error_count,
                "hypotheses": output_diagnostics,
                "fail_on_error": fail_on_error,
            },
            "started_at": started.isoformat(),
            "warnings": observed_warnings,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_provenance": pipeline_provenance,
        }
        if checkpoint:
            save_attempt("completed")
            metadata["attempts"] = checkpoint_state["attempts"]
            metadata["attempt_history_complete"] = checkpoint_state[
                "attempt_history_complete"
            ]
            # Adapter, model, environment, and warning observations are per attempt;
            # output counts and transcripts cover the complete logical run.
            metadata["observation_attempt_id"] = attempt["attempt_id"]
        # Evaluation consumes the adapter's unmodified pred_text.
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if checkpoint:
            checkpoint_state["status"] = "completed"
            checkpoint_state["completed_items"] = result_count
            checkpoint_state["errors"] = error_count
            checkpoint_state["completed_at"] = metadata["completed_at"]
            _write_json_atomic(staging_destination / "run_state.json", checkpoint_state)
        staging_destination.rename(destination)
        staging_destination = None
        if output_diagnostics["coverage"]["status"] == "warning":
            print(
                "[WARNING] Low non-empty hypothesis coverage: "
                f"{non_empty_result_count}/{result_count} "
                f"({output_diagnostics['coverage']['non_empty_fraction']:.1%}); "
                "hypotheses were preserved unchanged"
            )
        print(
            f"[INFO] Completed: {result_count} file(s), {error_count} error(s) | "
            f"{published_output_manifest}"
        )
        if fail_on_error and error_count:
            raise SystemExit(
                f"Inference produced {error_count} per-file error(s); "
                f"see {published_metadata_path}"
            )
        return published_output_manifest
    except BaseException as exc:
        try:
            save_attempt(
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                failure_type=type(exc).__name__,
            )
        except Exception as logging_error:
            # Preserve the original error if storage is also unavailable.
            print(
                f"[WARNING] Could not save attempt metadata: {type(logging_error).__name__}",
                file=sys.stderr,
            )
        raise
    finally:
        try:
            backend.close()
        finally:
            if (
                staging_destination is not None
                and staging_destination.exists()
                and not checkpoint
            ):
                shutil.rmtree(staging_destination)
