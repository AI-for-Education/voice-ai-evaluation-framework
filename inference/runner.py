"""Shared ordering, output, and provenance handling for ASR adapters."""

from __future__ import annotations

import json
import platform
import re
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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


def run_backend(
    *,
    backend: ASRBackend,
    profile: InferenceProfile,
    profile_path: str,
    model_path: str,
    root_audio_dir: str | None,
    audio_manifest: str | None,
    output_root: str,
    batch_size: int,
    smoke_test: bool = False,
    fail_on_error: bool = False,
    startup_warnings: Sequence[warnings.WarningMessage] = (),
) -> Path:
    """Run an adapter and write the stable transcription/metadata handoff."""
    if batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")
    started = datetime.now(timezone.utc)
    result_count = 0
    non_empty_result_count = 0
    error_count = 0
    backend_metadata: dict[str, Any] = {}
    observed_warnings = _warning_rows(startup_warnings, phase="initialization")
    audio_paths: list[str] = []
    staging_destination: Path | None = None

    try:
        audio_paths = resolve_audio_paths(
            root_audio_dir=root_audio_dir,
            audio_manifest=audio_manifest,
        )
        candidate = resolve_transcript_output_dir(
            output_root=output_root,
            inference_setup_id=profile.inference_setup_id,
            smoke_test=smoke_test,
            started_at=started,
        )
        destination, staging_destination = _reserve_transcript_output_dir(candidate)
        output_manifest = staging_destination / "transcriptions.jsonl"
        metadata_path = staging_destination / "run_metadata.json"
        published_output_manifest = destination / output_manifest.name
        published_metadata_path = destination / metadata_path.name
        backend_metadata = backend.metadata()
        print(f"[INFO] Inference run: {destination.name}")
        print(f"[INFO] Device: {backend_metadata.get('device', 'unknown')}")
        print(f"[INFO] Audio files: {len(audio_paths)} | batch_size={batch_size}")
        print(f"[INFO] Transcript output: {published_output_manifest}")
        with output_manifest.open("w", encoding="utf-8") as output:
            with tqdm(
                total=len(audio_paths),
                desc=f"Transcribing {profile.inference_setup_id}",
                unit="file",
                dynamic_ncols=True,
            ) as progress:
                for batch_paths in _chunks(audio_paths, batch_size):
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        warnings.simplefilter("always")
                        results = backend.transcribe_batch(batch_paths)
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
                    progress.update(len(batch_paths))
        backend_metadata = backend.metadata()

        profile_payload = profile.to_dict()
        profile_file = file_identity(profile_path)
        profile_file["canonical_content_sha256"] = canonical_json_sha256(
            profile_payload
        )
        selected_artifact_files = backend_metadata.get("selected_artifact_files")
        if not (
            isinstance(selected_artifact_files, list)
            and all(isinstance(item, str) for item in selected_artifact_files)
        ):
            selected_artifact_files = None
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
            "repository": git_identity(),
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
        # Evaluation consumes the adapter's unmodified pred_text.
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
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
    finally:
        try:
            backend.close()
        finally:
            if staging_destination is not None and staging_destination.exists():
                shutil.rmtree(staging_destination)
