"""Shared ordering, output, and provenance handling for ASR backends."""

from __future__ import annotations

import json
import platform
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from inference.common import resolve_audio_paths
from inference.contracts import ASRBackend, TranscriptionResult
from inference.profile import ModelProfile
from inference.pipeline_provenance import (
    PIPELINE_PROVENANCE_SCHEMA_VERSION,
    build_pipeline_provenance,
    runtime_launch_context_from_environment,
    summarize_wav_headers,
)
from inference.provenance import (
    canonical_json_sha256,
    file_identity,
    git_identity,
    model_identity,
    package_versions,
)


def _chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _safe_run_component(value: str) -> str:
    """Return a readable directory component that cannot introduce subdirectories."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or "model"


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


def resolve_transcript_output_dir(
    *,
    output_root: str | Path,
    profile_id: str,
    smoke_test: bool,
    started_at: datetime,
) -> Path:
    """Build the standard transcript directory for one model run."""
    timestamp = started_at.astimezone(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_UTC")
    run_name = f"{_safe_run_component(profile_id)}_{timestamp}"
    base = Path(output_root)
    if smoke_test:
        base = base / "smoke_tests"
    return base / "transcripts" / run_name


def run_backend(
    *,
    backend: ASRBackend,
    profile: ModelProfile,
    profile_path: str,
    model_path: str,
    root_audio_dir: str | None,
    audio_manifest: str | None,
    output_root: str,
    batch_size: int,
    smoke_test: bool = False,
    startup_warnings: Sequence[warnings.WarningMessage] = (),
) -> Path:
    """Run an adapter and write the stable transcription/metadata handoff."""
    if batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")
    started = datetime.now(timezone.utc)
    result_count = 0
    error_count = 0
    backend_metadata: dict[str, Any] = {}
    observed_warnings = _warning_rows(startup_warnings, phase="initialization")
    audio_paths: list[str] = []

    try:
        audio_paths = resolve_audio_paths(
            root_audio_dir=root_audio_dir,
            audio_manifest=audio_manifest,
        )
        destination = resolve_transcript_output_dir(
            output_root=output_root,
            profile_id=profile.id,
            smoke_test=smoke_test,
            started_at=started,
        )
        destination.mkdir(parents=True, exist_ok=True)
        output_manifest = destination / "transcriptions.jsonl"
        metadata_path = destination / "run_metadata.json"
        backend_metadata = backend.metadata()
        print(f"[INFO] Model run: {destination.name}")
        print(f"[INFO] Device: {backend_metadata.get('device', 'unknown')}")
        print(f"[INFO] Audio files: {len(audio_paths)} | batch_size={batch_size}")
        print(f"[INFO] Transcript output: {destination}")
        with output_manifest.open("w", encoding="utf-8") as output:
            with tqdm(
                total=len(audio_paths),
                desc=f"Transcribing {profile.id}",
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
                        # pred_text remains the unmodified backend hypothesis.
                        output.write(
                            json.dumps(result.to_row(), ensure_ascii=False) + "\n"
                        )
                        result_count += 1
                        error_count += int(result.error is not None)
                    output.flush()
                    progress.update(len(batch_paths))
        backend_metadata = backend.metadata()
    finally:
        backend.close()

    profile_payload = profile.to_dict()
    profile_file = file_identity(profile_path)
    profile_file["canonical_content_sha256"] = canonical_json_sha256(profile_payload)
    selected_model_identity = model_identity(model_path, artifact=profile.artifact)
    runtime_metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "batch_size": batch_size,
        "argv": list(sys.argv),
        "packages": package_versions(),
        "launch_context": runtime_launch_context_from_environment(),
    }
    input_audio_summary = summarize_wav_headers(audio_paths)
    pipeline_provenance = build_pipeline_provenance(
        profile=profile_payload,
        backend=backend_metadata,
        runtime=runtime_metadata,
        model_identity=selected_model_identity,
        profile_link={
            "path": str(Path(profile_path)),
            "identity": profile_file,
        },
        run_metadata_path=metadata_path,
        transcriptions_path=output_manifest,
        input_audio_summary=input_audio_summary,
        recording_mode="run_time",
    )
    metadata: dict[str, Any] = {
        "provenance_schema_version": PIPELINE_PROVENANCE_SCHEMA_VERSION,
        "status": "completed",
        "profile": profile_payload,
        "profile_path": str(Path(profile_path)),
        "profile_identity": profile_file,
        "model_path": model_path,
        "model_identity": selected_model_identity,
        "repository": git_identity(),
        "backend": backend_metadata,
        "runtime": runtime_metadata,
        "input": {
            "root_audio_dir": root_audio_dir,
            "audio_manifest": audio_manifest,
            "resolved_items": len(audio_paths),
            "audio_header_summary": input_audio_summary,
        },
        "output": {
            "transcriptions": str(output_manifest),
            "directory": str(destination),
            "smoke_test": smoke_test,
            "results": result_count,
            "errors": error_count,
        },
        "started_at": started.isoformat(),
        "warnings": observed_warnings,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_provenance": pipeline_provenance,
    }
    # Evaluation consumes the backend's unmodified pred_text.
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[INFO] Completed: {result_count} file(s), {error_count} error(s) | "
        f"{output_manifest}"
    )
    return output_manifest
