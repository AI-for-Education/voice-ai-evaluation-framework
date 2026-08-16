"""Shared ordering, output, and provenance handling for ASR backends."""

from __future__ import annotations

import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from inference.common import resolve_audio_paths
from inference.contracts import ASRBackend, TranscriptionResult
from inference.profile import ModelProfile


def _chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _safe_run_component(value: str) -> str:
    """Return a readable directory component that cannot introduce subdirectories."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or "model"


def resolve_transcript_output_dir(
    *,
    output_root: str | Path,
    profile_id: str,
    smoke_test: bool,
    started_at: datetime,
) -> Path:
    """Build the standard transcript directory for one model run."""
    timestamp = started_at.astimezone(timezone.utc).strftime(
        "%Y_%m_%d_%H_%M_%S_UTC"
    )
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
) -> Path:
    """Run an adapter and write the stable transcription/metadata handoff."""
    if batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")
    started = datetime.now(timezone.utc)
    result_count = 0
    error_count = 0
    postprocessed_count = 0
    postprocessed_words_removed = 0
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
                    results = backend.transcribe_batch(batch_paths)
                    if len(results) != len(batch_paths):
                        raise RuntimeError(
                            "Backend returned an unexpected number of results: "
                            f"expected {len(batch_paths)}, got {len(results)}"
                        )
                    for expected_path, result in zip(batch_paths, results):
                        if not isinstance(result, TranscriptionResult):
                            raise TypeError("Backend returned a non-TranscriptionResult value")
                        if result.audio_filepath != expected_path:
                            raise RuntimeError(
                                "Backend changed result order: "
                                f"expected {expected_path}, got {result.audio_filepath}"
                            )
                        if result.raw_pred_text is not None:
                            postprocessed_count += 1
                            raw_words = len(result.raw_pred_text.split())
                            scored_words = len(result.pred_text.split())
                            postprocessed_words_removed += max(
                                0, raw_words - scored_words
                            )
                        output.write(json.dumps(result.to_row(), ensure_ascii=False) + "\n")
                        result_count += 1
                        error_count += int(result.error is not None)
                    output.flush()
                    progress.update(len(batch_paths))
    finally:
        backend.close()

    metadata: dict[str, Any] = {
        "profile": profile.to_dict(),
        "profile_path": str(Path(profile_path)),
        "model_path": model_path,
        "backend": backend_metadata,
        "runtime": {
            "python": platform.python_version(),
            "batch_size": batch_size,
        },
        "input": {
            "root_audio_dir": root_audio_dir,
            "audio_manifest": audio_manifest,
            "resolved_items": len(audio_paths),
        },
        "output": {
            "transcriptions": str(output_manifest),
            "directory": str(destination),
            "smoke_test": smoke_test,
            "results": result_count,
            "errors": error_count,
        },
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    hallucination_guard = backend_metadata.get("hallucination_guard")
    if hallucination_guard is not None or postprocessed_count:
        metadata["postprocessing"] = {
            "stage": "post_decode_pre_evaluation",
            "method": "hallucination_guard",
            "scored_text_field": "pred_text",
            "raw_text_field": "raw_pred_text",
            "adjusted_results": postprocessed_count,
            "total_words_removed": postprocessed_words_removed,
        }
        print(
            "[INFO] Post-processing: hallucination_guard adjusted "
            f"{postprocessed_count} result(s), removed "
            f"{postprocessed_words_removed} word(s); evaluation uses pred_text"
        )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[INFO] Completed: {result_count} file(s), {error_count} error(s) | "
        f"{output_manifest}"
    )
    return output_manifest
