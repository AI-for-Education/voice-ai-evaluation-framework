"""Retrospectively add normalized provenance to existing run/evaluation metadata.

The command is dry-run by default. Pass ``--apply`` to atomically update JSON
metadata files. It never loads a model or decodes audio samples; optional audio
recovery reads WAV headers only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from inference.pipeline_provenance import (
    PIPELINE_PROVENANCE_SCHEMA_VERSION,
    build_evaluation_provenance,
    build_pipeline_provenance,
    evaluation_source_run_metadata_path,
    not_available,
    summarize_wav_headers,
)
from inference.provenance import canonical_json_sha256, file_identity


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.pipeline-provenance.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _host_path(raw_path: Any, *, repository_root: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    raw = raw_path.strip().replace("\\", "/")
    if raw == "/work":
        return repository_root
    if raw.startswith("/work/"):
        return repository_root / raw.removeprefix("/work/")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repository_root / path


def _json_objects(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
    except json.JSONDecodeError:
        pass
    rows: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            row, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        index = next_index
    return rows


def _recover_audio_paths(
    metadata: dict[str, Any],
    *,
    repository_root: Path,
) -> tuple[list[str], str | None]:
    input_metadata = metadata.get("input")
    if not isinstance(input_metadata, dict):
        return [], "Historical run metadata has no input section"
    manifest_path = _host_path(
        input_metadata.get("audio_manifest"), repository_root=repository_root
    )
    if manifest_path is not None:
        if not manifest_path.is_file():
            return [], f"Recorded audio manifest is no longer available: {manifest_path}"
        paths: list[str] = []
        seen: set[str] = set()
        for row in _json_objects(manifest_path):
            candidate = _host_path(
                row.get("audio_filepath"), repository_root=repository_root
            )
            if candidate is None:
                continue
            value = str(candidate)
            if value not in seen:
                paths.append(value)
                seen.add(value)
        if paths:
            return paths, None
        return [], f"No audio_filepath values were recoverable from {manifest_path}"

    root_path = _host_path(
        input_metadata.get("root_audio_dir"), repository_root=repository_root
    )
    if root_path is not None:
        if not root_path.is_dir():
            return [], f"Recorded audio root is no longer available: {root_path}"
        return [
            str(path)
            for path in sorted(
                (
                    item
                    for item in root_path.rglob("*")
                    if item.is_file() and item.suffix.lower() == ".wav"
                ),
                key=lambda item: str(item).lower(),
            )
        ], None
    return [], "Historical run metadata did not record an audio manifest or root"


def _profile_candidates(repository_root: Path) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    inference_root = repository_root / "inference"
    for family in inference_root.iterdir():
        profiles = family / "profiles"
        if not profiles.is_dir():
            continue
        for path in profiles.glob("*.yaml"):
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                if raw_line.startswith("id:"):
                    candidates[raw_line.partition(":")[2].strip()] = path
                    break
    return candidates


def _profile_link(
    metadata: dict[str, Any],
    *,
    repository_root: Path,
    candidates: dict[str, Path],
) -> dict[str, Any]:
    profile = metadata.get("profile")
    embedded = profile if isinstance(profile, dict) else {}
    link: dict[str, Any] = {
        "embedded_profile_sha256": canonical_json_sha256(embedded),
        "embedded_profile_status": "historical_run_record",
    }
    if isinstance(metadata.get("profile_path"), str):
        link["recorded_path"] = metadata["profile_path"]
    if isinstance(metadata.get("profile_identity"), dict):
        link["recorded_identity"] = metadata["profile_identity"]
    profile_id = embedded.get("id")
    candidate = candidates.get(profile_id) if isinstance(profile_id, str) else None
    if candidate is not None and candidate.is_file():
        link["current_profile_candidate"] = {
            **file_identity(candidate),
            "relation": (
                "same_profile_id_only; the embedded historical profile remains "
                "authoritative for the completed run"
            ),
        }
    return link


def _transcriptions_path(
    metadata_path: Path,
    metadata: dict[str, Any],
    *,
    repository_root: Path,
) -> Path:
    output = metadata.get("output")
    if isinstance(output, dict):
        candidate = _host_path(
            output.get("transcriptions"), repository_root=repository_root
        )
        if candidate is not None and candidate.is_file():
            return candidate
    return metadata_path.parent / "transcriptions.jsonl"


def _count_not_available(value: Any) -> int:
    if isinstance(value, dict):
        own = int(
            value.get("status") == "not_available"
            and isinstance(value.get("reason"), str)
        )
        return own + sum(_count_not_available(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_not_available(item) for item in value)
    return 0


def _not_available_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        if (
            value.get("status") == "not_available"
            and isinstance(value.get("reason"), str)
        ):
            return [prefix or "<root>"]
        paths: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_not_available_paths(child, child_prefix))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_not_available_paths(child, f"{prefix}[{index}]"))
        return paths
    return []


def backfill(
    *,
    output_root: str | Path,
    repository_root: str | Path,
    apply: bool,
    refresh: bool = False,
) -> dict[str, Any]:
    output = Path(output_root)
    repository = Path(repository_root).resolve()
    candidates = _profile_candidates(repository)
    audio_cache: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    recovered_run_metadata: dict[Path, dict[str, Any]] = {}
    not_available_counts: Counter[str] = Counter()
    summary: dict[str, Any] = {
        "mode": "apply" if apply else "dry_run",
        "run_metadata": {"discovered": 0, "updated": 0, "skipped": 0, "errors": 0},
        "evaluation_metadata": {
            "discovered": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
        },
        "not_available_fields": 0,
        "error_details": [],
    }

    run_paths = sorted(output.rglob("run_metadata.json"))
    summary["run_metadata"]["discovered"] = len(run_paths)
    for metadata_path in run_paths:
        try:
            metadata = _read_json(metadata_path)
            if not refresh and isinstance(metadata.get("pipeline_provenance"), dict):
                summary["run_metadata"]["skipped"] += 1
                continue
            profile = metadata.get("profile")
            if not isinstance(profile, dict):
                profile = {}
            backend = metadata.get("backend")
            if not isinstance(backend, dict):
                backend = {}
            runtime = metadata.get("runtime")
            if not isinstance(runtime, dict):
                runtime = {}
            model_identity = metadata.get("model_identity")
            if not isinstance(model_identity, dict):
                model_identity = None
            input_metadata = metadata.get("input")
            input_mapping = input_metadata if isinstance(input_metadata, dict) else {}
            cache_key = (
                input_mapping.get("audio_manifest")
                if isinstance(input_mapping.get("audio_manifest"), str)
                else None,
                input_mapping.get("root_audio_dir")
                if isinstance(input_mapping.get("root_audio_dir"), str)
                else None,
            )
            audio_summary = input_mapping.get("audio_header_summary")
            limitations: list[str] = []
            if not isinstance(audio_summary, dict):
                if cache_key not in audio_cache:
                    audio_paths, reason = _recover_audio_paths(
                        metadata, repository_root=repository
                    )
                    audio_cache[cache_key] = (
                        summarize_wav_headers(audio_paths)
                        if audio_paths
                        else not_available(reason or "Input audio could not be recovered")
                    )
                audio_summary = audio_cache[cache_key]
            if audio_summary.get("status") == "not_available":
                limitations.append(str(audio_summary.get("reason")))

            pipeline = build_pipeline_provenance(
                profile=profile,
                backend=backend,
                runtime=runtime,
                model_identity=model_identity,
                profile_link=_profile_link(
                    metadata,
                    repository_root=repository,
                    candidates=candidates,
                ),
                run_metadata_path=metadata_path,
                transcriptions_path=_transcriptions_path(
                    metadata_path, metadata, repository_root=repository
                ),
                input_audio_summary=audio_summary,
                recording_mode="retrospective_recovery",
                limitations=limitations,
            )
            metadata["provenance_schema_version"] = (
                PIPELINE_PROVENANCE_SCHEMA_VERSION
            )
            metadata["pipeline_provenance"] = pipeline
            recovered_run_metadata[metadata_path.resolve()] = metadata
            summary["not_available_fields"] += _count_not_available(pipeline)
            not_available_counts.update(_not_available_paths(pipeline))
            if apply:
                _atomic_write_json(metadata_path, metadata)
            summary["run_metadata"]["updated"] += 1
        except Exception as exc:  # keep auditing other historical files
            summary["run_metadata"]["errors"] += 1
            summary["error_details"].append(
                {"path": str(metadata_path), "error": str(exc)}
            )

    evaluation_paths = sorted(output.rglob("evaluation_metadata.json"))
    summary["evaluation_metadata"]["discovered"] = len(evaluation_paths)
    for metadata_path in evaluation_paths:
        try:
            metadata = _read_json(metadata_path)
            if not refresh and isinstance(metadata.get("pipeline_provenance"), dict):
                summary["evaluation_metadata"]["skipped"] += 1
                continue
            manifest_in = metadata.get("source_manifest")
            manifest_path = _host_path(manifest_in, repository_root=repository)
            manifest_value = manifest_path if manifest_path is not None else str(manifest_in or "")
            requested = str(
                metadata.get("requested_scoring_representation")
                or metadata.get("output_namespace")
                or "not_available"
            )
            scoring_units = str(metadata.get("effective_scoring_units") or "not_available")
            namespace = str(metadata.get("output_namespace") or metadata_path.parent.name)
            limitations = [
                "The exact historical reference-view metadata path was not recorded; the scored source manifest identity remains available."
            ]
            source_run_path = evaluation_source_run_metadata_path(manifest_value)
            pipeline = build_evaluation_provenance(
                base=metadata_path.parent,
                manifest_in=manifest_value,
                requested_representation=requested,
                scoring_units=scoring_units,
                namespace=namespace,
                reference_metadata_path=None,
                recording_mode="retrospective_recovery",
                limitations=limitations,
                run_metadata_override=(
                    recovered_run_metadata.get(source_run_path.resolve())
                    if source_run_path is not None
                    else None
                ),
            )
            metadata["schema_version"] = max(int(metadata.get("schema_version", 1)), 2)
            metadata["provenance_schema_version"] = (
                PIPELINE_PROVENANCE_SCHEMA_VERSION
            )
            metadata["pipeline_provenance"] = pipeline
            summary["not_available_fields"] += _count_not_available(pipeline)
            not_available_counts.update(_not_available_paths(pipeline))
            if apply:
                _atomic_write_json(metadata_path, metadata)
            summary["evaluation_metadata"]["updated"] += 1
        except Exception as exc:  # keep auditing other historical files
            summary["evaluation_metadata"]["errors"] += 1
            summary["error_details"].append(
                {"path": str(metadata_path), "error": str(exc)}
            )
    summary["not_available_by_path"] = dict(sorted(not_available_counts.items()))
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or apply an idempotent, non-inference provenance backfill "
            "to existing run and evaluation metadata."
        )
    )
    parser.add_argument(
        "--output_root",
        default="input_output_data/output",
        help="Root containing transcripts and evaluations",
    )
    parser.add_argument(
        "--repository_root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically update files; without this flag the command is a dry run",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild already-present pipeline provenance using current code",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = backfill(
        output_root=args.output_root,
        repository_root=args.repository_root,
        apply=args.apply,
        refresh=args.refresh,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(bool(summary["error_details"]))


if __name__ == "__main__":
    raise SystemExit(main())
