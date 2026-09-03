"""Build and cache orthographic and exact-system IPA reference views."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from egra_eval2.manifest_cleaner import clean_text
from egra_eval2.reference.g2p import (
    G2PSystem,
    G2PSystemError,
    make_test_system,
    resolve_g2p_system,
)
from inference.pipeline_provenance import inference_profile_from_run_metadata


SCHEMA_VERSION = 2
ORTHOGRAPHIC_VIEW_NAME = "orthographic.sw.v1.jsonl"
METADATA_NAME = "reference_views.metadata.json"


class ReferenceViewError(ValueError):
    """Raised when reference views cannot be built or safely reused."""


@dataclass(frozen=True)
class ReferenceViewPaths:
    """Files created or reused for one source manifest and G2P system."""

    output_dir: Path
    orthographic: Path
    ipa: Path
    metadata: Path
    g2p_system: G2PSystem
    reused: bool


PhonemizeBatch = Callable[[Sequence[str]], list[str]]


def phonemize_with_africa_g2p(
    texts: Sequence[str], *, language: str
) -> list[str]:
    """Convert a batch of orthographic strings to IPA with Africa G2P."""
    try:
        from africa_g2p import AfricaPipeline
    except ImportError as exc:
        raise ReferenceViewError(
            "Africa G2P is not installed. Rebuild the Docker image with the "
            "pinned africa-g2p dependency before generating IPA references."
        ) from exc

    pipeline = AfricaPipeline(lang=language, output="ipa", unknown="passthrough")
    values = pipeline.run(list(texts), sep=" ")
    if not isinstance(values, list) or len(values) != len(texts):
        raise ReferenceViewError(
            "Africa G2P must return exactly one IPA value for every input text"
        )
    return [str(value) for value in values]


def default_output_dir(dataset_root: str | Path) -> Path:
    """Return the dataset-adjacent generated-reference directory."""
    return Path(dataset_root) / "_derived" / "reference_views"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ipa_reference_inventory(language: str) -> str:
    """Return the legacy Africa-G2P inventory name for compatibility."""
    normalized = language.strip().lower()
    if normalized not in {"sw", "swh"}:
        raise ReferenceViewError("Africa G2P reference inventory requires Swahili")
    return "africa_g2p_swh_ipa_v1"


def _load_orthographic_rows(path: Path) -> list[dict]:
    """Load the source manifest, retain stable fields, and clean its text."""
    rows: list[dict] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ReferenceViewError(
                    f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ReferenceViewError(
                    f"Expected a JSON object in {path} at line {line_number}"
                )
            item_id = str(row.get("audio_filepath", "")).strip()
            if not item_id:
                raise ReferenceViewError(
                    f"Missing audio_filepath in {path} at line {line_number}"
                )
            if item_id in seen_ids:
                raise ReferenceViewError(f"Duplicate audio_filepath in {path}: {item_id}")
            seen_ids.add(item_id)

            cleaned = {
                "audio_filepath": item_id,
                "can_text": clean_text(row.get("can_text", "")),
                "ref_text": clean_text(row.get("ref_text", "")),
            }
            if "duration" in row:
                cleaned["duration"] = row["duration"]
            rows.append(cleaned)

    if not rows:
        raise ReferenceViewError(f"Reference manifest contains no rows: {path}")
    return rows


def _ipa_rows(rows: Sequence[dict], phonemize: PhonemizeBatch) -> list[dict]:
    """Convert CAN and REF in one batch while preserving row identity."""
    text_count = len(rows)
    texts = [str(row["can_text"]) for row in rows]
    texts.extend(str(row["ref_text"]) for row in rows)
    try:
        ipa_values = phonemize(texts)
    except G2PSystemError as exc:
        raise ReferenceViewError(str(exc)) from exc
    if len(ipa_values) != text_count * 2:
        raise ReferenceViewError(
            "Phonemizer changed the number of rows; CAN, REF, and item IDs must align"
        )
    can_values = ipa_values[:text_count]
    ref_values = ipa_values[text_count:]

    ipa_rows: list[dict] = []
    for source, can_text, ref_text in zip(rows, can_values, ref_values):
        row = {
            "audio_filepath": source["audio_filepath"],
            "can_text": str(can_text).strip(),
            "ref_text": str(ref_text).strip(),
        }
        if "duration" in source:
            row["duration"] = source["duration"]
        ipa_rows.append(row)
    return ipa_rows


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_metadata(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceViewError(f"Invalid reference-view metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise ReferenceViewError(f"Invalid reference-view metadata: {path}")
    return payload


def _validate_indexed_view(
    output_dir: Path,
    entry: object,
    *,
    label: str,
) -> None:
    if not isinstance(entry, dict):
        raise ReferenceViewError(f"Indexed {label} metadata is not an object")
    recorded_path = entry.get("path")
    if not isinstance(recorded_path, str) or not recorded_path.strip():
        raise ReferenceViewError(f"Indexed {label} has no path")
    view_path = (output_dir / recorded_path).resolve()
    try:
        view_path.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise ReferenceViewError(
            f"Indexed {label} path escapes its reference-view directory: "
            f"{recorded_path}"
        ) from exc
    if not view_path.is_file():
        raise ReferenceViewError(f"Indexed {label} file does not exist: {view_path}")
    recorded_sha256 = entry.get("sha256")
    if not isinstance(recorded_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", recorded_sha256
    ):
        raise ReferenceViewError(f"Indexed {label} has no valid SHA-256 hash")
    if recorded_sha256 != _sha256(view_path):
        raise ReferenceViewError(
            f"Indexed {label} SHA-256 does not match its file: {view_path}"
        )


def validate_reference_view_index(
    output_dir: str | Path,
    metadata: dict,
    *,
    skip_system_ids: Sequence[str] = (),
    validate_orthographic: bool = True,
) -> None:
    """Validate every file path and hash retained by the shared view index."""
    destination = Path(output_dir)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ReferenceViewError(
            f"Reference-view metadata must use schema {SCHEMA_VERSION}"
        )
    if validate_orthographic:
        _validate_indexed_view(
            destination,
            metadata.get("orthographic"),
            label="orthographic reference view",
        )
    ipa_views = metadata.get("ipa_views")
    if not isinstance(ipa_views, dict):
        raise ReferenceViewError("Reference-view metadata has no IPA view index")
    skipped = set(skip_system_ids)
    for system_id, entry in sorted(ipa_views.items()):
        if system_id in skipped:
            continue
        _validate_indexed_view(
            destination,
            entry,
            label=f"IPA reference view {system_id}",
        )


def _test_system(
    *,
    tool_id: str,
    language: str,
    phonemize: PhonemizeBatch,
    phonemizer_version: str,
) -> G2PSystem:
    if tool_id == "babygruut":
        return make_test_system(
            tool_id=tool_id,
            display_name="babygruut",
            language="sw",
            inventory="babygruut_sw_ipa_v1",
            version_value=phonemizer_version,
            phonemize=phonemize,
        )
    return make_test_system(
        language=language,
        version_value=phonemizer_version,
        phonemize=phonemize,
    )


def build_reference_views(
    *,
    dataset_root: str | Path,
    manifest_in: str | Path,
    language: str = "swh",
    g2p_tool: str = "africa_g2p",
    output_dir: str | Path | None = None,
    force: bool = False,
    phonemize: PhonemizeBatch | None = None,
    phonemizer_version: str | None = None,
    g2p_system: G2PSystem | None = None,
) -> ReferenceViewPaths:
    """Build or safely reuse one exact-system IPA reference view."""
    dataset_path = Path(dataset_root)
    manifest_path = Path(manifest_in)
    if not dataset_path.is_dir():
        raise ReferenceViewError(f"Dataset root does not exist: {dataset_path}")
    if not manifest_path.is_file():
        raise ReferenceViewError(f"Reference manifest does not exist: {manifest_path}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", language):
        raise ReferenceViewError(
            "Language must contain only letters, digits, underscores, or hyphens"
        )

    try:
        if g2p_system is not None:
            system = g2p_system
        elif phonemize is not None:
            if not phonemizer_version:
                raise ReferenceViewError(
                    "phonemizer_version is required with a custom phonemizer"
                )
            system = _test_system(
                tool_id=g2p_tool,
                language=language,
                phonemize=phonemize,
                phonemizer_version=phonemizer_version,
            )
        else:
            system = resolve_g2p_system(g2p_tool, language=language)
    except G2PSystemError as exc:
        raise ReferenceViewError(str(exc)) from exc

    destination = Path(output_dir) if output_dir else default_output_dir(dataset_path)
    orthographic_path = destination / ORTHOGRAPHIC_VIEW_NAME
    ipa_path = destination / f"phonemic.ipa.{system.system_id}.jsonl"
    metadata_path = destination / METADATA_NAME
    source_sha256 = _sha256(manifest_path)

    ipa_views: dict[str, dict] = {}
    if metadata_path.exists():
        existing_metadata = _read_metadata(metadata_path)
        existing_source = existing_metadata.get("source", {})
        same_source = (
            isinstance(existing_source, dict)
            and existing_source.get("sha256") == source_sha256
        )
        if not same_source and not force:
            raise ReferenceViewError(
                "Reference-view files use a different source manifest; rerun "
                "explicitly with --force."
            )
        if same_source and existing_metadata.get("schema_version") == SCHEMA_VERSION:
            validate_reference_view_index(
                destination,
                existing_metadata,
                skip_system_ids=(system.system_id,) if force else (),
                validate_orthographic=not force,
            )
            raw_views = existing_metadata.get("ipa_views", {})
            if isinstance(raw_views, dict):
                ipa_views = dict(raw_views)
    elif (orthographic_path.exists() or ipa_path.exists()) and not force:
        raise ReferenceViewError(
            "Reference-view files exist without exact metadata; inspect them and "
            "rerun explicitly with --force."
        )

    expected_identity = {**system.metadata(), "path": ipa_path.name}
    existing_entry = ipa_views.get(system.system_id)
    if existing_entry is not None and not force:
        identity_matches = all(
            existing_entry.get(key) == value
            for key, value in expected_identity.items()
        )
        recorded_ipa_sha = existing_entry.get("sha256")
        orthographic_metadata = existing_metadata.get("orthographic", {})
        recorded_orthographic_sha = (
            orthographic_metadata.get("sha256")
            if isinstance(orthographic_metadata, dict)
            else None
        )
        if not identity_matches or not (
            orthographic_path.is_file() and ipa_path.is_file()
        ) or recorded_ipa_sha != _sha256(ipa_path) or (
            recorded_orthographic_sha != _sha256(orthographic_path)
        ):
            raise ReferenceViewError(
                "The indexed G2P reference view is incomplete or has mismatched "
                "identity; rerun explicitly with --force."
            )
        return ReferenceViewPaths(
            destination,
            orthographic_path,
            ipa_path,
            metadata_path,
            system,
            True,
        )
    if ipa_path.exists() and existing_entry is None and not force:
        raise ReferenceViewError(
            "An unindexed IPA view already exists for this exact G2P system; "
            "rerun explicitly with --force."
        )

    orthographic_rows = _load_orthographic_rows(manifest_path)
    ipa_rows = _ipa_rows(orthographic_rows, system.phonemize)
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(orthographic_path, orthographic_rows)
    _write_jsonl(ipa_path, ipa_rows)
    ipa_views[system.system_id] = {
        **expected_identity,
        "rows": len(ipa_rows),
        "sha256": _sha256(ipa_path),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "manifest": str(manifest_path.resolve()),
            "sha256": source_sha256,
            "rows": len(orthographic_rows),
        },
        "orthographic": {
            "path": orthographic_path.name,
            "rows": len(orthographic_rows),
            "sha256": _sha256(orthographic_path),
        },
        "ipa_views": ipa_views,
    }
    validate_reference_view_index(destination, metadata)
    _write_json(metadata_path, metadata)
    return ReferenceViewPaths(
        destination,
        orthographic_path,
        ipa_path,
        metadata_path,
        system,
        False,
    )


def prepare_ipa_reference_view(
    *,
    dataset_root: str | Path,
    audio_manifest: str | None,
    prediction_manifests: Sequence[str] | None,
    logger: logging.Logger,
    g2p_tool: str | None = None,
    reference_language: str = "swh",
) -> None:
    """Generate or reuse the explicitly selected dataset IPA cache."""
    if not audio_manifest or not g2p_tool:
        return

    for manifest in prediction_manifests or []:
        metadata_path = Path(manifest).with_name("run_metadata.json")
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read ASR run metadata: %s", metadata_path)
            continue
        profile = inference_profile_from_run_metadata(metadata)
        output_units = profile.get("output_units")
        if output_units not in {"orthographic", "phoneme"}:
            continue

        views = build_reference_views(
            dataset_root=dataset_root,
            manifest_in=audio_manifest,
            language=reference_language,
            g2p_tool=g2p_tool,
        )
        logger.info(
            "%s %s IPA reference view for %s model output: %s",
            "Reused" if views.reused else "Generated",
            views.g2p_system.system_id,
            output_units,
            views.ipa,
        )
        return
