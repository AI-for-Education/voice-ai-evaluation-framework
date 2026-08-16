"""Build and cache orthographic and Africa G2P IPA reference views."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable, Sequence

from egra_eval2.manifest_cleaner import clean_text


SCHEMA_VERSION = 1
ORTHOGRAPHIC_VIEW_NAME = "orthographic.sw.v1.jsonl"
METADATA_NAME = "reference_views.metadata.json"


class ReferenceViewError(ValueError):
    """Raised when reference views cannot be built or safely reused."""


@dataclass(frozen=True)
class ReferenceViewPaths:
    """Files created or reused for one source manifest."""

    output_dir: Path
    orthographic: Path
    ipa: Path
    metadata: Path
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


def _safe_version(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    if not safe:
        raise ReferenceViewError("Phonemizer version must not be empty")
    return safe


def _safe_language(value: str) -> str:
    language = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9_-]+", language):
        raise ReferenceViewError(
            "Language must contain only lowercase letters, digits, '_' or '-'"
        )
    return language


def ipa_reference_inventory(language: str) -> str:
    """Return the stable scoring-inventory name for an Africa G2P IPA view."""
    return f"africa_g2p_{_safe_language(language)}_ipa_v1"


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


def _africa_g2p_version() -> str:
    try:
        return version("africa-g2p")
    except PackageNotFoundError as exc:
        raise ReferenceViewError(
            "Africa G2P is not installed. The repository code is ready, but the "
            "Docker image must later be rebuilt with a pinned africa-g2p package."
        ) from exc


def _ipa_rows(rows: Sequence[dict], phonemize: PhonemizeBatch) -> list[dict]:
    """Convert CAN and REF in one G2P batch while preserving row identity."""
    text_count = len(rows)
    texts = [str(row["can_text"]) for row in rows]
    texts.extend(str(row["ref_text"]) for row in rows)
    ipa_values = phonemize(texts)
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


def _can_reuse(
    *,
    metadata_path: Path,
    orthographic_path: Path,
    ipa_path: Path,
    source_sha256: str,
    language: str,
    phonemizer_version: str,
) -> bool:
    if not (metadata_path.exists() and orthographic_path.exists() and ipa_path.exists()):
        return False
    metadata = _read_metadata(metadata_path)
    return (
        metadata.get("schema_version") == SCHEMA_VERSION
        and metadata.get("source", {}).get("sha256") == source_sha256
        and metadata.get("orthographic", {}).get("path") == orthographic_path.name
        and metadata.get("ipa", {}).get("path") == ipa_path.name
        and metadata.get("ipa", {}).get("language") == language
        and metadata.get("ipa", {}).get("producer") == "africa-g2p"
        and metadata.get("ipa", {}).get("producer_version") == phonemizer_version
        and metadata.get("ipa", {}).get("inventory")
        == ipa_reference_inventory(language)
    )


def build_reference_views(
    *,
    dataset_root: str | Path,
    manifest_in: str | Path,
    language: str = "swh",
    output_dir: str | Path | None = None,
    force: bool = False,
    phonemize: PhonemizeBatch | None = None,
    phonemizer_version: str | None = None,
) -> ReferenceViewPaths:
    """Build or safely reuse orthographic and Africa-G2P IPA reference views."""
    dataset_path = Path(dataset_root)
    manifest_path = Path(manifest_in)
    if not dataset_path.is_dir():
        raise ReferenceViewError(f"Dataset root does not exist: {dataset_path}")
    if not manifest_path.is_file():
        raise ReferenceViewError(f"Reference manifest does not exist: {manifest_path}")
    effective_language = _safe_language(language)

    effective_version = phonemizer_version or _africa_g2p_version()
    safe_version = _safe_version(effective_version)
    destination = Path(output_dir) if output_dir else default_output_dir(dataset_path)
    orthographic_path = destination / ORTHOGRAPHIC_VIEW_NAME
    ipa_path = destination / (
        f"phonemic.ipa.africa_g2p.{effective_language}.{safe_version}.jsonl"
    )
    metadata_path = destination / METADATA_NAME
    source_sha256 = _sha256(manifest_path)

    existing_files = [path for path in (orthographic_path, ipa_path, metadata_path) if path.exists()]
    if existing_files and not force:
        if _can_reuse(
            metadata_path=metadata_path,
            orthographic_path=orthographic_path,
            ipa_path=ipa_path,
            source_sha256=source_sha256,
            language=effective_language,
            phonemizer_version=effective_version,
        ):
            return ReferenceViewPaths(
                destination, orthographic_path, ipa_path, metadata_path, True
            )
        raise ReferenceViewError(
            "Reference-view files already exist but their source or configuration "
            "does not match. Inspect them, then rerun explicitly with --force."
        )

    orthographic_rows = _load_orthographic_rows(manifest_path)
    phonemize_batch = phonemize or (
        lambda texts: phonemize_with_africa_g2p(texts, language=effective_language)
    )
    ipa_rows = _ipa_rows(orthographic_rows, phonemize_batch)

    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(orthographic_path, orthographic_rows)
    _write_jsonl(ipa_path, ipa_rows)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "manifest": str(manifest_path.resolve()),
            "sha256": source_sha256,
            "rows": len(orthographic_rows),
        },
        "orthographic": {"path": orthographic_path.name},
        "ipa": {
            "path": ipa_path.name,
            "language": effective_language,
            "inventory": ipa_reference_inventory(effective_language),
            "producer": "africa-g2p",
            "producer_version": effective_version,
        },
    }
    _write_json(metadata_path, metadata)
    return ReferenceViewPaths(
        destination, orthographic_path, ipa_path, metadata_path, False
    )


def prepare_ipa_reference_view(
    *,
    dataset_root: str | Path,
    manifest_base_in: str | None,
    asr_manifests: Sequence[str] | None,
    logger: logging.Logger,
    reference_language: str = "swh",
) -> None:
    """Generate or reuse the dataset IPA cache for any declared model output."""
    if not manifest_base_in:
        return

    for manifest in asr_manifests or []:
        metadata_path = Path(manifest).with_name("run_metadata.json")
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read ASR run metadata: %s", metadata_path)
            continue
        profile = metadata.get("profile", {})
        output_units = profile.get("output_units")
        if output_units not in {"orthographic", "phoneme"}:
            continue

        views = build_reference_views(
            dataset_root=dataset_root,
            manifest_in=manifest_base_in,
            language=reference_language,
        )
        logger.info(
            "%s IPA reference view for %s model output: %s",
            "Reused" if views.reused else "Generated",
            output_units,
            views.ipa,
        )
        return
