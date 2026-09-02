"""Resolve orthographic or IPA text immediately before shared scoring."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import pandas as pd

from egra_eval2.reference.ipa_inventory_maps import (
    IPAInventoryMapError,
    build_ipa_aligner,
)
from egra_eval2.reference.views import (
    METADATA_NAME,
    ReferenceViewError,
    default_output_dir,
    phonemize_with_africa_g2p,
)


ALIGNED_IPA_SUFFIX = ".ipa_aligned.jsonl"


class ScoringRepresentationError(ValueError):
    """Raised when model and reference representations cannot be aligned."""


ScoringRepresentation = Literal[
    "auto",
    "orthographic",
    "ipa",
]
PhonemizeBatch = Callable[[Sequence[str]], list[str]]


@dataclass(frozen=True)
class IPAReferenceView:
    rows: dict[str, tuple[str, str]]
    inventory: str
    language: str
    producer: str
    producer_version: str


def align_ipa_inventory(text: object, *, source: str, target: str) -> str:
    """Normalize IPA and apply one approved inventory adapter."""
    try:
        return build_ipa_aligner(source, target)(text)
    except IPAInventoryMapError as exc:
        raise ScoringRepresentationError(str(exc)) from exc


def _load_json_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringRepresentationError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ScoringRepresentationError(f"Invalid {label}: {path}")
    return payload


def _run_metadata_path(manifest_in: str | Path) -> Path | None:
    manifest_path = Path(manifest_in)
    if manifest_path.parent.name != "manifests":
        return None
    run_dir = manifest_path.parent.parent
    if run_dir.parent.name != "evaluations":
        return None
    return (
        run_dir.parent.parent
        / "transcripts"
        / run_dir.name
        / "run_metadata.json"
    )


def _load_run_profile(manifest_in: str | Path) -> dict | None:
    metadata_path = _run_metadata_path(manifest_in)
    if metadata_path is None:
        return None
    if not metadata_path.is_file():
        raise ScoringRepresentationError(
            "ASR run metadata not found for standard evaluation manifest: "
            f"{metadata_path}"
        )
    metadata = _load_json_object(metadata_path, "ASR run metadata")
    profile = metadata.get("inference_profile")
    if not isinstance(profile, dict):
        raise ScoringRepresentationError(
            f"ASR run metadata has no inference profile: {metadata_path}"
        )
    return profile


def _load_ipa_reference(dataset_root: str | Path) -> IPAReferenceView:
    view_dir = default_output_dir(dataset_root)
    metadata_path = view_dir / METADATA_NAME
    if not metadata_path.is_file():
        raise ScoringRepresentationError(
            f"IPA reference metadata not found: {metadata_path}. "
            "Run run_manifest.sh first."
        )
    metadata = _load_json_object(metadata_path, "IPA reference metadata")
    ipa = metadata.get("ipa")
    if not isinstance(ipa, dict):
        raise ScoringRepresentationError(
            f"IPA reference metadata is incomplete: {metadata_path}"
        )
    view_name = ipa.get("path")
    inventory = ipa.get("inventory")
    language = ipa.get("language")
    producer = ipa.get("producer")
    producer_version = ipa.get("producer_version")
    if not isinstance(view_name, str) or not view_name.strip():
        raise ScoringRepresentationError(
            f"IPA reference metadata has no path: {metadata_path}"
        )
    if not isinstance(inventory, str) or not inventory.strip():
        raise ScoringRepresentationError(
            f"IPA reference metadata has no inventory: {metadata_path}"
        )
    for label, value in (
        ("language", language),
        ("producer", producer),
        ("producer_version", producer_version),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ScoringRepresentationError(
                f"IPA reference metadata has no {label}: {metadata_path}"
            )

    view_path = (view_dir / view_name).resolve()
    try:
        view_path.relative_to(view_dir.resolve())
    except ValueError as exc:
        raise ScoringRepresentationError(
            "IPA reference path escapes its dataset view directory"
        ) from exc
    if not view_path.is_file():
        raise ScoringRepresentationError(f"IPA reference view not found: {view_path}")

    rows: dict[str, tuple[str, str]] = {}
    with view_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoringRepresentationError(
                    f"Invalid IPA reference JSON at line {line_number}: {view_path}"
                ) from exc
            if not isinstance(row, dict):
                raise ScoringRepresentationError(
                    f"IPA reference row {line_number} is not an object: {view_path}"
                )
            item_id = str(row.get("audio_filepath", "")).strip()
            if not item_id or item_id in rows:
                raise ScoringRepresentationError(
                    "Missing or duplicate audio_filepath in IPA reference at "
                    f"line {line_number}: {view_path}"
                )
            if not isinstance(row.get("can_text"), str) or not isinstance(
                row.get("ref_text"), str
            ):
                raise ScoringRepresentationError(
                    "IPA reference requires string can_text and ref_text at "
                    f"line {line_number}: {view_path}"
                )
            rows[item_id] = (row["can_text"], row["ref_text"])
    return IPAReferenceView(
        rows=rows,
        inventory=inventory.strip(),
        language=language.strip(),
        producer=producer.strip(),
        producer_version=producer_version.strip(),
    )


def _text_values(values: pd.Series) -> list[str]:
    result: list[str] = []
    for value in values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            result.append("")
        else:
            result.append(str(value))
    return result


def _write_aligned_ipa_manifest(
    manifest_df: pd.DataFrame, manifest_in: str | Path
) -> Path:
    """Persist segment-level IPA values derived from the scoring manifest."""
    source_path = Path(manifest_in)
    output_path = source_path.with_name(f"{source_path.stem}{ALIGNED_IPA_SUFFIX}")
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for _, row in manifest_df.iterrows():
                payload = {
                    "audio_filepath": str(row["audio_path"]),
                    "can_text": str(row["manifest_can_text"]),
                    "ref_text": str(row["manifest_ref_text"]),
                    "pred_text": str(row["manifest_hyp_text"]),
                }
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")
        temporary.replace(output_path)
    except OSError as exc:
        raise ScoringRepresentationError(
            f"Could not write aligned IPA manifest: {output_path}"
        ) from exc
    return output_path


def prepare_scoring_texts(
    manifest_df: pd.DataFrame,
    *,
    dataset_root: str | Path,
    manifest_in: str | Path,
    logger: logging.Logger,
    scoring_representation: ScoringRepresentation = "auto",
    phonemize: PhonemizeBatch | None = None,
) -> tuple[pd.DataFrame, str]:
    """Resolve one orthographic or IPA scoring view before shared metrics."""
    if scoring_representation not in {
        "auto",
        "orthographic",
        "ipa",
    }:
        raise ScoringRepresentationError(
            f"Unsupported scoring representation: {scoring_representation}"
        )
    profile = _load_run_profile(manifest_in)
    if profile is None:
        if scoring_representation == "ipa":
            raise ScoringRepresentationError(
                "IPA scoring requires standard run metadata so model output units "
                "are known"
            )
        return manifest_df, "orthographic"

    output_units = profile.get("output_units")
    if scoring_representation == "orthographic":
        if output_units != "orthographic":
            raise ScoringRepresentationError(
                "Orthographic scoring cannot consume a native phoneme hypothesis"
            )
        return manifest_df, "orthographic"
    if output_units == "orthographic" and scoring_representation == "auto":
        return manifest_df, "orthographic"
    if output_units not in {"orthographic", "phoneme"}:
        raise ScoringRepresentationError(
            f"Unsupported or missing model output units: {output_units}"
        )

    reference_view = _load_ipa_reference(dataset_root)
    required_columns = {"audio_path", "manifest_hyp_text"}
    missing_columns = sorted(required_columns - set(manifest_df.columns))
    if missing_columns:
        raise ScoringRepresentationError(
            "IPA scoring manifest is missing column(s): "
            + ", ".join(missing_columns)
        )

    out = manifest_df.copy()
    audio_ids = out["audio_path"].astype(str)
    missing = sorted(set(audio_ids) - set(reference_view.rows))
    if missing:
        raise ScoringRepresentationError(
            f"IPA reference view is missing {len(missing)} manifest row(s); "
            f"first: {missing[0]}"
        )
    out["manifest_can_text"] = audio_ids.map(
        lambda item_id: reference_view.rows[item_id][0]
    )
    out["manifest_ref_text"] = audio_ids.map(
        lambda item_id: reference_view.rows[item_id][1]
    )

    if output_units == "phoneme":
        source_inventory = profile.get("output_inventory")
        if (
            profile.get("output_notation") != "ipa"
            or not isinstance(source_inventory, str)
            or not source_inventory.strip()
        ):
            raise ScoringRepresentationError(
                "Phoneme scoring requires output_notation=ipa and output_inventory "
                "in run metadata"
            )
        source_inventory = source_inventory.strip()
        try:
            align_hypothesis = build_ipa_aligner(
                source_inventory, reference_view.inventory
            )
        except IPAInventoryMapError as exc:
            raise ScoringRepresentationError(str(exc)) from exc
        out["manifest_hyp_text"] = out["manifest_hyp_text"].map(align_hypothesis)
        hypothesis_route = f"native IPA {source_inventory} -> {reference_view.inventory}"
    else:
        if reference_view.producer != "africa-g2p":
            raise ScoringRepresentationError(
                "Orthographic-to-IPA hypothesis conversion requires an Africa G2P "
                f"reference view, found: {reference_view.producer}"
            )
        phonemize_batch = phonemize or (
            lambda texts: phonemize_with_africa_g2p(
                texts, language=reference_view.language
            )
        )
        try:
            converted = phonemize_batch(_text_values(out["manifest_hyp_text"]))
        except ReferenceViewError as exc:
            raise ScoringRepresentationError(str(exc)) from exc
        if len(converted) != len(out):
            raise ScoringRepresentationError(
                "G2P changed the number of hypotheses; IPA scoring rows no longer align"
            )
        out["manifest_hyp_text"] = [str(value).strip() for value in converted]
        hypothesis_route = (
            f"orthographic -> {reference_view.producer} "
            f"{reference_view.producer_version} ({reference_view.language})"
        )

    aligned_path = _write_aligned_ipa_manifest(out, manifest_in)
    logger.info(
        "Scoring IPA in canonical inventory %s; hypothesis route: %s",
        reference_view.inventory,
        hypothesis_route,
    )
    logger.info("Wrote aligned IPA scoring manifest: %s", aligned_path)
    return out, "phoneme"
