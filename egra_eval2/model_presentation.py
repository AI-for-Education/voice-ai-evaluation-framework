"""Presentation-only model names keyed by stable inference-setup identities."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

MODEL_PRESENTATION_SCHEMA_VERSION = 2
MODEL_PRESENTATION_REGISTRY_PATH = Path(__file__).with_name(
    "model_presentation.json"
)
_ARCHITECTURE_EVIDENCE_STATUSES = {
    "artifact_verified",
    "owner_documented",
    "not_available",
}
MODEL_PRESENTATION_COLUMNS = [
    "inference_setup_id",
    "model_group",
    "model_name",
    "model_variant",
    "official_model_url",
    "official_model_url_note",
    "architecture",
    "architecture_evidence_status",
    "architecture_evidence_source",
    "decoder",
    "execution_target",
    "leaderboard_status",
    "presentation_name",
]


def load_model_presentation_registry(
    path: str | Path = MODEL_PRESENTATION_REGISTRY_PATH,
) -> dict[str, Any]:
    """Load and validate the centralized stable-ID-to-presentation mapping."""
    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MODEL_PRESENTATION_SCHEMA_VERSION:
        raise ValueError(
            "Model presentation registry schema_version must be "
            f"{MODEL_PRESENTATION_SCHEMA_VERSION}: {registry_path}"
        )
    naming_format = payload.get("naming_format")
    if not isinstance(naming_format, str) or not naming_format.strip():
        raise ValueError(
            f"Model presentation registry has no naming format: {registry_path}"
        )
    inference_setups = payload.get("inference_setups")
    if not isinstance(inference_setups, dict):
        raise ValueError(
            f"Model presentation registry has no inference setups: {registry_path}"
        )
    for inference_setup_id, entry in inference_setups.items():
        if not isinstance(inference_setup_id, str) or not inference_setup_id.strip():
            raise ValueError(
                "Model presentation registry contains an empty inference-setup ID"
            )
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid presentation entry for {inference_setup_id}")
        for field in ("model_group", "model_name"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid {field} for {inference_setup_id}")
        if not isinstance(entry.get("variant"), str):
            raise ValueError(f"Invalid variant for {inference_setup_id}")
        for field in ("official_model_url", "official_model_url_note"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid {field} for {inference_setup_id}")
        architecture = entry.get("architecture")
        if not isinstance(architecture, dict):
            raise ValueError(f"Invalid architecture for {inference_setup_id}")
        for field in ("label", "evidence_status", "source"):
            value = architecture.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Invalid architecture.{field} for {inference_setup_id}"
                )
        if architecture["evidence_status"] not in _ARCHITECTURE_EVIDENCE_STATUSES:
            raise ValueError(
                "Invalid architecture evidence status for "
                f"{inference_setup_id}: "
                f"{architecture['evidence_status']}"
            )
    return payload


@lru_cache(maxsize=1)
def default_model_presentation_registry() -> dict[str, Any]:
    return load_model_presentation_registry()


def presentation_for_inference_setup(inference_setup_id: str) -> dict[str, Any]:
    inference_setups = default_model_presentation_registry()["inference_setups"]
    entry = inference_setups.get(inference_setup_id)
    if isinstance(entry, dict):
        return {
            **entry,
            "mapping_status": "registered",
        }
    return {
        "model_group": "Uncatalogued",
        "model_name": inference_setup_id,
        "variant": "Unspecified",
        "official_model_url": "",
        "official_model_url_note": "No official model page is registered",
        "architecture": {
            "label": "not available",
            "evidence_status": "not_available",
            "source": (
                "No presentation registry entry exists for this "
                "inference-setup ID"
            ),
        },
        "mapping_status": "uncatalogued",
    }


def structured_model_label(presentation: dict[str, Any], decoder: str) -> str:
    parts = [presentation["model_group"], presentation["model_name"]]
    variant = str(presentation.get("variant") or "").strip()
    if variant:
        parts.append(variant)
    return f"{' · '.join(parts)} ({decoder})"


def build_presentation_rows(
    evaluated_by_id: Mapping[str, Mapping[str, Any]],
    profile_context_by_id: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    """Build the auditable inference-setup presentation table.

    Completed-run values take precedence because they describe what actually ran;
    profile-derived values fill the rows for registered models not yet evaluated.
    """
    registry = default_model_presentation_registry()
    inference_setup_ids = sorted(
        set(registry["inference_setups"])
        | set(profile_context_by_id)
        | set(evaluated_by_id)
    )
    rows: list[dict[str, str]] = []
    for inference_setup_id in inference_setup_ids:
        evaluated = evaluated_by_id.get(inference_setup_id)
        if evaluated is not None:
            decoder = str(evaluated.get("decoding") or "not recorded")
            execution_target = str(
                evaluated.get("execution_target") or "execution-target-not-recorded"
            )
            leaderboard_status = "evaluated"
        else:
            profile_context = profile_context_by_id.get(inference_setup_id, {})
            decoder = str(profile_context.get("decoder") or "not recorded")
            execution_target = str(
                profile_context.get("execution_target")
                or "execution-target-not-recorded"
            )
            leaderboard_status = "profile_only"

        presentation = presentation_for_inference_setup(inference_setup_id)
        architecture = presentation["architecture"]
        rows.append(
            {
                "inference_setup_id": inference_setup_id,
                "model_group": presentation["model_group"],
                "model_name": presentation["model_name"],
                "model_variant": presentation["variant"],
                "official_model_url": presentation["official_model_url"],
                "official_model_url_note": presentation["official_model_url_note"],
                "architecture": architecture["label"],
                "architecture_evidence_status": architecture["evidence_status"],
                "architecture_evidence_source": architecture["source"],
                "decoder": decoder,
                "execution_target": execution_target,
                "leaderboard_status": leaderboard_status,
                "presentation_name": structured_model_label(
                    presentation, decoder
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            row["model_group"],
            row["model_name"],
            row["model_variant"],
            row["decoder"],
        ),
    )
