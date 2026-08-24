"""Presentation-only model names, separate from stable inference identities."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from egra_eval2.leaderboard_context import decoder_slug


MODEL_PRESENTATION_SCHEMA_VERSION = 1
MODEL_PRESENTATION_REGISTRY_PATH = Path(__file__).with_name(
    "model_presentation.json"
)
_ARCHITECTURE_EVIDENCE_STATUSES = {
    "artifact_verified",
    "owner_documented",
    "not_available",
}
MODEL_NAME_MAPPING_COLUMNS = [
    "previous_model_id",
    "previous_presentation_name",
    "model_group",
    "model_name",
    "model_variant",
    "official_model_url",
    "official_model_url_note",
    "architecture",
    "architecture_evidence_status",
    "architecture_evidence_source",
    "decoder",
    "platform",
    "leaderboard_status",
    "new_presentation_name",
]


def load_model_presentation_registry(
    path: str | Path = MODEL_PRESENTATION_REGISTRY_PATH,
) -> dict[str, Any]:
    """Load and validate the centralized stable-ID-to-presentation mapping."""
    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MODEL_PRESENTATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported model presentation registry: {registry_path}")
    naming_format = payload.get("naming_format")
    if not isinstance(naming_format, str) or not naming_format.strip():
        raise ValueError(
            f"Model presentation registry has no naming format: {registry_path}"
        )
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"Model presentation registry has no models: {registry_path}")
    for model_id, entry in models.items():
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Model presentation registry contains an empty model ID")
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid presentation entry for {model_id}")
        for field in ("model_group", "model_name"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid {field} for {model_id}")
        if not isinstance(entry.get("variant"), str):
            raise ValueError(f"Invalid variant for {model_id}")
        for field in ("official_model_url", "official_model_url_note"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid {field} for {model_id}")
        architecture = entry.get("architecture")
        if not isinstance(architecture, dict):
            raise ValueError(f"Invalid architecture for {model_id}")
        for field in ("label", "evidence_status", "source"):
            value = architecture.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid architecture.{field} for {model_id}")
        if architecture["evidence_status"] not in _ARCHITECTURE_EVIDENCE_STATUSES:
            raise ValueError(
                f"Invalid architecture evidence status for {model_id}: "
                f"{architecture['evidence_status']}"
            )
    return payload


@lru_cache(maxsize=1)
def default_model_presentation_registry() -> dict[str, Any]:
    return load_model_presentation_registry()


def presentation_for_model(model_id: str) -> dict[str, Any]:
    models = default_model_presentation_registry()["models"]
    entry = models.get(model_id)
    if isinstance(entry, dict):
        return {
            **entry,
            "mapping_status": "registered",
        }
    return {
        "model_group": "Uncatalogued",
        "model_name": model_id,
        "variant": "Unspecified",
        "official_model_url": "",
        "official_model_url_note": "No official model page is registered",
        "architecture": {
            "label": "not available",
            "evidence_status": "not_available",
            "source": "No presentation registry entry exists for this model ID",
        },
        "mapping_status": "fallback",
    }


def structured_model_label(presentation: dict[str, Any], decoder: str) -> str:
    parts = [presentation["model_group"], presentation["model_name"]]
    variant = str(presentation.get("variant") or "").strip()
    if variant:
        parts.append(variant)
    return f"{' · '.join(parts)} ({decoder})"


def build_name_mapping_rows(
    evaluated_by_id: Mapping[str, Mapping[str, Any]],
    profile_context_by_id: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    """Build the auditable old-to-new presentation-name mapping.

    Completed-run values take precedence because they describe what actually ran;
    profile-derived values fill the rows for registered models not yet evaluated.
    """
    registry = default_model_presentation_registry()
    model_ids = sorted(
        set(registry["models"]) | set(profile_context_by_id) | set(evaluated_by_id)
    )
    rows: list[dict[str, str]] = []
    for model_id in model_ids:
        evaluated = evaluated_by_id.get(model_id)
        if evaluated is not None:
            decoder = str(evaluated.get("decoding") or "not recorded")
            platform = str(evaluated.get("platform") or "platform-not-recorded")
            leaderboard_status = "evaluated"
        else:
            profile_context = profile_context_by_id.get(model_id, {})
            decoder = str(profile_context.get("decoder") or "not recorded")
            platform = str(
                profile_context.get("platform") or "platform-not-recorded"
            )
            leaderboard_status = "profile_only"

        presentation = presentation_for_model(model_id)
        architecture = presentation["architecture"]
        rows.append(
            {
                "previous_model_id": model_id,
                "previous_presentation_name": (
                    f"{platform} · {decoder_slug(decoder)} · {model_id}"
                ),
                "model_group": presentation["model_group"],
                "model_name": presentation["model_name"],
                "model_variant": presentation["variant"],
                "official_model_url": presentation["official_model_url"],
                "official_model_url_note": presentation["official_model_url_note"],
                "architecture": architecture["label"],
                "architecture_evidence_status": architecture["evidence_status"],
                "architecture_evidence_source": architecture["source"],
                "decoder": decoder,
                "platform": platform,
                "leaderboard_status": leaderboard_status,
                "new_presentation_name": structured_model_label(
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
