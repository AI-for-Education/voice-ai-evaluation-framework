"""Prepared Fairseq2 cache contract for offline Omnilingual inference."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from inference.profile import InferenceProfile, ProfileError, resolve_model_path


PREPARED_MARKER = ".prepared.json"
TOKENIZER_URL = (
    "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model"
)
MODEL_SOURCES = {
    "omniASR_CTC_3B_v2": (
        "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-3B-v2.pt"
    ),
    "omniASR_LLM_3B_v2": (
        "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B-v2.pt"
    ),
}


def configure_model_cache(
    profile: InferenceProfile,
    *,
    create: bool,
) -> Path:
    """Select an isolated Fairseq2 cache for one model card."""
    cache_path = resolve_model_path(profile, require_exists=False)
    if create:
        cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["FAIRSEQ2_CACHE_DIR"] = str(cache_path)
    return cache_path


def _version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def write_prepared_marker(profile: InferenceProfile, cache_path: Path) -> Path:
    """Publish a marker only after the owner pipeline loads the cached assets."""
    if profile.artifact not in MODEL_SOURCES:
        raise ProfileError(f"Unsupported Omnilingual asset card: {profile.artifact}")
    marker = cache_path / PREPARED_MARKER
    temporary = cache_path / f"{PREPARED_MARKER}.tmp"
    files = [
        path
        for path in cache_path.rglob("*")
        if path.is_file() and path not in {marker, temporary}
    ]
    payload = {
        "schema_version": 1,
        "model_card": profile.artifact,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "owner_assets": {
            "checkpoint": MODEL_SOURCES[profile.artifact],
            "tokenizer": TOKENIZER_URL,
        },
        "cache_inventory": {
            "file_count": len(files),
            "total_size_bytes": sum(path.stat().st_size for path in files),
        },
        "packages": {
            "omnilingual_asr": _version("omnilingual-asr"),
            "fairseq2": _version("fairseq2"),
            "torch": _version("torch"),
            "torchaudio": _version("torchaudio"),
        },
    }
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)
    return marker


def prepared_model_identity(profile: InferenceProfile, cache_path: Path) -> dict[str, Any]:
    """Require and describe a successfully prepared, model-specific cache."""
    marker = cache_path / PREPARED_MARKER
    if not marker.is_file():
        raise ProfileError(
            "Omnilingual model cache is not prepared for "
            f"'{profile.artifact}': {marker}. Run run_omnilingual_prepare.sh first."
        )
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Invalid Omnilingual prepared-cache marker: {marker}") from exc
    if payload.get("model_card") != profile.artifact:
        raise ProfileError(
            "Omnilingual prepared-cache marker selects a different model card"
        )
    return {
        "kind": "fairseq2_asset_card",
        "artifact": profile.artifact,
        "path": str(cache_path),
        "exists": True,
        "prepared_cache": payload,
    }
