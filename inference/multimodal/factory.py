"""Lazy adapter selection for dependency-isolated multimodal images."""

from __future__ import annotations

from pathlib import Path

from inference.contracts import ASRBackend
from inference.profile import InferenceProfile, ProfileError


def create_backend(profile: InferenceProfile, model_path: str | Path) -> ASRBackend:
    """Import only the adapter supported by the selected execution image."""
    try:
        if profile.adapter == "gemma4_audio":
            from inference.multimodal.adapters.gemma4_audio import Gemma4AudioBackend

            return Gemma4AudioBackend(profile, model_path)
        if profile.adapter == "phi4_audio":
            from inference.multimodal.adapters.phi4_audio import Phi4AudioBackend

            return Phi4AudioBackend(profile, model_path)
        if profile.adapter == "qwen_omni_audio":
            from inference.multimodal.adapters.qwen_omni_audio import QwenOmniAudioBackend

            return QwenOmniAudioBackend(profile, model_path)
    except ImportError as exc:
        raise RuntimeError(
            f"Adapter '{profile.adapter}' is unavailable in this image; "
            "use its model-family launcher and dedicated Docker image"
        ) from exc
    raise ProfileError(f"Unsupported multimodal adapter: {profile.adapter}")
