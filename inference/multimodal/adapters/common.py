"""Shared helpers for prompt-driven multimodal audio adapters."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from inference.profile import InferenceProfile


def torch_dtype_from_profile(value: str) -> torch.dtype:
    """Resolve a validated profile dtype for multimodal model loading."""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(
            "Multimodal adapters require an explicit floating-point profile dtype"
        ) from exc


def split_audio(
    audio: np.ndarray,
    *,
    sampling_rate: int,
    profile: InferenceProfile,
) -> list[np.ndarray]:
    """Split audio according to the profile while preserving source order."""
    if profile.audio is None:
        raise RuntimeError("Multimodal audio configuration is unavailable")
    maximum_samples = max(1, int(profile.audio.maximum_seconds * sampling_rate))
    if len(audio) <= maximum_samples:
        return [audio]
    chunk_samples = max(1, int(profile.audio.chunk_seconds * sampling_rate))
    return [
        audio[start : start + chunk_samples]
        for start in range(0, len(audio), chunk_samples)
    ]


def move_inputs(
    encoded: Any,
    *,
    device: torch.device,
    floating_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Move a processor batch without casting integer token tensors."""
    moved: dict[str, Any] = {}
    for name, value in encoded.items():
        if not hasattr(value, "to"):
            moved[name] = value
        elif floating_dtype is not None and torch.is_floating_point(value):
            moved[name] = value.to(device=device, dtype=floating_dtype)
        else:
            moved[name] = value.to(device=device)
    return moved


def cuda_memory_gib(device: torch.device) -> float:
    """Return physical memory for one CUDA device in GiB."""
    if device.type != "cuda":
        return 0.0
    properties = torch.cuda.get_device_properties(device)
    return float(properties.total_memory / (1024**3))


def serializable_device_map(model: Any) -> dict[str, str]:
    """Normalize Accelerate's device map for run metadata."""
    raw = getattr(model, "hf_device_map", None)
    if not isinstance(raw, dict):
        return {}
    return {str(name): str(device) for name, device in raw.items()}
