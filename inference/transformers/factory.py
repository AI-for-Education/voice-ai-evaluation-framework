"""Factory for profile-selected Transformers inference adapters."""

from __future__ import annotations

from pathlib import Path

from inference.contracts import ASRBackend
from inference.profile import ModelProfile, ProfileError


def create_backend(profile: ModelProfile, model_path: str | Path) -> ASRBackend:
    """Construct the adapter selected by a validated Transformers profile."""
    if profile.framework != "transformers":
        raise ProfileError(
            f"Transformers backend cannot load framework '{profile.framework}'"
        )

    if profile.adapter == "ctc":
        try:
            from inference.transformers.adapters.ctc import TransformersCTCBackend
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Transformers CTC inference requires torch, numpy, and transformers"
            ) from exc

        return TransformersCTCBackend(profile, model_path)
    if profile.adapter == "speech_seq2seq":
        try:
            from inference.transformers.adapters.speech_seq2seq import (
                TransformersSpeechSeq2SeqBackend,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Transformers speech-seq2seq inference requires torch, numpy, "
                "and transformers"
            ) from exc

        return TransformersSpeechSeq2SeqBackend(profile, model_path)
    raise ProfileError(f"Unsupported Transformers adapter: {profile.adapter}")
