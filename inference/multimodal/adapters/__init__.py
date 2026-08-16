"""Lazy model-family exports for dependency-isolated runtime images."""

from typing import Any

__all__ = ["Gemma4AudioBackend", "Phi4AudioBackend", "QwenOmniAudioBackend"]


def __getattr__(name: str) -> Any:
    if name == "Gemma4AudioBackend":
        from inference.multimodal.adapters.gemma4_audio import Gemma4AudioBackend

        return Gemma4AudioBackend
    if name == "Phi4AudioBackend":
        from inference.multimodal.adapters.phi4_audio import Phi4AudioBackend

        return Phi4AudioBackend
    if name == "QwenOmniAudioBackend":
        from inference.multimodal.adapters.qwen_omni_audio import QwenOmniAudioBackend

        return QwenOmniAudioBackend
    raise AttributeError(name)
