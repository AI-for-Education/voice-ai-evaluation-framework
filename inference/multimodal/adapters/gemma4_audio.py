"""Gemma 4 audio-to-text generation using a fully local checkpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from inference.common import load_audio_and_resample
from inference.contracts import TranscriptionResult
from inference.multimodal.adapters.common import torch_dtype_from_profile
from inference.profile import ModelProfile, ProfileError


def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Gemma 4 E2B inference requires a CUDA GPU for the frozen bfloat16 profile"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Gemma 4 E2B inference requires a CUDA GPU with bfloat16 support"
        )
    return torch.device("cuda:0")


def _model_dtype(model: Any) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except (AttributeError, StopIteration):
        return torch.bfloat16


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "".join(parts).strip()


class Gemma4AudioBackend:
    """Prompt-controlled Gemma 4 audio transcription adapter.

    Files longer than the profile's maximum duration are split into consecutive,
    non-overlapping chunks. Each chunk is generated independently and the text is
    joined in source order so the shared output runner still receives one result
    row per input file.
    """

    def __init__(
        self,
        profile: ModelProfile,
        model_path: str | Path,
        *,
        device: torch.device | None = None,
    ) -> None:
        if profile.framework != "multimodal" or profile.adapter != "gemma4_audio":
            raise ProfileError(
                "Gemma4AudioBackend requires a multimodal/gemma4_audio profile"
            )
        if profile.prompt is None or profile.audio is None:
            raise ProfileError("The Gemma 4 audio profile requires prompt and audio settings")
        if profile.decoding.strategy != "generate":
            raise ProfileError("The Gemma 4 audio adapter requires generate decoding")

        self.profile = profile
        self.model_path = Path(model_path)
        self.model = None
        self.processor = None
        if not self.model_path.is_dir():
            raise ProfileError(f"Multimodal model directory not found: {self.model_path}")

        self.device = device or _select_device()
        if self.device.type != "cuda":
            raise RuntimeError(
                "Gemma 4 E2B inference requires CUDA for the frozen bfloat16 profile"
            )

        common_kwargs = {
            "local_files_only": profile.loader.local_files_only,
            "trust_remote_code": profile.loader.trust_remote_code,
        }
        model_dtype = torch_dtype_from_profile(profile.loader.torch_dtype)
        try:
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                **common_kwargs,
            )
            self.model = AutoModelForMultimodalLM.from_pretrained(
                str(self.model_path),
                **common_kwargs,
                dtype=model_dtype,
                device_map={"": str(self.device)},
            )
            self.model.eval()
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Failed to load local Gemma 4 model from {self.model_path}: {exc}"
            ) from exc

        feature_extractor = getattr(self.processor, "feature_extractor", None)
        self.sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
        if self.sampling_rate <= 0:
            self.close()
            raise RuntimeError(
                f"Gemma processor reported an invalid sampling rate: {self.sampling_rate}"
            )

        self._rendered_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": profile.prompt},
                        {"type": "audio"},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self._generation_kwargs = dict(profile.decoding.generation_kwargs)
        self._model_class = type(self.model).__name__
        self._processor_class = type(self.processor).__name__
        self._metadata: dict[str, Any] = {
            "framework": "multimodal",
            "adapter": "gemma4_audio",
            "model_class": self._model_class,
            "processor_class": self._processor_class,
            "device": str(self.device),
            "sampling_rate": self.sampling_rate,
            "requested_torch_dtype": profile.loader.torch_dtype,
            "effective_torch_dtype": str(_model_dtype(self.model)).removeprefix("torch."),
            "generation_kwargs": dict(self._generation_kwargs),
            "prompt": profile.prompt,
            "long_audio": {
                "strategy": profile.audio.long_audio_strategy,
                "maximum_seconds": profile.audio.maximum_seconds,
                "chunk_seconds": profile.audio.chunk_seconds,
                "overlap_seconds": profile.audio.overlap_seconds,
            },
            "files_seen": 0,
            "long_audio_files": 0,
            "chunks_generated": 0,
        }

    def _move_inputs(self, encoded: Any) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for name, value in encoded.items():
            if hasattr(value, "to"):
                moved[name] = value.to(device=self.device)
            else:
                moved[name] = value
        return moved

    def _decode_chunk(self, audio: np.ndarray) -> str:
        encoded = self.processor(
            text=self._rendered_prompt,
            audio=[np.asarray(audio, dtype=np.float32)],
            return_tensors="pt",
        )
        model_inputs = self._move_inputs(encoded)
        input_length = int(model_inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                **self._generation_kwargs,
            )
        generated = generated_ids[0][input_length:]
        response = self.processor.decode(generated, skip_special_tokens=False)
        parsed = self.processor.parse_response(
            response,
            prefix=model_inputs["input_ids"],
        )
        if isinstance(parsed, dict):
            text = _text_from_content(parsed.get("content"))
            if text:
                return text

        # Keep an intelligible fallback if a future processor changes only the
        # response wrapper while preserving its token decoding behavior.
        return str(self.processor.decode(generated, skip_special_tokens=True)).strip()

    def _chunks(self, audio: np.ndarray) -> list[np.ndarray]:
        audio_config = self.profile.audio
        if audio_config is None:  # guarded by __init__; narrows the static type
            raise RuntimeError("Gemma audio configuration is unavailable")
        maximum_samples = max(1, int(audio_config.maximum_seconds * self.sampling_rate))
        if len(audio) <= maximum_samples:
            return [audio]
        chunk_samples = max(1, int(audio_config.chunk_seconds * self.sampling_rate))
        return [
            audio[start : start + chunk_samples]
            for start in range(0, len(audio), chunk_samples)
        ]

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        rows: list[TranscriptionResult] = []
        for raw_path in audio_paths:
            audio_path = str(raw_path)
            self._metadata["files_seen"] += 1
            try:
                audio, _, duration, _ = load_audio_and_resample(
                    audio_path,
                    self.sampling_rate,
                )
            except Exception as exc:
                rows.append(
                    TranscriptionResult(
                        audio_filepath=audio_path,
                        duration=0.0,
                        pred_text="",
                        error=f"failed_to_read_audio: {exc}",
                    )
                )
                continue

            chunks = self._chunks(audio)
            self._metadata["chunks_generated"] += len(chunks)
            self._metadata["long_audio_files"] += int(len(chunks) > 1)
            try:
                chunk_texts = [self._decode_chunk(chunk) for chunk in chunks]
                prediction = " ".join(text for text in chunk_texts if text).strip()
            except Exception as exc:
                rows.append(
                    TranscriptionResult(
                        audio_filepath=audio_path,
                        duration=duration,
                        pred_text="",
                        error=f"inference_failed: {exc}",
                    )
                )
                continue

            rows.append(
                TranscriptionResult(
                    audio_filepath=audio_path,
                    duration=duration,
                    pred_text=prediction,
                )
            )
        return rows

    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def close(self) -> None:
        self.model = None
        self.processor = None
        if getattr(self, "device", None) is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
