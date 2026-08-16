"""Qwen2.5-Omni text-only audio adapter for large-memory GPUs.

This adapter is intentionally hardware-gated. The downloaded 7B BF16 checkpoint
is not runnable on the current 16 GiB RTX 5080 Laptop GPU; initialization stops
before loading weights and reports the profile's large-GPU requirement.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

from inference.common import load_audio_and_resample
from inference.contracts import TranscriptionResult
from inference.multimodal.adapters.common import (
    cuda_memory_gib,
    move_inputs,
    serializable_device_map,
    split_audio,
    torch_dtype_from_profile,
)
from inference.profile import ModelProfile, ProfileError


class QwenOmniAudioBackend:
    """Qwen Omni audio understanding with speech generation disabled."""

    def __init__(
        self,
        profile: ModelProfile,
        model_path: str | Path,
        *,
        device: torch.device | None = None,
    ) -> None:
        if (
            profile.framework != "multimodal"
            or profile.adapter != "qwen_omni_audio"
        ):
            raise ProfileError(
                "QwenOmniAudioBackend requires a multimodal/qwen_omni_audio profile"
            )
        if profile.prompt is None or profile.audio is None or profile.hardware is None:
            raise ProfileError(
                "The Qwen profile requires prompt, audio, and hardware settings"
            )
        if profile.hardware.output_mode != "text_only":
            raise ProfileError("Qwen Omni is restricted to text_only output")

        self.profile = profile
        self.model_path = Path(model_path)
        self.model = None
        self.processor = None
        if not self.model_path.is_dir():
            raise ProfileError(f"Multimodal model directory not found: {self.model_path}")

        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        if self.device.type != "cuda":
            raise RuntimeError("Qwen2.5-Omni-7B requires a CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Qwen2.5-Omni-7B requires CUDA bfloat16 support")

        total_gpu_gib = cuda_memory_gib(self.device)
        minimum_gpu_gib = profile.hardware.minimum_gpu_memory_gib
        if total_gpu_gib < minimum_gpu_gib:
            raise RuntimeError(
                "Qwen2.5-Omni-7B is not runnable on this GPU in the tracked BF16 "
                "text-only configuration: "
                f"{total_gpu_gib:.1f} GiB detected, at least "
                f"{minimum_gpu_gib:.1f} GiB required. The current 16 GiB host is "
                "intentionally blocked before model loading."
            )

        common_kwargs = {
            "local_files_only": profile.loader.local_files_only,
            "trust_remote_code": profile.loader.trust_remote_code,
        }
        self._model_dtype = torch_dtype_from_profile(profile.loader.torch_dtype)
        attention_implementation = (
            profile.loader.attention_implementation or "sdpa"
        )
        try:
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                **common_kwargs,
            )
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                str(self.model_path),
                **common_kwargs,
                torch_dtype=self._model_dtype,
                device_map="auto",
                attn_implementation=attention_implementation,
            )
            self.model.disable_talker()
            self.model.eval()
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Failed to load local Qwen2.5-Omni model from {self.model_path}: {exc}"
            ) from exc

        feature_extractor = getattr(self.processor, "feature_extractor", None)
        self.sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
        if self.sampling_rate <= 0:
            self.close()
            raise RuntimeError(
                f"Qwen processor reported an invalid sampling rate: {self.sampling_rate}"
            )

        self._rendered_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a speech transcription system. "
                                "Return text only and never generate speech."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": "in-memory-audio"},
                        {"type": "text", "text": profile.prompt},
                    ],
                },
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        self._generation_kwargs = dict(profile.decoding.generation_kwargs)
        self._metadata: dict[str, Any] = {
            "framework": "multimodal",
            "adapter": "qwen_omni_audio",
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "device": str(self.device),
            "device_map": serializable_device_map(self.model),
            "sampling_rate": self.sampling_rate,
            "requested_torch_dtype": profile.loader.torch_dtype,
            "effective_torch_dtype": str(self._model_dtype).removeprefix("torch."),
            "attention_implementation": attention_implementation,
            "hardware": {
                "detected_gpu_memory_gib": round(total_gpu_gib, 2),
                "minimum_gpu_memory_gib": minimum_gpu_gib,
                "memory_strategy": profile.hardware.memory_strategy,
                "current_16gib_host_supported": False,
            },
            "output": {
                "mode": "text_only",
                "talker_disabled": True,
                "return_audio": False,
            },
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

    def _decode_chunk(self, audio: np.ndarray) -> str:
        encoded = self.processor(
            text=self._rendered_prompt,
            audio=[np.asarray(audio, dtype=np.float32)],
            return_tensors="pt",
            padding=True,
        )
        model_inputs = move_inputs(
            encoded,
            device=self.device,
            floating_dtype=self._model_dtype,
        )
        with torch.inference_mode():
            text_ids = self.model.generate(
                **model_inputs,
                return_audio=False,
                **self._generation_kwargs,
            )
        if isinstance(text_ids, tuple):
            text_ids = text_ids[0]

        input_ids = model_inputs["input_ids"]
        input_length = int(input_ids.shape[-1])
        if (
            text_ids.ndim == 2
            and text_ids.shape[-1] >= input_length
            and torch.equal(text_ids[:, :input_length], input_ids)
        ):
            text_ids = text_ids[:, input_length:]
        decoded = self.processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return str(decoded[0]).strip()

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

            chunks = split_audio(
                audio,
                sampling_rate=self.sampling_rate,
                profile=self.profile,
            )
            self._metadata["chunks_generated"] += len(chunks)
            self._metadata["long_audio_files"] += int(len(chunks) > 1)
            try:
                predictions = [self._decode_chunk(chunk) for chunk in chunks]
                prediction = " ".join(text for text in predictions if text).strip()
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
