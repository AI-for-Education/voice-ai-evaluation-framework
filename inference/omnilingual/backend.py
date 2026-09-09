"""Adapter around Meta's reference Omnilingual ASR inference pipeline."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch

from inference.common import load_audio_and_resample
from inference.contracts import TranscriptionResult
from inference.profile import InferenceProfile, ProfileError


PipelineFactory = Callable[..., Any]


def _owner_components() -> tuple[Callable[..., Any], Callable[..., Any], PipelineFactory]:
    from fairseq2.data.tokenizers import load_tokenizer
    from fairseq2.models import load_model
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    return load_model, load_tokenizer, ASRInferencePipeline


def load_owner_pipeline(
    profile: InferenceProfile,
    device: torch.device,
) -> Any:
    """Load owner assets with Fairseq2 mmap, then retain the owner pipeline."""
    if profile.artifact is None:
        raise ProfileError("Omnilingual profile is missing its model artifact")
    load_model, load_tokenizer, pipeline_factory = _owner_components()
    model = load_model(
        profile.artifact,
        device=device,
        dtype=torch.bfloat16,
        mmap=True,
    )
    tokenizer = load_tokenizer(profile.artifact)
    return pipeline_factory(
        model_card=None,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.bfloat16,
    )


class OmnilingualBackend:
    """Run one packaged CTC or language-conditioned LLM asset card."""

    sampling_rate = 16000

    def __init__(
        self,
        profile: InferenceProfile,
        *,
        batch_size: int = 1,
        device: torch.device | None = None,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        if profile.inference_library != "omnilingual":
            raise ProfileError("OmnilingualBackend requires an omnilingual profile")
        if profile.audio is None or profile.loader is None or profile.artifact is None:
            raise ProfileError("Omnilingual profile is missing its local audio contract")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("Omnilingual 3B inference requires an available CUDA GPU")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Omnilingual 3B inference requires CUDA BF16 support")
            device = torch.device("cuda:0")

        self.profile = profile
        self.device = device
        self.batch_size = batch_size
        if pipeline_factory is None:
            self.pipeline = load_owner_pipeline(profile, device)
            checkpoint_mmap = True
        else:
            self.pipeline = pipeline_factory(
                model_card=profile.artifact,
                device=device,
                dtype=torch.bfloat16,
            )
            checkpoint_mmap = False
        self._checkpoint_mmap = checkpoint_mmap
        self._statistics = {
            "files_seen": 0,
            "chunks_generated": 0,
            "long_audio_files": 0,
            "resampled_files": 0,
        }

    def _split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        audio = np.asarray(audio, dtype=np.float32)
        maximum_samples = max(
            1, int(self.profile.audio.maximum_seconds * self.sampling_rate)
        )
        if len(audio) <= maximum_samples:
            return [audio]
        chunk_samples = max(
            1, int(self.profile.audio.chunk_seconds * self.sampling_rate)
        )
        return [
            audio[start : start + chunk_samples]
            for start in range(0, len(audio), chunk_samples)
        ]

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        rows: list[TranscriptionResult] = []
        for raw_path in audio_paths:
            audio_path = str(raw_path)
            self._statistics["files_seen"] += 1
            try:
                audio, _, duration, was_resampled = load_audio_and_resample(
                    audio_path, self.sampling_rate
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

            chunks = self._split_audio(audio)
            self._statistics["chunks_generated"] += len(chunks)
            self._statistics["long_audio_files"] += int(len(chunks) > 1)
            self._statistics["resampled_files"] += int(was_resampled)
            decoded_audio = [
                {"waveform": chunk, "sample_rate": self.sampling_rate}
                for chunk in chunks
            ]
            language = (
                [self.profile.language] * len(chunks)
                if self.profile.adapter == "llm"
                else None
            )
            try:
                predictions = self.pipeline.transcribe(
                    decoded_audio,
                    lang=language,
                    batch_size=self.batch_size,
                )
                if len(predictions) != len(chunks):
                    raise RuntimeError(
                        "owner pipeline returned an unexpected number of hypotheses"
                    )
                prediction = " ".join(
                    str(text).strip() for text in predictions if str(text).strip()
                ).strip()
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
        return {
            "inference_library": "omnilingual_asr",
            "adapter": self.profile.adapter,
            "model_card": self.profile.artifact,
            "device": str(self.device),
            "requested_torch_dtype": self.profile.loader.torch_dtype,
            "effective_torch_dtype": "bfloat16",
            "sampling_rate": self.sampling_rate,
            "checkpoint_loading": {"memory_mapped": self._checkpoint_mmap},
            "language_conditioning": {
                "enabled": self.profile.adapter == "llm",
                "language": (
                    self.profile.language if self.profile.adapter == "llm" else None
                ),
            },
            "decoding": {
                "strategy": self.profile.decoding.strategy,
                "implementation": "owner_reference_pipeline_default",
            },
            "statistics": dict(self._statistics),
        }

    def close(self) -> None:
        self.pipeline = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
