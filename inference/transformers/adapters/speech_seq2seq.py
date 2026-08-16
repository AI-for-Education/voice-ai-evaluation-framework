"""Autoregressive speech-to-text inference for local Transformers checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from inference.common import load_audio_and_resample
from inference.contracts import TranscriptionResult
from inference.profile import HallucinationGuardConfig, ModelProfile, ProfileError
from inference.transformers.adapters.ctc import (
    _loader_dtype,
    _model_dtype,
    _move_model_inputs,
    _select_device,
)


def _repeated_phrase_boundary(
    words: list[str],
    config: HallucinationGuardConfig,
) -> int | None:
    """Return the end of the first phrase when it repeats consecutively."""
    repetitions = config.repeated_phrase_repetitions
    for start in range(len(words)):
        for size in range(
            config.repeated_phrase_min_words,
            config.repeated_phrase_max_words + 1,
        ):
            end = start + size * repetitions
            if end > len(words):
                continue
            phrase = words[start : start + size]
            if all(
                words[start + repeat * size : start + (repeat + 1) * size]
                == phrase
                for repeat in range(1, repetitions)
            ):
                return start + size
    return None


def apply_hallucination_guard(
    text: str,
    *,
    duration: float,
    config: HallucinationGuardConfig,
) -> tuple[str, bool]:
    """Apply the profile's inference-time output safety limits."""
    words = text.split()
    limit = min(
        len(words),
        max(1, int(np.ceil(duration * config.max_words_per_second))),
    )
    repetition_boundary = _repeated_phrase_boundary(words, config)
    if repetition_boundary is not None:
        limit = min(limit, repetition_boundary)
    guarded = " ".join(words[:limit])
    return guarded, guarded != text.strip()


class TransformersSpeechSeq2SeqBackend:
    """Offline adapter for Whisper-compatible ``model.generate`` inference."""

    def __init__(
        self,
        profile: ModelProfile,
        model_path: str | Path,
        *,
        device: torch.device | None = None,
    ) -> None:
        if profile.framework != "transformers" or profile.adapter != "speech_seq2seq":
            raise ProfileError(
                "TransformersSpeechSeq2SeqBackend requires a "
                "transformers/speech_seq2seq profile"
            )
        if profile.decoding.strategy != "generate":
            raise ProfileError("The speech_seq2seq adapter requires generate decoding")
        if profile.loader.processor_mode != "auto":
            raise ProfileError(
                "The speech_seq2seq adapter requires loader.processor_mode: auto"
            )

        self.profile = profile
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise ProfileError(f"Transformers model directory not found: {self.model_path}")

        self.device = device or _select_device()
        dtype_argument, self._dtype_fallback = _loader_dtype(
            profile.loader.torch_dtype,
            self.device,
        )
        common_kwargs = {
            "local_files_only": profile.loader.local_files_only,
            "trust_remote_code": profile.loader.trust_remote_code,
        }

        try:
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                **common_kwargs,
            )
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(self.model_path),
                **common_kwargs,
                torch_dtype=dtype_argument,
            )
            self.model.to(self.device).eval()
        except Exception as exc:
            self.close()
            raise RuntimeError(
                "Failed to load local Transformers speech-seq2seq model "
                f"from {self.model_path}: {exc}"
            ) from exc

        feature_extractor = getattr(self.processor, "feature_extractor", None)
        self.sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16000))
        if self.sampling_rate <= 0:
            self.close()
            raise RuntimeError(
                f"Model processor reported an invalid sampling rate: {self.sampling_rate}"
            )
        self._model_class = type(self.model).__name__
        self._processor_class = type(self.processor).__name__
        self._effective_dtype = str(_model_dtype(self.model)).removeprefix("torch.")
        self._generation_kwargs = dict(profile.decoding.generation_kwargs)
        self._generation_kwargs.setdefault("return_timestamps", False)
        self._generation_kwargs.setdefault("task", profile.task)
        if profile.language:
            self._generation_kwargs.setdefault("language", profile.language)

    def _decode(
        self,
        audio_batch: list[np.ndarray],
        *,
        long_form: bool = False,
    ) -> list[str]:
        processor_kwargs: dict[str, Any] = {
            "sampling_rate": self.sampling_rate,
            "return_tensors": "pt",
        }
        if long_form:
            # Hugging Face Whisper long-form inference requires complete, untruncated
            # features, longest-in-batch padding, and an attention mask.
            processor_kwargs.update(
                truncation=False,
                padding="longest",
                return_attention_mask=True,
            )
        else:
            processor_kwargs["padding"] = True

        encoded = self.processor(audio_batch, **processor_kwargs)
        model_inputs = _move_model_inputs(
            encoded,
            device=self.device,
            floating_dtype=_model_dtype(self.model),
        )
        generation_kwargs = dict(self._generation_kwargs)
        if long_form:
            # Timestamp prediction activates Whisper's sequential long-form
            # algorithm instead of the <=30-second single-call path.
            generation_kwargs["return_timestamps"] = True

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                **generation_kwargs,
            )
        decoded_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        if isinstance(decoded_texts, str):
            decoded_texts = [decoded_texts]
        return [str(text).strip() for text in decoded_texts]

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        rows: list[TranscriptionResult | None] = [None] * len(audio_paths)
        valid_audio: list[np.ndarray] = []
        valid_indices: list[int] = []
        durations: dict[int, float] = {}

        for index, raw_path in enumerate(audio_paths):
            audio_path = str(raw_path)
            try:
                audio, _, duration, _ = load_audio_and_resample(
                    audio_path,
                    self.sampling_rate,
                )
            except Exception as exc:
                rows[index] = TranscriptionResult(
                    audio_filepath=audio_path,
                    duration=0.0,
                    pred_text="",
                    error=f"failed_to_read_audio: {exc}",
                )
                continue
            valid_audio.append(audio)
            valid_indices.append(index)
            durations[index] = duration

        if valid_audio:
            long_form_config = self.profile.decoding.long_form
            decode_groups: dict[bool, list[tuple[int, np.ndarray]]] = {
                False: [],
                True: [],
            }
            for index, audio in zip(valid_indices, valid_audio):
                is_long_form = bool(
                    long_form_config is not None
                    and durations[index] > long_form_config.threshold_seconds
                )
                decode_groups[is_long_form].append((index, audio))

            guard = self.profile.decoding.hallucination_guard
            for long_form, group in decode_groups.items():
                if not group:
                    continue
                group_indices = [index for index, _ in group]
                group_audio = [audio for _, audio in group]
                try:
                    predictions = self._decode(group_audio, long_form=long_form)
                    if len(predictions) != len(group_indices):
                        raise RuntimeError(
                            "Transformers speech-seq2seq inference returned an "
                            "unexpected number of predictions: expected "
                            f"{len(group_indices)}, got {len(predictions)}"
                        )
                except Exception as exc:
                    for index in group_indices:
                        rows[index] = TranscriptionResult(
                            audio_filepath=str(audio_paths[index]),
                            duration=durations[index],
                            pred_text="",
                            error=f"inference_failed: {exc}",
                        )
                    continue

                for index, prediction in zip(group_indices, predictions):
                    raw_prediction = prediction
                    adjusted = False
                    if guard is not None:
                        prediction, adjusted = apply_hallucination_guard(
                            prediction,
                            duration=durations[index],
                            config=guard,
                        )
                    rows[index] = TranscriptionResult(
                        audio_filepath=str(audio_paths[index]),
                        duration=durations[index],
                        pred_text=prediction,
                        raw_pred_text=raw_prediction if adjusted else None,
                    )

        if any(row is None for row in rows):
            raise RuntimeError(
                "Transformers speech-seq2seq backend failed to produce every result row"
            )
        return [row for row in rows if row is not None]

    def metadata(self) -> dict[str, Any]:
        return {
            "framework": "transformers",
            "adapter": "speech_seq2seq",
            "model_class": self._model_class,
            "processor_class": self._processor_class,
            "device": str(self.device),
            "sampling_rate": self.sampling_rate,
            "requested_torch_dtype": self.profile.loader.torch_dtype,
            "effective_torch_dtype": self._effective_dtype,
            "dtype_fallback": self._dtype_fallback,
            "generation_kwargs": dict(self._generation_kwargs),
            "long_form": (
                vars(self.profile.decoding.long_form)
                if self.profile.decoding.long_form is not None
                else None
            ),
            "hallucination_guard": (
                vars(self.profile.decoding.hallucination_guard)
                if self.profile.decoding.hallucination_guard is not None
                else None
            ),
        }

    def close(self) -> None:
        self.model = None
        self.processor = None
        if getattr(self, "device", None) is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
