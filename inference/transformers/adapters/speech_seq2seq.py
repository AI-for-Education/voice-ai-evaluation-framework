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
from inference.profile import ModelProfile, ProfileError
from inference.provenance import generation_provenance
from inference.transformers.adapters.ctc import (
    _loader_dtype,
    _model_dtype,
    _move_model_inputs,
    _select_device,
)


# Inference emits the model hypothesis without post-decoding text modification.


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
            raise ProfileError(
                f"Transformers model directory not found: {self.model_path}"
            )

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
        self._chunked_files = 0
        self._chunks_generated = 0
        if profile.language:
            self._generation_kwargs.setdefault("language", profile.language)
        self._generation_provenance = generation_provenance(
            getattr(self.model, "generation_config", None),
            requested_kwargs=profile.decoding.generation_kwargs,
            call_kwargs=self._generation_kwargs,
            conditional_overrides=(
                {"long_form": {"return_timestamps": True}}
                if profile.decoding.long_form is not None
                else {}
            ),
        )

    def _split_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        audio_config = self.profile.audio
        if audio_config is None:
            return [audio]
        maximum_samples = max(1, int(audio_config.maximum_seconds * self.sampling_rate))
        if len(audio) <= maximum_samples:
            return [audio]
        chunk_samples = max(1, int(audio_config.chunk_seconds * self.sampling_rate))
        return [
            audio[start : start + chunk_samples]
            for start in range(0, len(audio), chunk_samples)
        ]

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
            chunked_audio: list[tuple[int, list[np.ndarray]]] = []
            decoded_predictions: dict[int, str] = {}
            for index, audio in zip(valid_indices, valid_audio):
                chunks = self._split_audio(audio)
                if len(chunks) > 1:
                    self._chunked_files += 1
                    self._chunks_generated += len(chunks)
                    chunked_audio.append((index, chunks))
                    continue
                is_long_form = bool(
                    long_form_config is not None
                    and durations[index] > long_form_config.threshold_seconds
                )
                decode_groups[is_long_form].append((index, chunks[0]))

            # Decode and return the model hypothesis without text postprocessing.
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

                decoded_predictions.update(zip(group_indices, predictions))

            for index, chunks in chunked_audio:
                try:
                    predictions = self._decode(chunks, long_form=False)
                    if len(predictions) != len(chunks):
                        raise RuntimeError(
                            "Transformers speech-seq2seq chunk inference returned an "
                            f"unexpected number of predictions: expected {len(chunks)}, "
                            f"got {len(predictions)}"
                        )
                except Exception as exc:
                    rows[index] = TranscriptionResult(
                        audio_filepath=str(audio_paths[index]),
                        duration=durations[index],
                        pred_text="",
                        error=f"inference_failed: {exc}",
                    )
                    continue
                decoded_predictions[index] = " ".join(
                    text for text in predictions if text
                ).strip()

            for index in valid_indices:
                if rows[index] is not None:
                    continue
                rows[index] = TranscriptionResult(
                    audio_filepath=str(audio_paths[index]),
                    duration=durations[index],
                    pred_text=decoded_predictions[index],
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
            "generation": self._generation_provenance,
            "long_form": (
                vars(self.profile.decoding.long_form)
                if self.profile.decoding.long_form is not None
                else None
            ),
            "chunking": (
                vars(self.profile.audio) if self.profile.audio is not None else None
            ),
            "chunking_stats": {
                "long_audio_files": self._chunked_files,
                "chunks_generated": self._chunks_generated,
            },
            "model_output_field": "pred_text",
        }

    def close(self) -> None:
        self.model = None
        self.processor = None
        if getattr(self, "device", None) is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
