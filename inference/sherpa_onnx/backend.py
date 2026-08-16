"""Streaming transducer inference for local Sherpa-ONNX artifacts."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from inference.common import load_audio_and_resample
from inference.contracts import TranscriptionResult
from inference.profile import ModelProfile, ProfileError


SAMPLE_RATE = 16000
TAIL_PADDING_SECONDS = 0.66


def _one_model_file(model_path: Path, prefix: str) -> Path:
    matches = sorted(
        path
        for path in model_path.glob(f"{prefix}*.onnx")
        if ".int8." not in path.name
    )
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "none"
        raise RuntimeError(
            f"Expected exactly one float {prefix}*.onnx file in {model_path}; "
            f"found: {names}"
        )
    return matches[0]


class SherpaOnnxOnlineTransducerBackend:
    """Offline file transcription through Sherpa's streaming RNN-T API."""

    def __init__(
        self,
        profile: ModelProfile,
        model_path: str | Path,
        *,
        provider: str | None = None,
        num_threads: int = 2,
    ) -> None:
        if profile.framework != "sherpa_onnx" or profile.adapter != "online_transducer":
            raise ProfileError(
                "SherpaOnnxOnlineTransducerBackend requires a "
                "sherpa_onnx/online_transducer profile"
            )
        if profile.decoding.strategy not in {
            "greedy_search",
            "modified_beam_search",
        }:
            raise ProfileError(
                "The Sherpa-ONNX adapter requires greedy_search or "
                "modified_beam_search decoding"
            )
        if num_threads < 1:
            raise ProfileError("Sherpa-ONNX num_threads must be at least 1")

        self.profile = profile
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise ProfileError(f"Sherpa-ONNX model directory not found: {self.model_path}")

        self.encoder_path = _one_model_file(self.model_path, "encoder-")
        self.decoder_path = _one_model_file(self.model_path, "decoder-")
        self.joiner_path = _one_model_file(self.model_path, "joiner-")
        self.tokens_path = self.model_path / "tokens.txt"
        if not self.tokens_path.is_file():
            raise RuntimeError(f"Sherpa-ONNX tokens file is missing: {self.tokens_path}")

        try:
            import sherpa_onnx
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Sherpa-ONNX inference requires the sherpa-onnx package in the image"
            ) from exc

        self._sherpa_onnx = sherpa_onnx
        self.provider = provider or self._select_provider()
        self.num_threads = num_threads
        self.decoding_method = profile.decoding.strategy
        search_config = profile.decoding.transducer_search_kwargs
        self.max_active_paths = (
            search_config.max_active_paths if search_config is not None else None
        )
        recognizer_kwargs: dict[str, Any] = {
            "tokens": str(self.tokens_path),
            "encoder": str(self.encoder_path),
            "decoder": str(self.decoder_path),
            "joiner": str(self.joiner_path),
            "num_threads": num_threads,
            "provider": self.provider,
            "sample_rate": SAMPLE_RATE,
            "feature_dim": 80,
            "decoding_method": self.decoding_method,
        }
        if self.max_active_paths is not None:
            recognizer_kwargs["max_active_paths"] = self.max_active_paths
        try:
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                **recognizer_kwargs
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local Sherpa-ONNX model from {self.model_path}: {exc}"
            ) from exc

    def _select_provider(self) -> str:
        available = []
        getter = getattr(self._sherpa_onnx, "get_available_providers", None)
        if callable(getter):
            available = [str(value).lower() for value in getter()]
        if any("cuda" in value for value in available):
            return "cuda"
        try:
            version = importlib.metadata.version("sherpa-onnx")
        except importlib.metadata.PackageNotFoundError:
            version = str(getattr(self._sherpa_onnx, "__version__", ""))
        return "cuda" if "+cuda" in version.lower() else "cpu"

    def _decode(self, audio_batch: list[np.ndarray]) -> list[str]:
        streams = []
        tail = np.zeros(int(TAIL_PADDING_SECONDS * SAMPLE_RATE), dtype=np.float32)
        for audio in audio_batch:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            stream.accept_waveform(SAMPLE_RATE, tail)
            stream.input_finished()
            streams.append(stream)

        while True:
            ready = [stream for stream in streams if self.recognizer.is_ready(stream)]
            if not ready:
                break
            self.recognizer.decode_streams(ready)
        return [str(self.recognizer.get_result(stream)).strip() for stream in streams]

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
                    SAMPLE_RATE,
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
            try:
                predictions = self._decode(valid_audio)
                if len(predictions) != len(valid_indices):
                    raise RuntimeError(
                        "Sherpa-ONNX inference returned an unexpected number of "
                        f"predictions: expected {len(valid_indices)}, got {len(predictions)}"
                    )
            except Exception as exc:
                for index in valid_indices:
                    rows[index] = TranscriptionResult(
                        audio_filepath=str(audio_paths[index]),
                        duration=durations[index],
                        pred_text="",
                        error=f"inference_failed: {exc}",
                    )
            else:
                for index, prediction in zip(valid_indices, predictions):
                    rows[index] = TranscriptionResult(
                        audio_filepath=str(audio_paths[index]),
                        duration=durations[index],
                        pred_text=prediction,
                    )

        if any(row is None for row in rows):
            raise RuntimeError("Sherpa-ONNX backend failed to produce every result row")
        return [row for row in rows if row is not None]

    def metadata(self) -> dict[str, Any]:
        try:
            version = importlib.metadata.version("sherpa-onnx")
        except importlib.metadata.PackageNotFoundError:
            version = str(getattr(self._sherpa_onnx, "__version__", "unknown"))
        payload = {
            "framework": "sherpa_onnx",
            "adapter": "online_transducer",
            "device": "cuda:0" if self.provider == "cuda" else "cpu",
            "provider": self.provider,
            "sampling_rate": SAMPLE_RATE,
            "decoding_strategy": self.decoding_method,
            "num_threads": self.num_threads,
            "sherpa_onnx_version": version,
            "encoder": self.encoder_path.name,
            "decoder": self.decoder_path.name,
            "joiner": self.joiner_path.name,
            "tokens": self.tokens_path.name,
        }
        if self.max_active_paths is not None:
            payload["max_active_paths"] = self.max_active_paths
        return payload

    def close(self) -> None:
        self.recognizer = None
