"""CPU ONNX Runtime backend for PC validation of mobile-target Exp41 exports."""

from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from inference.common import load_audio_and_resample
from inference.contracts import TranscriptionResult
from inference.onnxruntime.android_backend import verify_android_bundle
from inference.onnxruntime.artifacts import (
    FP32_MODEL_NAME,
    INT8_MODEL_NAME,
    ArtifactError,
    verify_artifact,
)
from inference.profile import ModelProfile, ProfileError

SAMPLE_RATE = 16000


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


class OnnxRuntimeCtcBackend:
    """Greedy CTC inference for FP32 or dynamic-weight INT8 Exp41 ONNX."""

    def __init__(
        self,
        profile: ModelProfile,
        model_path: str | Path,
        *,
        num_threads: int = 2,
        verify_hashes: bool = True,
    ) -> None:
        if profile.framework != "onnxruntime" or profile.adapter != "ctc":
            raise ProfileError(
                "OnnxRuntimeCtcBackend requires an onnxruntime/ctc profile"
            )
        if profile.decoding.strategy != "greedy":
            raise ProfileError("The ONNX Runtime CTC adapter requires greedy decoding")
        if num_threads < 1:
            raise ProfileError("ONNX Runtime num_threads must be at least 1")

        self.profile = profile
        self.model_path = Path(model_path)
        if self.model_path.name == FP32_MODEL_NAME:
            self.quantization = None
            self.precision = "fp32"
        elif self.model_path.name == INT8_MODEL_NAME:
            self.quantization = "int8"
            self.precision = "int8_dynamic_weights"
        else:
            raise ProfileError(
                "ONNX Runtime profile artifact must select model.onnx or "
                f"model.int8.onnx, got: {self.model_path.name}"
            )

        try:
            verified = verify_artifact(
                self.model_path,
                verify_hashes=verify_hashes,
            )
        except ArtifactError as local_error:
            try:
                published = verify_android_bundle(self.model_path)
            except RuntimeError as published_error:
                raise RuntimeError(
                    "Invalid Exp41 ONNX artifact bundle: "
                    f"local export validation failed ({local_error}); "
                    f"published Android validation failed ({published_error})"
                ) from published_error
            self.artifact_metadata = None
            self.vocab_size = len(published["vocab"])
            self._artifact_metadata_fields = {
                "artifact_target": "packaged_android_reference",
                "source_checkpoint_verified": False,
                "selected_artifact": self.model_path.name,
                "selected_artifact_sha256": published["model_sha256"],
            }
        else:
            self.artifact_metadata = verified["metadata"]
            self.vocab_size = int(verified["vocab_size"])
            source = self.artifact_metadata["source_checkpoint"]
            artifact_record = self.artifact_metadata["artifacts"][
                self.model_path.name
            ]
            self._artifact_metadata_fields = {
                "artifact_target": artifact_record["deployment_role"],
                "same_trained_checkpoint": True,
                "source_profile_id": source["profile_id"],
                "source_checkpoint_sha256": source["sha256"],
                "selected_artifact": self.model_path.name,
                "selected_artifact_sha256": artifact_record["sha256"],
            }

        try:
            import onnx_asr
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ONNX inference requires onnx-asr==0.12.0 and ONNX Runtime; "
                "run it through the shared ASR image"
            ) from exc

        self._onnx_asr = onnx_asr
        self._ort = ort
        self.num_threads = num_threads
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = 1
        self._model_load_call = {
            "api": "onnx_asr.load_model",
            "model_type": "nemo-conformer-ctc",
            "quantization": self.quantization,
            "providers": ["CPUExecutionProvider"],
            "session_options": {
                "intra_op_num_threads": num_threads,
                "inter_op_num_threads": 1,
            },
            "preprocessor_config": {
                "max_concurrent_workers": 1,
                "use_numpy_preprocessors": True,
            },
        }
        try:
            self.recognizer = onnx_asr.load_model(
                "nemo-conformer-ctc",
                self.model_path.parent,
                quantization=self.quantization,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
                preprocessor_config={
                    "max_concurrent_workers": 1,
                    "use_numpy_preprocessors": True,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Exp41 ONNX artifact {self.model_path}: {exc}"
            ) from exc

    def _decode(self, waveforms: list[np.ndarray]) -> list[str]:
        predictions = self.recognizer.recognize(
            waveforms,
            sample_rate=SAMPLE_RATE,
        )
        if not isinstance(predictions, list):
            raise TypeError("onnx-asr returned a non-list result for a waveform batch")
        return [str(value or "").strip() for value in predictions]

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        rows: list[TranscriptionResult | None] = [None] * len(audio_paths)
        valid_audio: list[np.ndarray] = []
        valid_indices: list[int] = []
        durations: dict[int, float] = {}

        for index, raw_path in enumerate(audio_paths):
            audio_path = str(raw_path)
            try:
                waveform, _, duration, _ = load_audio_and_resample(
                    audio_path,
                    SAMPLE_RATE,
                )
            except Exception as exc:  # noqa: BLE001 - preserve one row per input.
                rows[index] = TranscriptionResult(
                    audio_filepath=audio_path,
                    duration=0.0,
                    pred_text="",
                    error=f"failed_to_read_audio: {exc}",
                )
                continue
            valid_audio.append(waveform)
            valid_indices.append(index)
            durations[index] = duration

        if valid_audio:
            try:
                predictions = self._decode(valid_audio)
                if len(predictions) != len(valid_indices):
                    raise RuntimeError(
                        "ONNX inference returned an unexpected number of predictions: "
                        f"expected {len(valid_indices)}, got {len(predictions)}"
                    )
            except Exception as exc:  # noqa: BLE001 - convert batch failure to rows.
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
            raise RuntimeError("ONNX backend failed to produce every result row")
        return [row for row in rows if row is not None]

    def metadata(self) -> dict[str, Any]:
        return {
            "framework": "onnxruntime",
            "adapter": "ctc",
            "device": "cpu",
            "provider": "CPUExecutionProvider",
            "execution_platform": "pc",
            "execution_architecture": platform.machine(),
            "mobile_hardware_emulated": False,
            **self._artifact_metadata_fields,
            "precision": self.precision,
            "quantization": self.quantization,
            "decoding_strategy": "greedy_ctc",
            "sample_rate": SAMPLE_RATE,
            "vocab_size_including_ctc_blank": self.vocab_size,
            "num_threads": self.num_threads,
            "model_load_call": self._model_load_call,
            "segmentation": "reuses supplied framework audio paths; no re-segmentation",
            "onnx_asr_version": _distribution_version("onnx-asr"),
            "onnxruntime_version": _distribution_version(
                "onnxruntime-gpu", "onnxruntime"
            ),
        }

    def close(self) -> None:
        self.recognizer = None
