"""Android-parity CPU inference for the packaged Exp41 INT8 model.

This stays separate from the NeMo-compatible ``onnx-asr`` backend so the
desktop baseline and the mobile-behaviour proxy remain distinct measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from inference.contracts import TranscriptionResult
from inference.profile import ModelProfile, ProfileError

SAMPLE_RATE = 16_000
FEATURE_DIM = 80
WINDOW_SIZE = 400
HOP_SIZE = 160
FFT_SIZE = 512
FFT_BINS = FFT_SIZE // 2 + 1
VOCAB_SIZE = 1_025
BLANK_ID = 1_024
EXPECTED_ORT_VERSION = "1.22.0"
ANDROID_REFERENCE_REVISION = "6cd031267aed45e47347bc05a0eb63eee285f0d8"
ANDROID_MODEL_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1Y0ozSCenl6Ftfyzhee1j7X51nLCxrTMJ&export=download&confirm=t"
)
ANDROID_MODEL_SHA256 = (
    "bef2607617aef3520a5313d76c08a03eda692f69f1f1561bc3e980dab49afaa7"
)
ANDROID_MODEL_SIZE = 139_737_438
FRONTEND_VERSION = f"nemoasr_android_{ANDROID_REFERENCE_REVISION[:12]}_kotlin_proxy_v1"

_EXPECTED_CONFIG = {
    "features_size": FEATURE_DIM,
    "model_type": "nemo-conformer-ctc",
    "subsampling_factor": 8,
}
_EXPECTED_QUANTIZED_OPS = {
    "DynamicQuantizeLinear": 170,
    "MatMulInteger": 163,
    "ConvInteger": 60,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_vocab(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise RuntimeError(f"Android-parity vocabulary is missing: {path}")
    tokens: dict[int, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        trimmed = line.strip()
        if not trimmed:
            continue
        split = trimmed.rfind(" ")
        if split <= 0:
            raise RuntimeError(f"Invalid vocabulary line {line_number}: {line!r}")
        token = trimmed[:split]
        try:
            token_id = int(trimmed[split + 1 :])
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid vocabulary id on line {line_number}: {line!r}"
            ) from exc
        if token_id in tokens:
            raise RuntimeError(f"Duplicate vocabulary id: {token_id}")
        tokens[token_id] = token

    if sorted(tokens) != list(range(VOCAB_SIZE)):
        raise RuntimeError(
            f"Android-parity vocabulary must contain ids 0..{VOCAB_SIZE - 1}"
        )
    if tokens[BLANK_ID] != "<blk>":
        raise RuntimeError(f"Expected <blk> at vocabulary id {BLANK_ID}")
    return tokens


def verify_android_bundle(model_path: str | Path) -> dict[str, Any]:
    """Fail closed unless the selected bundle is the pinned mobile artifact."""
    selected = Path(model_path)
    if selected.name != "model.int8.onnx":
        raise RuntimeError(
            "Android-parity profile must select a file named model.int8.onnx"
        )
    if not selected.is_file():
        raise RuntimeError(
            f"Android INT8 model is missing: {selected}. "
            "See inference/onnxruntime/README.md for the pinned download."
        )
    if selected.stat().st_size != ANDROID_MODEL_SIZE:
        raise RuntimeError(
            f"Android INT8 model size mismatch: expected {ANDROID_MODEL_SIZE}, "
            f"got {selected.stat().st_size}"
        )
    digest = _sha256_file(selected)
    if digest != ANDROID_MODEL_SHA256:
        raise RuntimeError(
            "Android INT8 model SHA-256 mismatch: "
            f"expected {ANDROID_MODEL_SHA256}, got {digest}"
        )

    config_path = selected.parent / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Android-parity config is unreadable: {config_path}") from exc
    if config != _EXPECTED_CONFIG:
        raise RuntimeError(
            f"Android-parity config mismatch: expected {_EXPECTED_CONFIG}, got {config}"
        )
    vocab = _load_vocab(selected.parent / "vocab.txt")

    try:
        import onnx
        from onnx import TensorProto
    except ModuleNotFoundError as exc:
        raise RuntimeError("Android artifact validation requires onnx==1.17.0") from exc

    graph = onnx.load(str(selected), load_external_data=False)
    onnx.checker.check_model(graph)
    inputs = {item.name: item for item in graph.graph.input}
    if set(inputs) != {"audio_signal", "length"}:
        raise RuntimeError(
            f"Android ONNX inputs must be audio_signal and length, got {sorted(inputs)}"
        )
    audio_type = inputs["audio_signal"].type.tensor_type
    length_type = inputs["length"].type.tensor_type
    audio_dims = audio_type.shape.dim
    if audio_type.elem_type != TensorProto.FLOAT or len(audio_dims) != 3:
        raise RuntimeError("audio_signal must be a rank-3 float tensor")
    if audio_dims[1].dim_value != FEATURE_DIM:
        raise RuntimeError(f"audio_signal feature dimension must be {FEATURE_DIM}")
    if length_type.elem_type != TensorProto.INT64:
        raise RuntimeError("length must be an int64 tensor")
    if len(graph.graph.output) != 1:
        raise RuntimeError("Android ONNX graph must expose exactly one logits output")
    output_type = graph.graph.output[0].type.tensor_type
    output_dims = output_type.shape.dim
    if output_type.elem_type != TensorProto.FLOAT or output_dims[-1].dim_value != VOCAB_SIZE:
        raise RuntimeError(f"Android ONNX output must end in {VOCAB_SIZE} float logits")

    op_counts = Counter(node.op_type for node in graph.graph.node)
    for name, expected in _EXPECTED_QUANTIZED_OPS.items():
        if op_counts[name] != expected:
            raise RuntimeError(
                f"Android ONNX quantized operator mismatch for {name}: "
                f"expected {expected}, got {op_counts[name]}"
            )
    return {
        "model_sha256": digest,
        "model_size_bytes": selected.stat().st_size,
        "vocab": vocab,
        "quantized_operator_counts": dict(_EXPECTED_QUANTIZED_OPS),
    }


def linear_resample(
    samples: np.ndarray,
    from_rate: int,
    to_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Mirror ``OnnxAsrEngine.resampleLinear`` from the Android demo."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Android audio must be a non-empty mono waveform")
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError("sample rates must be positive")
    if from_rate == to_rate:
        return samples

    output_size = max(1, (int(samples.size) * int(to_rate)) // int(from_rate))
    scale = float(from_rate) / float(to_rate)
    positions = np.arange(output_size, dtype=np.float64) * scale
    left = np.clip(positions.astype(np.int64), 0, samples.size - 1)
    right = np.minimum(left + 1, samples.size - 1)
    alpha = (positions - left).astype(np.float32)
    return (
        samples[left] * (np.float32(1.0) - alpha) + samples[right] * alpha
    ).astype(np.float32, copy=False)


def _create_android_mel_filterbank() -> np.ndarray:
    filters = np.zeros((FEATURE_DIM, FFT_BINS), dtype=np.float32)

    mel_min = 2595.0 * math.log10(1.0)
    mel_max = 2595.0 * math.log10(1.0 + (SAMPLE_RATE / 2.0) / 700.0)
    mel_points = np.asarray(
        [
            mel_min + (mel_max - mel_min) * index / (FEATURE_DIM + 1)
            for index in range(FEATURE_DIM + 2)
        ],
        dtype=np.float64,
    )
    hz_points = np.asarray(
        [700.0 * (10.0 ** (mel / 2595.0) - 1.0) for mel in mel_points],
        dtype=np.float64,
    )
    bin_points = np.asarray(
        [
            min(max(int((FFT_SIZE + 1) * hz / SAMPLE_RATE), 0), FFT_BINS - 1)
            for hz in hz_points
        ],
        dtype=np.int64,
    )

    for mel_index in range(1, FEATURE_DIM + 1):
        left = int(bin_points[mel_index - 1])
        center = max(int(bin_points[mel_index]), left + 1)
        right = max(int(bin_points[mel_index + 1]), center + 1)
        for frequency_index in range(left, min(center, FFT_BINS)):
            filters[mel_index - 1, frequency_index] = (
                (frequency_index - left) / (center - left)
            )
        for frequency_index in range(center, min(right, FFT_BINS)):
            filters[mel_index - 1, frequency_index] = (
                (right - frequency_index) / (right - center)
            )
        norm = np.float32(
            2.0 / max(float(hz_points[mel_index + 1] - hz_points[mel_index - 1]), 1e-10)
        )
        filters[mel_index - 1] = np.asarray(
            filters[mel_index - 1] * norm,
            dtype=np.float32,
        )
    return filters


def _create_android_fft_tables() -> tuple[
    np.ndarray,
    tuple[tuple[int, np.ndarray, np.ndarray], ...],
]:
    """Precompute the Android FFT's bit reversal and float32 twiddles."""
    bit_reversal = np.arange(FFT_SIZE, dtype=np.intp)
    reversed_index = 0
    for index in range(1, FFT_SIZE):
        bit = FFT_SIZE >> 1
        while reversed_index & bit:
            reversed_index ^= bit
            bit >>= 1
        reversed_index ^= bit
        if index < reversed_index:
            bit_reversal[index], bit_reversal[reversed_index] = (
                bit_reversal[reversed_index],
                bit_reversal[index],
            )

    stages: list[tuple[int, np.ndarray, np.ndarray]] = []
    length = 2
    while length <= FFT_SIZE:
        half = length // 2
        angle = np.float32(-2.0 * math.pi / length)
        step_cos = np.float32(math.cos(float(angle)))
        step_sin = np.float32(math.sin(float(angle)))
        cosines = np.empty(half, dtype=np.float32)
        sines = np.empty(half, dtype=np.float32)
        current_cos = np.float32(1.0)
        current_sin = np.float32(0.0)
        for index in range(half):
            cosines[index] = current_cos
            sines[index] = current_sin
            next_cos = np.float32(
                np.float32(current_cos * step_cos)
                - np.float32(current_sin * step_sin)
            )
            next_sin = np.float32(
                np.float32(current_cos * step_sin)
                + np.float32(current_sin * step_cos)
            )
            current_cos = next_cos
            current_sin = next_sin
        stages.append((length, cosines, sines))
        length <<= 1
    return bit_reversal, tuple(stages)


def _android_fft_power(frames: np.ndarray) -> np.ndarray:
    """Run the repository's float32 radix-2 FFT over a batch of frames."""
    real = frames[:, _FFT_BIT_REVERSAL].copy()
    imaginary = np.zeros_like(real)

    for length, cosines, sines in _FFT_STAGES:
        half = length // 2
        real_view = real.reshape(real.shape[0], -1, length)
        imaginary_view = imaginary.reshape(imaginary.shape[0], -1, length)
        left_real = real_view[:, :, :half].copy()
        left_imaginary = imaginary_view[:, :, :half].copy()
        right_real = real_view[:, :, half:].copy()
        right_imaginary = imaginary_view[:, :, half:].copy()

        right_real_cos = np.asarray(right_real * cosines, dtype=np.float32)
        right_imaginary_sin = np.asarray(
            right_imaginary * sines,
            dtype=np.float32,
        )
        value_real = np.asarray(
            right_real_cos - right_imaginary_sin,
            dtype=np.float32,
        )
        right_real_sin = np.asarray(right_real * sines, dtype=np.float32)
        right_imaginary_cos = np.asarray(
            right_imaginary * cosines,
            dtype=np.float32,
        )
        value_imaginary = np.asarray(
            right_real_sin + right_imaginary_cos,
            dtype=np.float32,
        )

        real_view[:, :, :half] = np.asarray(
            left_real + value_real,
            dtype=np.float32,
        )
        imaginary_view[:, :, :half] = np.asarray(
            left_imaginary + value_imaginary,
            dtype=np.float32,
        )
        real_view[:, :, half:] = np.asarray(
            left_real - value_real,
            dtype=np.float32,
        )
        imaginary_view[:, :, half:] = np.asarray(
            left_imaginary - value_imaginary,
            dtype=np.float32,
        )

    real = real[:, :FFT_BINS]
    imaginary = imaginary[:, :FFT_BINS]
    return np.asarray(
        np.asarray(real * real, dtype=np.float32)
        + np.asarray(imaginary * imaginary, dtype=np.float32),
        dtype=np.float32,
    )


_HANN_WINDOW = np.asarray(
    [
        np.float32(
            0.5
            - 0.5
            * math.cos(2.0 * math.pi * index / (WINDOW_SIZE - 1))
        )
        for index in range(WINDOW_SIZE)
    ],
    dtype=np.float32,
)
_MEL_FILTERBANK = _create_android_mel_filterbank()
_FFT_BIT_REVERSAL, _FFT_STAGES = _create_android_fft_tables()


def extract_android_features(samples: np.ndarray) -> np.ndarray:
    """Reproduce the Android demo's 80-bin normalized log-mel frontend."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Android audio must be a non-empty mono waveform")

    frame_count = (
        1
        if samples.size < WINDOW_SIZE
        else 1 + (samples.size - WINDOW_SIZE) // HOP_SIZE
    )
    frames = np.zeros((frame_count, FFT_SIZE), dtype=np.float32)
    if samples.size < WINDOW_SIZE:
        frames[0, : samples.size] = samples * _HANN_WINDOW[: samples.size]
    else:
        windows = np.lib.stride_tricks.sliding_window_view(samples, WINDOW_SIZE)
        frames[:, :WINDOW_SIZE] = windows[::HOP_SIZE][:frame_count] * _HANN_WINDOW

    power = _android_fft_power(frames)
    features = np.empty((FEATURE_DIM, frame_count), dtype=np.float32)
    for mel_index, mel_filter in enumerate(_MEL_FILTERBANK):
        products = np.asarray(power * mel_filter[None, :], dtype=np.float32)
        energy = np.cumsum(products, axis=1, dtype=np.float32)[:, -1]
        features[mel_index] = np.asarray(
            [
                np.float32(math.log(float(max(value, np.float32(1e-10)))))
                for value in energy
            ],
            dtype=np.float32,
        )

    for mel_index in range(FEATURE_DIM):
        values = features[mel_index]
        mean = np.float32(
            np.cumsum(values, dtype=np.float32)[-1] / np.float32(values.size)
        )
        differences = np.asarray(values - mean, dtype=np.float32)
        squared_differences = np.asarray(
            differences * differences,
            dtype=np.float32,
        )
        variance = np.float32(
            np.cumsum(squared_differences, dtype=np.float32)[-1]
            / np.float32(values.size)
        )
        std = np.float32(
            max(np.float32(math.sqrt(float(variance))), np.float32(1e-5))
        )
        features[mel_index] = np.asarray(differences / std, dtype=np.float32)
    return np.ascontiguousarray(features, dtype=np.float32)


def decode_android_greedy(
    logits: np.ndarray,
    id_to_token: dict[int, str],
    blank_id: int = BLANK_ID,
) -> str:
    """Mirror the Android demo's argmax/collapse/blank CTC decoder."""
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != len(id_to_token):
        raise ValueError("logits must have shape [frames, vocabulary]")
    token_ids: list[int] = []
    previous = -1
    for best_id in np.argmax(values, axis=1).tolist():
        if best_id != previous and best_id != blank_id:
            token_ids.append(int(best_id))
        previous = int(best_id)
    return "".join(id_to_token.get(token_id, "") for token_id in token_ids).replace(
        "▁", " "
    ).strip()


def _load_android_wav(path: str) -> tuple[np.ndarray, int, float]:
    info = sf.info(path)
    if info.channels != 1:
        raise ValueError("Android demo supports only mono WAV audio")
    if info.subtype != "PCM_16":
        raise ValueError("Android demo supports only 16-bit PCM WAV audio")
    if info.samplerate <= 0:
        raise ValueError(f"invalid source sample rate: {info.samplerate}")
    pcm, sample_rate = sf.read(path, dtype="int16", always_2d=False)
    samples = np.asarray(pcm, dtype=np.float32) / np.float32(32768.0)
    duration = float(samples.size / sample_rate)
    return samples, int(sample_rate), duration


class AndroidParityCtcBackend:
    """Single-item Android pipeline proxy using the exact packaged INT8 graph."""

    def __init__(
        self,
        profile: ModelProfile,
        model_path: str | Path,
        *,
        num_threads: int = 1,
    ) -> None:
        if profile.framework != "onnxruntime" or profile.adapter != "android_ctc":
            raise ProfileError(
                "AndroidParityCtcBackend requires an onnxruntime/android_ctc profile"
            )
        if profile.decoding.strategy != "greedy":
            raise ProfileError("Android parity requires greedy CTC decoding")
        if num_threads != 1:
            raise ProfileError("Android parity requires exactly one runtime thread")

        self.profile = profile
        self.model_path = Path(model_path)
        verified = verify_android_bundle(self.model_path)
        self.id_to_token = verified.pop("vocab")
        self.verified = verified

        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Android parity requires the dedicated ONNX Runtime 1.22.0 image"
            ) from exc
        if ort.__version__ != EXPECTED_ORT_VERSION:
            raise RuntimeError(
                "Android parity requires onnxruntime==1.22.0, "
                f"found {ort.__version__}"
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def _decode_waveform(self, samples: np.ndarray, source_rate: int) -> str:
        resampled = linear_resample(samples, source_rate, SAMPLE_RATE)
        features = extract_android_features(resampled)
        outputs = self.session.run(
            None,
            {
                "audio_signal": features[None, :, :],
                "length": np.asarray([features.shape[1]], dtype=np.int64),
            },
        )
        if not outputs:
            raise RuntimeError("Android ONNX session returned no outputs")
        logits = np.asarray(outputs[0])
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise RuntimeError(f"Unexpected Android logits shape: {logits.shape}")
        return decode_android_greedy(logits[0], self.id_to_token)

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        if len(audio_paths) != 1:
            raise RuntimeError(
                "Android parity accepts exactly one audio item per inference batch"
            )
        audio_path = str(audio_paths[0])
        try:
            samples, source_rate, duration = _load_android_wav(audio_path)
        except Exception as exc:  # noqa: BLE001 - preserve one row per input.
            return [
                TranscriptionResult(
                    audio_filepath=audio_path,
                    duration=0.0,
                    pred_text="",
                    error=f"failed_to_read_android_audio: {exc}",
                )
            ]
        try:
            prediction = self._decode_waveform(samples, source_rate)
        except Exception as exc:  # noqa: BLE001 - preserve one row per input.
            return [
                TranscriptionResult(
                    audio_filepath=audio_path,
                    duration=duration,
                    pred_text="",
                    error=f"android_inference_failed: {exc}",
                )
            ]
        return [
            TranscriptionResult(
                audio_filepath=audio_path,
                duration=duration,
                pred_text=prediction,
            )
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "framework": "onnxruntime",
            "adapter": "android_ctc",
            "device": "cpu",
            "provider": "CPUExecutionProvider",
            "execution_platform": "pc",
            "execution_architecture": platform.machine(),
            "mobile_hardware_emulated": False,
            "artifact_target": "packaged_android_reference",
            "selected_artifact": self.model_path.name,
            "selected_artifact_sha256": self.verified["model_sha256"],
            "precision": "int8_dynamic_weights",
            "quantization": "int8",
            "quantized_operator_counts": self.verified["quantized_operator_counts"],
            "decoding_strategy": "android_greedy_ctc",
            "frontend": FRONTEND_VERSION,
            "frontend_reference_revision": ANDROID_REFERENCE_REVISION,
            "audio_contract": "mono_pcm16_wav",
            "resampling": "android_linear",
            "sample_rate": SAMPLE_RATE,
            "features_size": FEATURE_DIM,
            "batch_size_required": 1,
            "num_threads": 1,
            "onnxruntime_version_required": EXPECTED_ORT_VERSION,
            "vocab_size_including_ctc_blank": VOCAB_SIZE,
            "validation": {
                "artifact_hash": "passed",
                "onnx_contract": "passed",
                "quantized_operators": "passed",
                "runtime_version": "passed",
            },
        }

    def close(self) -> None:
        self.session = None
