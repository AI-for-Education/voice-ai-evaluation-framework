from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inference.onnxruntime import android_infer
from inference.onnxruntime.android_backend import (
    AndroidParityCtcBackend,
    decode_android_greedy,
    extract_android_features,
    linear_resample,
)
from inference.profile import parse_profile


def _profile():
    return parse_profile(
        {
            "profile_schema_version": 2,
            "inference_setup_id": "android-int8-test",
            "inference_library": "onnxruntime",
            "adapter": "android_ctc",
            "artifact": "android/model.int8.onnx",
            "language": "sw",
            "decoding": {"strategy": "greedy"},
        }
    )


def test_android_profile_is_a_separate_supported_adapter() -> None:
    profile = _profile()
    assert profile.adapter == "android_ctc"
    assert profile.decoding.strategy == "greedy"


def test_android_cli_enforces_single_item_and_single_thread_before_loading(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        android_infer,
        "load_profile",
        lambda *args, **kwargs: pytest.fail("profile loading must not start"),
    )
    with pytest.raises(SystemExit, match="--batch_size 1"):
        android_infer.main(
            [
                "--inference_profile",
                "profile.yaml",
                "--root_audio_dir",
                "audio",
                "--batch_size",
                "2",
            ]
        )
    with pytest.raises(SystemExit, match="--num_threads 1"):
        android_infer.main(
            [
                "--inference_profile",
                "profile.yaml",
                "--root_audio_dir",
                "audio",
                "--num_threads",
                "2",
            ]
        )


def test_android_linear_resampler_matches_reference_positions() -> None:
    actual = linear_resample(np.asarray([0.0, 1.0], dtype=np.float32), 8, 16)
    np.testing.assert_array_equal(
        actual,
        np.asarray([0.0, 0.5, 1.0, 1.0], dtype=np.float32),
    )


def test_android_frontend_is_finite_normalized_and_uses_unpadded_frame_count() -> None:
    samples = np.linspace(-0.5, 0.5, 800, dtype=np.float32)
    features = extract_android_features(samples)

    assert features.shape == (80, 3)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    np.testing.assert_allclose(
        features[[0, 10, 40, 79]],
        np.asarray(
            [
                [1.1692165, -1.2735927, 0.1043759],
                [1.3368778, -1.0674690, -0.27067965],
                [1.2600459, -1.1860917, -0.07396338],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(features.mean(axis=1), 0.0, atol=5e-4)


def test_android_greedy_decoder_collapses_repeats_and_blanks() -> None:
    vocab = {0: "<unk>", 1: "▁a", 2: "b", 3: "<blk>"}
    ids = [1, 1, 3, 1, 2, 2, 3]
    logits = np.full((len(ids), len(vocab)), -10.0, dtype=np.float32)
    logits[np.arange(len(ids)), ids] = 10.0

    assert decode_android_greedy(logits, vocab, blank_id=3) == "a ab"


def test_android_backend_rejects_multi_item_batches() -> None:
    backend = AndroidParityCtcBackend.__new__(AndroidParityCtcBackend)
    with pytest.raises(RuntimeError, match="exactly one audio item"):
        backend.transcribe_batch(["one.wav", "two.wav"])


def test_android_metadata_explicitly_excludes_physical_device_validation() -> None:
    backend = AndroidParityCtcBackend.__new__(AndroidParityCtcBackend)
    backend.model_path = Path("model.int8.onnx")
    backend.verified = {
        "model_sha256": "test-sha256",
        "quantized_operator_counts": {},
    }

    metadata = backend.metadata()

    assert metadata["execution_platform"] == "pc"
    assert metadata["mobile_hardware_emulated"] is False
    assert metadata["physical_device_validated"] is False
    assert metadata["validation_scope"] == "pc_executed_android_behavior_accuracy_proxy"


def test_android_metadata_marks_controlled_shared_runtime() -> None:
    backend = AndroidParityCtcBackend.__new__(AndroidParityCtcBackend)
    backend.model_path = Path("model.int8.onnx")
    backend.verified = {
        "model_sha256": "test-sha256",
        "quantized_operator_counts": {},
    }
    backend.expected_ort_version = None
    backend.onnxruntime_version = "1.23.2"

    metadata = backend.metadata()

    assert metadata["validation_scope"] == "pc_controlled_android_frontend"
    assert metadata["onnxruntime_version"] == "1.23.2"
    assert "onnxruntime_version_required" not in metadata
    assert metadata["validation"]["inference_engine_version"] == (
        "recorded_not_pinned"
    )
