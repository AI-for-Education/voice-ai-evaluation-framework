from __future__ import annotations

import numpy as np

from inference.onnxruntime import backend as onnx_backend
from inference.onnxruntime.backend import OnnxRuntimeCtcBackend
from inference.profile import parse_profile


def _profile():
    return parse_profile(
        {
            "profile_schema_version": 2,
            "inference_setup_id": "exp41-onnx-test",
            "inference_library": "onnxruntime",
            "adapter": "ctc",
            "artifact": "bundle/model.onnx",
            "language": "sw",
            "decoding": {"strategy": "greedy"},
        }
    )


def test_backend_preserves_order_and_reports_read_errors(monkeypatch) -> None:
    backend = OnnxRuntimeCtcBackend.__new__(OnnxRuntimeCtcBackend)
    backend.profile = _profile()
    backend.recognizer = object()

    def fake_load(path: str, target_sr: int):
        if path == "bad.wav":
            raise RuntimeError("cannot decode")
        duration = 1.0 if path == "one.wav" else 2.0
        return np.zeros(16, dtype=np.float32), target_sr, duration, False

    monkeypatch.setattr(onnx_backend, "load_audio_and_resample", fake_load)
    monkeypatch.setattr(backend, "_decode", lambda _audio: ["moja", "mbili"])

    rows = backend.transcribe_batch(["one.wav", "bad.wav", "two.wav"])

    assert [row.audio_filepath for row in rows] == [
        "one.wav",
        "bad.wav",
        "two.wav",
    ]
    assert [row.pred_text for row in rows] == ["moja", "", "mbili"]
    assert rows[1].error == "failed_to_read_audio: cannot decode"
    assert [row.duration for row in rows] == [1.0, 0.0, 2.0]


def test_backend_turns_batch_failure_into_one_error_per_input(monkeypatch) -> None:
    backend = OnnxRuntimeCtcBackend.__new__(OnnxRuntimeCtcBackend)
    backend.profile = _profile()
    backend.recognizer = object()

    monkeypatch.setattr(
        onnx_backend,
        "load_audio_and_resample",
        lambda path, target_sr: (
            np.zeros(16, dtype=np.float32),
            target_sr,
            1.0,
            False,
        ),
    )

    def fail(_audio):
        raise RuntimeError("inference engine failed")

    monkeypatch.setattr(backend, "_decode", fail)
    rows = backend.transcribe_batch(["one.wav", "two.wav"])

    assert len(rows) == 2
    assert all(row.pred_text == "" for row in rows)
    assert all(
        row.error == "inference_failed: inference engine failed" for row in rows
    )
