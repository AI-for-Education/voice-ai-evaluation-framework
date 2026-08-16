from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference.onnxruntime.backend import OnnxRuntimeCtcBackend
from inference.profile import load_profile, resolve_model_path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ARTIFACT_SMOKE = os.environ.get("RUN_ASR_ARTIFACT_SMOKE") == "1"


@pytest.mark.artifact
@pytest.mark.skipif(
    not RUN_ARTIFACT_SMOKE,
    reason="set RUN_ASR_ARTIFACT_SMOKE=1 to load local model weights",
)
@pytest.mark.parametrize(
    ("profile_name", "precision"),
    [
        ("swahili-exp41-ctc-onnx-fp32.yaml", "fp32"),
        ("swahili-exp41-ctc-onnx-int8.yaml", "int8_dynamic_weights"),
    ],
)
def test_exp41_onnx_artifact_loads(profile_name: str, precision: str) -> None:
    profile = load_profile(REPO_ROOT / "inference/onnxruntime/profiles" / profile_name)
    model_path = resolve_model_path(profile, require_exists=False)
    if not model_path.exists():
        pytest.skip(f"local model artifact is absent: {model_path}")

    backend = OnnxRuntimeCtcBackend(
        profile=profile,
        model_path=model_path,
        num_threads=1,
    )
    try:
        metadata = backend.metadata()
        assert metadata["framework"] == "onnxruntime"
        assert metadata["provider"] == "CPUExecutionProvider"
        assert metadata["precision"] == precision
        assert metadata["same_trained_checkpoint"] is True
        assert metadata["mobile_hardware_emulated"] is False
    finally:
        backend.close()
