from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference.onnxruntime.android_backend import (
    ANDROID_MODEL_SHA256,
    AndroidParityCtcBackend,
    verify_android_bundle,
)
from inference.profile import load_profile, resolve_model_path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ARTIFACT_SMOKE = os.environ.get("RUN_ASR_ARTIFACT_SMOKE") == "1"


@pytest.mark.artifact
@pytest.mark.skipif(
    not RUN_ARTIFACT_SMOKE,
    reason="set RUN_ASR_ARTIFACT_SMOKE=1 inside the Android-parity image",
)
def test_packaged_android_int8_artifact_contract_and_runtime() -> None:
    profile = load_profile(
        REPO_ROOT
        / "inference/onnxruntime/profiles/swahili-exp41-ctc-android-int8.yaml"
    )
    model_path = resolve_model_path(profile, require_exists=False)
    if not model_path.exists():
        pytest.skip(f"local Android model artifact is absent: {model_path}")

    verified = verify_android_bundle(model_path)
    assert verified["model_sha256"] == ANDROID_MODEL_SHA256

    backend = AndroidParityCtcBackend(profile, model_path)
    try:
        metadata = backend.metadata()
        assert metadata["adapter"] == "android_ctc"
        assert metadata["batch_size_required"] == 1
        assert metadata["num_threads"] == 1
        assert metadata["validation"] == {
            "artifact_hash": "passed",
            "onnx_contract": "passed",
            "quantized_operators": "passed",
            "runtime_version": "passed",
        }
    finally:
        backend.close()
