from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference.profile import load_profile, resolve_model_path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ARTIFACT_SMOKE = os.environ.get("RUN_ASR_ARTIFACT_SMOKE") == "1"


def _artifact_or_skip(profile_path: Path) -> tuple[object, Path]:
    profile = load_profile(profile_path)
    model_path = resolve_model_path(profile, require_exists=False)
    if not model_path.exists():
        pytest.skip(f"local model artifact is absent: {model_path}")
    return profile, model_path


@pytest.mark.artifact
@pytest.mark.skipif(
    not RUN_ARTIFACT_SMOKE,
    reason="set RUN_ASR_ARTIFACT_SMOKE=1 to load local model weights",
)
def test_nemo_artifact_loads() -> None:
    profile, model_path = _artifact_or_skip(
        REPO_ROOT / "inference/nemo/profiles/swahili-exp41-ctc.yaml"
    )
    from inference.nemo.backend import NemoBackend

    backend = NemoBackend(profile=profile, model_path=model_path, batch_size=1)
    try:
        assert backend.metadata()["framework"] == "nemo"
    finally:
        backend.close()


@pytest.mark.artifact
@pytest.mark.skipif(
    not RUN_ARTIFACT_SMOKE,
    reason="set RUN_ASR_ARTIFACT_SMOKE=1 to load local model weights",
)
def test_transformers_artifact_loads() -> None:
    profile, model_path = _artifact_or_skip(
        REPO_ROOT
        / "inference/transformers/profiles/bookbot-orthographic-ctc.yaml"
    )
    from inference.transformers.factory import create_backend

    backend = create_backend(profile, model_path)
    try:
        assert backend.metadata()["framework"] == "transformers"
    finally:
        backend.close()


@pytest.mark.artifact
@pytest.mark.skipif(
    not RUN_ARTIFACT_SMOKE,
    reason="set RUN_ASR_ARTIFACT_SMOKE=1 to load local model weights",
)
def test_bookbot_5gram_artifact_loads() -> None:
    profile, model_path = _artifact_or_skip(
        REPO_ROOT
        / "inference/transformers/profiles/bookbot-orthographic-ctc-5gram.yaml"
    )
    from inference.transformers.factory import create_backend

    backend = create_backend(profile, model_path)
    try:
        metadata = backend.metadata()
        assert metadata["strategy"] == "beam_search"
        assert metadata["processor_class"] == "Wav2Vec2ProcessorWithLM"
        assert metadata["decoder_class"] == "BeamSearchDecoderCTC"
        assert metadata["language_model_size_bytes"] > 0
        assert metadata["pyctcdecode_version"] == "0.5.0"
        assert metadata["kenlm_version"] == "0.3.0"
    finally:
        backend.close()


@pytest.mark.artifact
@pytest.mark.skipif(
    not RUN_ARTIFACT_SMOKE,
    reason="set RUN_ASR_ARTIFACT_SMOKE=1 to load local model weights",
)
def test_sherpa_onnx_artifact_loads() -> None:
    profile, model_path = _artifact_or_skip(
        REPO_ROOT
        / "inference/sherpa_onnx/profiles/zipformer-streaming-robust-sw-v4.yaml"
    )
    from inference.sherpa_onnx.backend import SherpaOnnxOnlineTransducerBackend

    backend = SherpaOnnxOnlineTransducerBackend(
        profile=profile,
        model_path=model_path,
        num_threads=1,
    )
    try:
        metadata = backend.metadata()
        assert metadata["framework"] == "sherpa_onnx"
        assert metadata["decoding_strategy"] == "greedy_search"
    finally:
        backend.close()


@pytest.mark.artifact
@pytest.mark.skipif(
    not RUN_ARTIFACT_SMOKE,
    reason="set RUN_ASR_ARTIFACT_SMOKE=1 to load local model weights",
)
def test_gemma4_multimodal_artifact_loads() -> None:
    profile, model_path = _artifact_or_skip(
        REPO_ROOT / "inference/multimodal/profiles/gemma-4-E2B-sw.yaml"
    )
    from inference.multimodal.adapters import Gemma4AudioBackend

    backend = Gemma4AudioBackend(profile=profile, model_path=model_path)
    try:
        metadata = backend.metadata()
        assert metadata["framework"] == "multimodal"
        assert metadata["adapter"] == "gemma4_audio"
        assert metadata["effective_torch_dtype"] == "bfloat16"
    finally:
        backend.close()
