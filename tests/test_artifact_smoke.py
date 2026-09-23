from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference.profile import load_profile, resolve_model_path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ARTIFACT_SMOKE = os.environ.get("RUN_ASR_ARTIFACT_SMOKE") == "1"
pytestmark = [
    pytest.mark.artifact,
    pytest.mark.skipif(
        not RUN_ARTIFACT_SMOKE,
        reason="set RUN_ASR_ARTIFACT_SMOKE=1 to load local model weights",
    ),
]


def _artifact_or_skip(profile_path: Path) -> tuple[object, Path]:
    profile = load_profile(profile_path)
    model_path = resolve_model_path(profile, require_exists=False)
    if not model_path.exists():
        pytest.skip(f"local model artifact is absent: {model_path}")
    return profile, model_path


def test_nemo_artifact_loads() -> None:
    profile, model_path = _artifact_or_skip(
        REPO_ROOT / "inference/nemo/profiles/swahili-exp41-ctc.yaml"
    )
    from inference.nemo.backend import NemoBackend

    backend = NemoBackend(profile=profile, model_path=model_path, batch_size=1)
    try:
        assert backend.metadata()["inference_library"] == "nemo"
    finally:
        backend.close()


def test_transformers_artifact_loads() -> None:
    profile, model_path = _artifact_or_skip(
        REPO_ROOT
        / "inference/transformers/profiles/bookbot-orthographic-ctc.yaml"
    )
    from inference.transformers.factory import create_backend

    backend = create_backend(profile, model_path)
    try:
        assert backend.metadata()["inference_library"] == "transformers"
    finally:
        backend.close()


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
        assert metadata["inference_library"] == "sherpa_onnx"
        assert metadata["decoding_strategy"] == "greedy_search"
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("profile_name", "artifact_format"),
    [
        ("zipformer-streaming-robust-sw-v4-onnx-int8.yaml", "onnx"),
        ("zipformer-streaming-robust-sw-v4-ort-int8.yaml", "ort"),
    ],
)
def test_sherpa_quantized_artifact_loads(
    profile_name: str,
    artifact_format: str,
) -> None:
    profile, model_path = _artifact_or_skip(
        REPO_ROOT / "inference/sherpa_onnx/profiles" / profile_name
    )
    from inference.sherpa_onnx.backend import SherpaOnnxOnlineTransducerBackend

    backend = SherpaOnnxOnlineTransducerBackend(
        profile=profile,
        model_path=model_path,
        num_threads=1,
    )
    try:
        metadata = backend.metadata()
        assert metadata["artifact_format"] == artifact_format
        assert metadata["artifact_precision"] == "int8"
    finally:
        backend.close()


def test_bookbot_native_torchscript_owner_sample() -> None:
    profile, model_path = _artifact_or_skip(
        REPO_ROOT
        / "inference/torch/profiles/zipformer-streaming-robust-sw-v4-torchscript.yaml"
    )
    sample_path = model_path / "test_waves/sample1.wav"
    if not sample_path.is_file():
        pytest.skip(f"BookBot owner sample is absent: {sample_path}")

    from inference.torch.backend import TorchScriptStreamingTransducerBackend

    backend = TorchScriptStreamingTransducerBackend(
        profile=profile,
        model_path=model_path,
        num_threads=1,
    )
    try:
        rows = backend.transcribe_batch([str(sample_path)])
        assert len(rows) == 1
        assert rows[0].error is None
        assert rows[0].pred_text == (
            "wɑʃiɑɑᵐɓɑɔwɑnɑiʃihɑsɑkɑtikɑɛnɛɔlɑmɑʃɑɾikikɑtikɑ"
            "ufɑlmɛhuɔwɛnjɛutɑʄiɾiwɑmɑfutɑ"
        )
    finally:
        backend.close()


def test_gemma4_multimodal_artifact_loads() -> None:
    profile, model_path = _artifact_or_skip(
        REPO_ROOT / "inference/multimodal/profiles/gemma-4-E2B-sw.yaml"
    )
    from inference.multimodal.adapters import Gemma4AudioBackend

    backend = Gemma4AudioBackend(profile=profile, model_path=model_path)
    try:
        metadata = backend.metadata()
        assert metadata["inference_library"] == "multimodal"
        assert metadata["adapter"] == "gemma4_audio"
        assert metadata["effective_torch_dtype"] == "bfloat16"
    finally:
        backend.close()
