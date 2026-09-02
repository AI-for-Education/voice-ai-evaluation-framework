from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference.multimodal.adapters import phi4_audio, qwen_omni_audio
from inference.multimodal.adapters.phi4_audio import Phi4AudioBackend
from inference.multimodal.adapters.qwen_omni_audio import QwenOmniAudioBackend
from inference.profile import parse_profile


def _profile(adapter: str):
    is_phi4 = adapter == "phi4_audio"
    return parse_profile(
        {
            "profile_schema_version": 2,
            "inference_setup_id": f"{adapter}-test",
            "inference_library": "multimodal",
            "adapter": adapter,
            "artifact": "model",
            "language": "sw",
            "task": "transcribe",
            "output_units": "orthographic",
            "prompt": "Transcribe to Swahili. Output only text.",
            "loader": {
                "processor_mode": "auto",
                "local_files_only": True,
                "trust_remote_code": is_phi4,
                "torch_dtype": "bfloat16",
                "attention_implementation": "sdpa",
            },
            "decoding": {
                "strategy": "generate",
                "generation_kwargs": {"max_new_tokens": 8},
            },
            "audio": {
                "maximum_seconds": 1,
                "long_audio_strategy": "sequential_chunks",
                "chunk_seconds": 1,
                "overlap_seconds": 0,
            },
            "hardware": (
                {
                    "memory_strategy": "cpu_disk_offload",
                    "minimum_gpu_memory_gib": 12,
                    "gpu_max_memory_gib": 13,
                    "cpu_max_memory_gib": 12,
                    "output_mode": "text_only",
                }
                if is_phi4
                else {
                    "memory_strategy": "large_gpu_only",
                    "minimum_gpu_memory_gib": 40,
                    "output_mode": "text_only",
                }
            ),
        }
    )


class _PhiProcessor:
    def __init__(self) -> None:
        self.audio_processor = SimpleNamespace(sampling_rate=4)
        self.audio_lengths: list[int] = []

    def __call__(self, *, text, audios, return_tensors):
        assert "<|audio_1|>" in text
        assert return_tensors == "pt"
        self.audio_lengths.append(len(audios[0][0]))
        return {"input_ids": torch.tensor([[1, 2]]), "audio": torch.ones(1)}

    def batch_decode(self, ids, **kwargs):
        return ["habari"]


class _PhiModel:
    hf_device_map = {"model.embed_tokens": 0, "model.layers.31": "cpu"}

    def __init__(self) -> None:
        self.calls = []

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2, 3]])


def _write_phi_processor_files(model_path: Path) -> None:
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "auto_map": {
                    "AutoConfig": "configuration_phi4mm.Phi4MMConfig",
                    "AutoTokenizer": "remote/repo--tokenizer.LocalTokenizer",
                }
            }
        ),
        encoding="utf-8",
    )
    (model_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "auto_map": {
                    "AutoProcessor": (
                        "microsoft/Phi-4-multimodal-instruct--"
                        "processing_phi4mm.Phi4MMProcessor"
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    (model_path / "processing_phi4mm.py").write_text("# local code\n", encoding="utf-8")
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_phi4_loader_uses_local_overlay_and_bounded_offload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_phi_processor_files(model_path)
    processor = _PhiProcessor()
    model = _PhiModel()
    calls = {}

    def load_processor(path, **kwargs):
        calls["processor"] = (Path(path), kwargs)
        overlay = json.loads(
            (Path(path) / "preprocessor_config.json").read_text(encoding="utf-8")
        )
        assert overlay["auto_map"]["AutoProcessor"] == (
            "processing_phi4mm.Phi4MMProcessor"
        )
        return processor

    def load_model(path, **kwargs):
        calls["model"] = (Path(path), kwargs)
        return model

    monkeypatch.setattr(phi4_audio, "cuda_memory_gib", lambda device: 16.0)
    monkeypatch.setattr(phi4_audio.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(phi4_audio.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(phi4_audio.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(
        phi4_audio.GenerationConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(phi4_audio, "move_inputs", lambda encoded, **kwargs: encoded)

    backend = Phi4AudioBackend(
        _profile("phi4_audio"),
        model_path,
        device=torch.device("cuda:0"),
        offload_root=tmp_path / "offload",
    )
    try:
        loader_kwargs = calls["model"][1]
        assert loader_kwargs["device_map"] == "auto"
        assert loader_kwargs["max_memory"] == {0: "13GiB", "cpu": "12GiB"}
        assert loader_kwargs["offload_state_dict"] is True
        assert loader_kwargs["torch_dtype"] is torch.bfloat16
        assert loader_kwargs["_attn_implementation"] == "sdpa"
        assert calls["processor"][1]["local_files_only"] is True
        assert calls["processor"][1]["trust_remote_code"] is True

        monkeypatch.setattr(
            phi4_audio,
            "load_audio_and_resample",
            lambda path, sr: (np.ones(10, dtype=np.float32), sr, 2.5, False),
        )
        rows = backend.transcribe_batch(["long.wav"])
        assert rows[0].pred_text == "habari habari habari"
        assert processor.audio_lengths == [4, 4, 2]
        metadata = backend.metadata()
        assert metadata["hardware"]["memory_strategy"] == "cpu_disk_offload"
        assert metadata["generation"]["actual_call_kwargs"]["max_new_tokens"] == 8
        assert "<|audio_1|>" in metadata["rendered_prompt"]
    finally:
        work_dir = backend._work_dir
        backend.close()
    assert work_dir is not None
    assert not work_dir.exists()


def test_qwen_current_hardware_is_blocked_before_model_loading(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    monkeypatch.setattr(qwen_omni_audio, "cuda_memory_gib", lambda device: 16.0)
    monkeypatch.setattr(qwen_omni_audio.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        qwen_omni_audio.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("processor loading must not start"),
    )
    monkeypatch.setattr(
        qwen_omni_audio.Qwen2_5OmniForConditionalGeneration,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("model loading must not start"),
    )

    with pytest.raises(RuntimeError, match="not runnable.*16.0 GiB detected"):
        QwenOmniAudioBackend(
            _profile("qwen_omni_audio"),
            model_path,
            device=torch.device("cuda:0"),
        )


class _QwenProcessor:
    def __init__(self) -> None:
        self.feature_extractor = SimpleNamespace(sampling_rate=4)
        self.template = None

    def apply_chat_template(self, messages, **kwargs):
        self.template = messages
        return "qwen audio prompt"

    def __call__(self, **kwargs):
        return {"input_ids": torch.tensor([[7, 8]]), "features": torch.ones(1)}

    def batch_decode(self, ids, **kwargs):
        return ["jambo"]


class _QwenModel:
    hf_device_map = {"": 0}

    def __init__(self) -> None:
        self.talker_disabled = False
        self.calls = []

    def disable_talker(self):
        self.talker_disabled = True

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[9]])


def test_qwen_large_gpu_path_is_text_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    processor = _QwenProcessor()
    model = _QwenModel()

    monkeypatch.setattr(qwen_omni_audio, "cuda_memory_gib", lambda device: 48.0)
    monkeypatch.setattr(qwen_omni_audio.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        qwen_omni_audio.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: processor,
    )
    monkeypatch.setattr(
        qwen_omni_audio.Qwen2_5OmniForConditionalGeneration,
        "from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        qwen_omni_audio, "move_inputs", lambda encoded, **kwargs: encoded
    )
    monkeypatch.setattr(
        qwen_omni_audio,
        "load_audio_and_resample",
        lambda path, sr: (np.ones(2, dtype=np.float32), sr, 0.5, False),
    )

    backend = QwenOmniAudioBackend(
        _profile("qwen_omni_audio"),
        model_path,
        device=torch.device("cuda:0"),
    )
    rows = backend.transcribe_batch(["audio.wav"])

    assert rows[0].pred_text == "jambo"
    assert model.talker_disabled is True
    assert model.calls[0]["return_audio"] is False
    metadata = backend.metadata()
    assert metadata["attention_implementation"] == "sdpa"
    assert metadata["generation"]["actual_call_kwargs"]["return_audio"] is False
    assert metadata["generation"]["effective_generation_config"]["max_new_tokens"] == 8
    assert backend.metadata()["output"] == {
        "mode": "text_only",
        "talker_disabled": True,
        "return_audio": False,
    }
