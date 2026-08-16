from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from inference.multimodal.adapters import gemma4_audio
from inference.multimodal.adapters.gemma4_audio import Gemma4AudioBackend
from inference.profile import parse_profile


def _profile(*, seconds: int = 1):
    return parse_profile(
        {
            "id": "gemma-test",
            "framework": "multimodal",
            "adapter": "gemma4_audio",
            "artifact": "gemma-model",
            "language": "sw",
            "task": "transcribe",
            "output_units": "orthographic",
            "prompt": "Transcribe in Swahili. Output only the transcription.",
            "loader": {
                "processor_mode": "auto",
                "local_files_only": True,
                "trust_remote_code": False,
                "torch_dtype": "bfloat16",
            },
            "decoding": {
                "strategy": "generate",
                "generation_kwargs": {"max_new_tokens": 8},
            },
            "audio": {
                "maximum_seconds": seconds,
                "long_audio_strategy": "sequential_chunks",
                "chunk_seconds": seconds,
                "overlap_seconds": 0,
            },
        }
    )


class _FakeProcessor:
    def __init__(self) -> None:
        self.feature_extractor = SimpleNamespace(sampling_rate=4)
        self.template_calls = []
        self.audio_lengths: list[int] = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return "rendered prompt with <|audio|>"

    def __call__(self, *, text, audio, return_tensors):
        assert text == "rendered prompt with <|audio|>"
        assert return_tensors == "pt"
        self.audio_lengths.append(len(audio[0]))
        return {"input_ids": torch.tensor([[11, 12]]), "input_features": torch.ones(1)}

    def decode(self, tokens, *, skip_special_tokens):
        return "jambo" if skip_special_tokens else "<|turn>model\njambo<turn|>"

    def parse_response(self, response, *, prefix):
        assert response.startswith("<|turn>model")
        assert prefix.shape == (1, 2)
        return {"role": "model", "content": [{"type": "text", "text": "jambo"}]}


class _FakeModel:
    def __init__(self) -> None:
        self.generation_calls = []

    def parameters(self):
        yield torch.zeros(1, dtype=torch.bfloat16)

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generation_calls.append(kwargs)
        return torch.tensor([[11, 12, 99]])


def _backend(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "gemma-model"
    model_path.mkdir()
    processor = _FakeProcessor()
    model = _FakeModel()
    load_calls = {}

    def load_processor(path, **kwargs):
        load_calls["processor"] = (path, kwargs)
        return processor

    def load_model(path, **kwargs):
        load_calls["model"] = (path, kwargs)
        return model

    monkeypatch.setattr(gemma4_audio.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(
        gemma4_audio.AutoModelForMultimodalLM,
        "from_pretrained",
        load_model,
    )
    backend = Gemma4AudioBackend(
        _profile(),
        model_path,
        device=torch.device("cuda:0"),
    )
    # Unit tests exercise the adapter without exposing a real CUDA device.
    backend._move_inputs = lambda encoded: dict(encoded)
    return backend, processor, model, load_calls


def test_gemma_loader_is_offline_bfloat16_and_uses_fixed_prompt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend, processor, _, load_calls = _backend(monkeypatch, tmp_path)

    assert load_calls["processor"][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert load_calls["model"][1]["dtype"] is torch.bfloat16
    assert load_calls["model"][1]["device_map"] == {"": "cuda:0"}
    messages, template_kwargs = processor.template_calls[0]
    assert messages[0]["content"][0]["text"] == backend.profile.prompt
    assert messages[0]["content"][1] == {"type": "audio"}
    assert template_kwargs["enable_thinking"] is False


def test_long_audio_is_chunked_and_rejoined_as_one_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend, processor, model, _ = _backend(monkeypatch, tmp_path)
    monkeypatch.setattr(
        gemma4_audio,
        "load_audio_and_resample",
        lambda path, sample_rate: (np.ones(10, dtype=np.float32), sample_rate, 2.5, False),
    )

    results = backend.transcribe_batch(["long.wav"])

    assert len(results) == 1
    assert results[0].audio_filepath == "long.wav"
    assert results[0].duration == 2.5
    assert results[0].pred_text == "jambo jambo jambo"
    assert processor.audio_lengths == [4, 4, 2]
    assert len(model.generation_calls) == 3
    assert all(call["max_new_tokens"] == 8 for call in model.generation_calls)
    assert backend.metadata()["long_audio_files"] == 1
    assert backend.metadata()["chunks_generated"] == 3


def test_audio_read_failure_preserves_order_and_returns_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    backend, _, _, _ = _backend(monkeypatch, tmp_path)

    def fake_load(path, sample_rate):
        if path == "bad.wav":
            raise ValueError("broken file")
        return np.ones(2, dtype=np.float32), sample_rate, 0.5, False

    monkeypatch.setattr(gemma4_audio, "load_audio_and_resample", fake_load)
    rows = backend.transcribe_batch(["bad.wav", "good.wav"])

    assert [row.audio_filepath for row in rows] == ["bad.wav", "good.wav"]
    assert rows[0].error == "failed_to_read_audio: broken file"
    assert rows[1].pred_text == "jambo"
