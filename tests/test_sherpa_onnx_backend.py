from __future__ import annotations

import sys
import types
import importlib.metadata
from pathlib import Path

import numpy as np

from inference.profile import parse_profile


def _profile(strategy: str = "greedy_search"):
    decoding: dict[str, object] = {"strategy": strategy}
    if strategy == "modified_beam_search":
        decoding["transducer_search_kwargs"] = {"max_active_paths": 4}
    return parse_profile(
        {
            "id": "zipformer-streaming-robust-sw-v4",
            "framework": "sherpa_onnx",
            "adapter": "online_transducer",
            "artifact": "zipformer",
            "language": "sw",
            "task": "transcribe",
            "output_units": "phoneme",
            "output_notation": "ipa",
            "output_inventory": "bookbot_gruut_sw_v1",
            "loader": {
                "processor_mode": "auto",
                "local_files_only": True,
                "trust_remote_code": False,
                "torch_dtype": "auto",
            },
            "decoding": decoding,
        }
    )


def _model_files(path: Path) -> None:
    path.mkdir()
    for name in ("encoder-model.onnx", "decoder-model.onnx", "joiner-model.onnx"):
        (path / name).write_bytes(b"onnx")
    (path / "tokens.txt").write_text("<eps> 0\na 1\n", encoding="utf-8")


def test_sherpa_backend_uses_cuda_and_preserves_batch_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "zipformer"
    _model_files(model_path)

    class FakeStream:
        def __init__(self) -> None:
            self.calls = []
            self.finished = False

        def accept_waveform(self, sample_rate, samples) -> None:
            self.calls.append((sample_rate, len(samples)))

        def input_finished(self) -> None:
            self.finished = True

    class FakeRecognizer:
        def __init__(self) -> None:
            self.streams = []
            self.decode_calls = []

        def create_stream(self):
            stream = FakeStream()
            self.streams.append(stream)
            return stream

        def is_ready(self, stream) -> bool:
            return not hasattr(stream, "decoded")

        def decode_streams(self, streams) -> None:
            self.decode_calls.append(list(streams))
            for stream in streams:
                stream.decoded = True

        def get_result(self, stream) -> str:
            return f" result-{self.streams.index(stream)} "

    recognizer = FakeRecognizer()
    constructor_calls = []

    class OnlineRecognizer:
        @staticmethod
        def from_transducer(**kwargs):
            constructor_calls.append(kwargs)
            return recognizer

    fake_module = types.SimpleNamespace(
        __version__="test",
        OnlineRecognizer=OnlineRecognizer,
        get_available_providers=lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_module)

    from inference.sherpa_onnx import backend as backend_module

    monkeypatch.setattr(
        backend_module,
        "load_audio_and_resample",
        lambda path, target_sr: (
            np.ones(4, dtype=np.float32),
            target_sr,
            0.25,
            False,
        ),
    )
    backend = backend_module.SherpaOnnxOnlineTransducerBackend(
        _profile(),
        model_path,
    )
    rows = backend.transcribe_batch(["first.wav", "second.wav"])

    assert constructor_calls[0]["provider"] == "cuda"
    assert constructor_calls[0]["decoding_method"] == "greedy_search"
    assert [(row.audio_filepath, row.pred_text) for row in rows] == [
        ("first.wav", "result-0"),
        ("second.wav", "result-1"),
    ]
    assert all(stream.finished for stream in recognizer.streams)
    assert backend.metadata()["device"] == "cuda:0"


def test_sherpa_modified_beam_passes_search_config_to_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "zipformer"
    _model_files(model_path)
    constructor_calls = []

    class OnlineRecognizer:
        @staticmethod
        def from_transducer(**kwargs):
            constructor_calls.append(kwargs)
            return types.SimpleNamespace()

    fake_module = types.SimpleNamespace(
        __version__="test",
        OnlineRecognizer=OnlineRecognizer,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_module)

    from inference.sherpa_onnx import backend as backend_module

    backend = backend_module.SherpaOnnxOnlineTransducerBackend(
        _profile("modified_beam_search"),
        model_path,
    )

    assert constructor_calls[0]["decoding_method"] == "modified_beam_search"
    assert constructor_calls[0]["max_active_paths"] == 4
    assert backend.metadata()["decoding_strategy"] == "modified_beam_search"
    assert backend.metadata()["max_active_paths"] == 4
    assert backend.metadata()["recognizer_call"]["arguments"][
        "max_active_paths"
    ] == 4


def test_sherpa_cuda_wheel_selects_cuda_without_provider_introspection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "zipformer"
    _model_files(model_path)
    constructor_calls = []

    class OnlineRecognizer:
        @staticmethod
        def from_transducer(**kwargs):
            constructor_calls.append(kwargs)
            return types.SimpleNamespace()

    fake_module = types.SimpleNamespace(
        __version__="1.13.5+cuda12.cudnn9.onnxruntime1.27.1",
        OnlineRecognizer=OnlineRecognizer,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_module)
    real_version = importlib.metadata.version

    def fake_version(name: str) -> str:
        if name == "sherpa-onnx":
            return fake_module.__version__
        return real_version(name)

    from inference.sherpa_onnx import backend as backend_module

    monkeypatch.setattr(backend_module.importlib.metadata, "version", fake_version)
    backend = backend_module.SherpaOnnxOnlineTransducerBackend(
        _profile(),
        model_path,
    )

    assert constructor_calls[0]["provider"] == "cuda"
    assert backend.metadata()["device"] == "cuda:0"
