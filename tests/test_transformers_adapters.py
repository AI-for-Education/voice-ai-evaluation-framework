from __future__ import annotations

import importlib
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inference.profile import ModelProfile, parse_profile


def _profile(
    *,
    adapter: str = "ctc",
    profile_id: str = "test-ctc",
    artifact: str = "test-model",
    language: str | None = None,
    processor_mode: str = "auto",
    generation_kwargs: dict[str, Any] | None = None,
    decoding_strategy: str | None = None,
    ctc_lm_kwargs: dict[str, Any] | None = None,
    hallucination_guard: dict[str, Any] | None = None,
    long_form: dict[str, Any] | None = None,
    audio: dict[str, Any] | None = None,
) -> ModelProfile:
    return parse_profile(
        {
            "id": profile_id,
            "framework": "transformers",
            "adapter": adapter,
            "artifact": artifact,
            "language": language,
            "task": "transcribe",
            "output_units": "orthographic",
            "loader": {
                "processor_mode": processor_mode,
                "local_files_only": True,
                "trust_remote_code": False,
                "torch_dtype": "auto",
            },
            "decoding": {
                "strategy": decoding_strategy
                or ("greedy" if adapter == "ctc" else "generate"),
                "generation_kwargs": generation_kwargs or {},
                **(
                    {"ctc_lm_kwargs": ctc_lm_kwargs}
                    if ctc_lm_kwargs is not None
                    else {}
                ),
                **(
                    {"hallucination_guard": hallucination_guard}
                    if hallucination_guard is not None
                    else {}
                ),
                **(
                    {"long_form": long_form}
                    if long_form is not None
                    else {}
                ),
            },
            **({"audio": audio} if audio is not None else {}),
        }
    )


class _FakeDevice:
    def __init__(self, spec: str) -> None:
        self.spec = spec
        self.type = spec.split(":", 1)[0]

    def __str__(self) -> str:
        return self.spec


class _FakeDType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"torch.{self.name}"


class _FakeTensor:
    def __init__(self, *, floating: bool = True) -> None:
        self.floating = floating
        self.to_calls: list[dict[str, Any]] = []

    def to(self, **kwargs: Any) -> _FakeTensor:
        self.to_calls.append(kwargs)
        return self


class _FakeLogits(_FakeTensor):
    def __init__(self, values: np.ndarray) -> None:
        super().__init__()
        self.values = values
        self.detach_calls = 0

    def detach(self) -> _FakeLogits:
        self.detach_calls += 1
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _FakeTokenizer:
    def __init__(self) -> None:
        self.target_languages: list[str] = []
        self.word_delimiter_token = "|"
        self.vocab = {"|": 0, "a": 1, "[UNK]": 2, "[PAD]": 3}

    def set_target_lang(self, language: str) -> None:
        self.target_languages.append(language)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)


class _FakeProcessor:
    def __init__(
        self,
        *,
        feature_extractor: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.feature_extractor = feature_extractor or types.SimpleNamespace(
            sampling_rate=16000
        )
        self.tokenizer = tokenizer or _FakeTokenizer()
        self.encoded = {"input_values": _FakeTensor()}
        self.decoded: list[str] = ["decoded"]
        self.encode_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.decode_calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.encode_calls.append((args, kwargs))
        return self.encoded

    def batch_decode(self, token_ids: Any, **kwargs: Any) -> list[str]:
        self.decode_calls.append((token_ids, kwargs))
        return self.decoded


class _FakeModel:
    def __init__(self, dtype: Any) -> None:
        self.dtype = dtype
        self.logits = object()
        self.generated_ids = object()
        self.to_calls: list[Any] = []
        self.eval_calls = 0
        self.forward_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.adapter_calls: list[tuple[str, dict[str, Any]]] = []
        self.config = types.SimpleNamespace(vocab_size=4)

    def parameters(self):
        yield types.SimpleNamespace(dtype=self.dtype)

    def to(self, device: Any) -> _FakeModel:
        self.to_calls.append(device)
        return self

    def eval(self) -> _FakeModel:
        self.eval_calls += 1
        return self

    def __call__(self, **kwargs: Any) -> Any:
        self.forward_calls.append(kwargs)
        return types.SimpleNamespace(logits=self.logits)

    def generate(self, **kwargs: Any) -> Any:
        self.generate_calls.append(kwargs)
        return self.generated_ids

    def load_adapter(self, language: str, **kwargs: Any) -> None:
        self.adapter_calls.append((language, kwargs))


@pytest.fixture
def fake_transformers_runtime(monkeypatch: pytest.MonkeyPatch):
    """Install minimal torch/transformers modules before importing the adapters."""
    state = types.SimpleNamespace()

    torch_module = types.ModuleType("torch")
    torch_module.float32 = _FakeDType("float32")
    torch_module.float16 = _FakeDType("float16")
    torch_module.bfloat16 = _FakeDType("bfloat16")
    torch_module.device = _FakeDevice
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
        empty_cache=lambda: None,
    )
    torch_module.is_floating_point = lambda value: bool(
        getattr(value, "floating", False)
    )
    state.argmax_result = object()
    state.argmax_calls = []

    def argmax(value: Any, *, dim: int) -> Any:
        state.argmax_calls.append((value, dim))
        return state.argmax_result

    torch_module.argmax = argmax

    @contextmanager
    def inference_mode():
        yield

    torch_module.inference_mode = inference_mode
    state.torch = torch_module

    state.auto_processor_calls = []
    state.auto_ctc_model_calls = []
    state.auto_seq2seq_model_calls = []
    state.feature_extractor_calls = []
    state.tokenizer_calls = []
    state.plain_processor_inits = []
    state.lm_processor_calls = []
    state.auto_processor_return = _FakeProcessor()
    state.auto_ctc_model_return = _FakeModel(torch_module.float32)
    state.auto_seq2seq_model_return = _FakeModel(torch_module.float32)
    state.feature_extractor_return = types.SimpleNamespace(sampling_rate=16000)
    state.tokenizer_return = _FakeTokenizer()

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            state.auto_processor_calls.append((args, kwargs))
            return state.auto_processor_return

    class AutoModelForCTC:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            state.auto_ctc_model_calls.append((args, kwargs))
            return state.auto_ctc_model_return

    class AutoModelForSpeechSeq2Seq:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            state.auto_seq2seq_model_calls.append((args, kwargs))
            return state.auto_seq2seq_model_return

    class Wav2Vec2FeatureExtractor:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            state.feature_extractor_calls.append((args, kwargs))
            return state.feature_extractor_return

    class Wav2Vec2CTCTokenizer:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            state.tokenizer_calls.append((args, kwargs))
            return state.tokenizer_return

    class Wav2Vec2Processor(_FakeProcessor):
        def __init__(self, *, feature_extractor: Any, tokenizer: Any) -> None:
            state.plain_processor_inits.append((feature_extractor, tokenizer))
            super().__init__(
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
            )

    class Wav2Vec2ProcessorWithLM(_FakeProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = types.SimpleNamespace(
                _alphabet=types.SimpleNamespace(labels=[" ", "a", "⁇", ""])
            )

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            state.lm_processor_calls.append((args, kwargs))
            return state.lm_processor_return

    state.lm_processor_return = Wav2Vec2ProcessorWithLM()

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoProcessor = AutoProcessor
    transformers_module.AutoModelForCTC = AutoModelForCTC
    transformers_module.AutoModelForSpeechSeq2Seq = AutoModelForSpeechSeq2Seq
    transformers_module.Wav2Vec2FeatureExtractor = Wav2Vec2FeatureExtractor
    transformers_module.Wav2Vec2CTCTokenizer = Wav2Vec2CTCTokenizer
    transformers_module.Wav2Vec2Processor = Wav2Vec2Processor
    transformers_module.Wav2Vec2ProcessorWithLM = Wav2Vec2ProcessorWithLM
    state.transformers = transformers_module

    pyctcdecode_module = types.ModuleType("pyctcdecode")
    pyctcdecode_module.__version__ = "0.5.0"
    pyctcdecode_alphabet = types.ModuleType("pyctcdecode.alphabet")
    pyctcdecode_alphabet.BLANK_TOKEN_PTN = re.compile(r"^\[PAD\]$")
    pyctcdecode_alphabet.UNK_TOKEN = "⁇"
    pyctcdecode_alphabet.UNK_TOKEN_PTN = re.compile(r"^\[UNK\]$")
    pyctcdecode_module.alphabet = pyctcdecode_alphabet
    kenlm_module = types.ModuleType("kenlm")
    kenlm_module.__version__ = "0.3.0"

    adapter_modules = (
        "inference.transformers.adapters.ctc",
        "inference.transformers.adapters.speech_seq2seq",
    )
    for name in adapter_modules:
        sys.modules.pop(name, None)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "pyctcdecode", pyctcdecode_module)
    monkeypatch.setitem(sys.modules, "pyctcdecode.alphabet", pyctcdecode_alphabet)
    monkeypatch.setitem(sys.modules, "kenlm", kenlm_module)

    yield state

    for name in adapter_modules:
        sys.modules.pop(name, None)


def _create_lm_snapshot(model_path: Path) -> None:
    for relative in (
        "alphabet.json",
        "config.json",
        "preprocessor_config.json",
        "vocab.json",
        "language_model/5gram_mms.bin",
        "language_model/unigrams.txt",
        "language_model/attrs.json",
    ):
        path = model_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"lm" if path.suffix == ".bin" else b"{}")


def _bookbot_lm_profile(*, decoder_workers: int = 0) -> ModelProfile:
    return _profile(
        profile_id="bookbot-orthographic-ctc-5gram",
        artifact="bookbot-lm",
        language="sw",
        processor_mode="wav2vec2_with_lm",
        decoding_strategy="beam_search",
        ctc_lm_kwargs={
            "beam_width": 100,
            "alpha": 0.5,
            "beta": 1.5,
            "unk_score_offset": -10.0,
            "lm_score_boundary": True,
            "n_best": 1,
            "decoder_workers": decoder_workers,
        },
    )


@pytest.mark.parametrize(
    ("profile", "module_name", "class_name"),
    [
        (
            _profile(adapter="ctc"),
            "inference.transformers.adapters.ctc",
            "TransformersCTCBackend",
        ),
        (
            _profile(adapter="speech_seq2seq"),
            "inference.transformers.adapters.speech_seq2seq",
            "TransformersSpeechSeq2SeqBackend",
        ),
    ],
)
def test_factory_selects_profile_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: ModelProfile,
    module_name: str,
    class_name: str,
) -> None:
    calls: list[tuple[ModelProfile, Path]] = []

    class DummyBackend:
        def __init__(self, selected_profile: ModelProfile, model_path: Path) -> None:
            calls.append((selected_profile, model_path))

    adapter_module = types.ModuleType(module_name)
    setattr(adapter_module, class_name, DummyBackend)
    monkeypatch.setitem(sys.modules, module_name, adapter_module)

    from inference.transformers.factory import create_backend

    backend = create_backend(profile, tmp_path)

    assert isinstance(backend, DummyBackend)
    assert calls == [(profile, tmp_path)]


def test_bookbot_plain_mode_constructs_processor_without_auto_processor(
    fake_transformers_runtime: Any,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    ctc = importlib.import_module("inference.transformers.adapters.ctc")
    model_path = tmp_path / "bookbot-ctc"
    model_path.mkdir()
    profile = _profile(
        profile_id="bookbot-orthographic-ctc",
        artifact="bookbot-ctc",
        processor_mode="wav2vec2_plain",
    )

    backend = ctc.TransformersCTCBackend(
        profile,
        model_path,
        device=state.torch.device("cpu"),
    )

    assert state.auto_processor_calls == []
    assert state.feature_extractor_calls == [
        ((str(model_path),), {"local_files_only": True})
    ]
    assert state.tokenizer_calls == [
        ((str(model_path),), {"local_files_only": True})
    ]
    assert state.plain_processor_inits == [
        (state.feature_extractor_return, state.tokenizer_return)
    ]
    assert backend.metadata()["processor_mode"] == "wav2vec2_plain"


def test_mms_configures_tokenizer_and_model_language_adapter(
    fake_transformers_runtime: Any,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    ctc = importlib.import_module("inference.transformers.adapters.ctc")
    model_path = tmp_path / "mms-1b-all"
    model_path.mkdir()
    tokenizer = _FakeTokenizer()
    state.auto_processor_return = _FakeProcessor(tokenizer=tokenizer)
    state.auto_ctc_model_return = _FakeModel(state.torch.float32)
    profile = _profile(
        profile_id="mms-1b-all-swh",
        artifact="mms-1b-all",
        language="swh",
    )

    backend = ctc.TransformersCTCBackend(
        profile,
        model_path,
        device=state.torch.device("cpu"),
    )

    assert tokenizer.target_languages == ["swh"]
    assert state.auto_ctc_model_return.adapter_calls == [
        ("swh", {"local_files_only": True})
    ]
    assert backend.metadata()["language_adapter"] == "swh"


def test_mms_requires_language_before_loading_processor_or_model(
    fake_transformers_runtime: Any,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    ctc = importlib.import_module("inference.transformers.adapters.ctc")
    model_path = tmp_path / "mms-1b-all"
    model_path.mkdir()
    profile = _profile(
        profile_id="mms-1b-all",
        artifact="mms-1b-all",
        language=None,
    )

    with pytest.raises(ctc.ProfileError, match="require a language adapter code"):
        ctc.TransformersCTCBackend(
            profile,
            model_path,
            device=state.torch.device("cpu"),
        )

    assert state.auto_processor_calls == []
    assert state.auto_ctc_model_calls == []


def test_auto_ctc_loading_and_greedy_argmax_batch_decoding(
    fake_transformers_runtime: Any,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    ctc = importlib.import_module("inference.transformers.adapters.ctc")
    model_path = tmp_path / "w2v-bert-2.0-swahili"
    model_path.mkdir()
    processor = _FakeProcessor()
    processor.decoded = [" first transcript ", "second transcript"]
    state.auto_processor_return = processor
    state.auto_ctc_model_return = _FakeModel(state.torch.float32)
    profile = _profile(
        profile_id="w2v-bert-2.0-swahili-asr",
        artifact="w2v-bert-2.0-swahili",
        language="sw",
    )

    backend = ctc.TransformersCTCBackend(
        profile,
        model_path,
        device=state.torch.device("cpu"),
    )
    decoded = backend._decode(
        [np.zeros(4, dtype=np.float32), np.ones(5, dtype=np.float32)]
    )

    assert state.auto_processor_calls == [
        (
            (str(model_path),),
            {"local_files_only": True, "trust_remote_code": False},
        )
    ]
    assert state.auto_ctc_model_calls == [
        (
            (str(model_path),),
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "torch_dtype": state.torch.float32,
            },
        )
    ]
    assert processor.encode_calls[0][1] == {
        "sampling_rate": 16000,
        "return_tensors": "pt",
        "padding": True,
    }
    assert state.auto_ctc_model_return.forward_calls == [processor.encoded]
    assert state.argmax_calls == [(state.auto_ctc_model_return.logits, -1)]
    assert processor.decode_calls == [(state.argmax_result, {})]
    assert decoded == ["first transcript", "second transcript"]


def test_bookbot_lm_receives_full_logits_and_records_provenance(
    fake_transformers_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    ctc = importlib.import_module("inference.transformers.adapters.ctc")
    model_path = tmp_path / "bookbot-lm"
    _create_lm_snapshot(model_path)
    logits_values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    logits = _FakeLogits(logits_values)
    state.auto_ctc_model_return.logits = logits
    state.lm_processor_return.decoded = types.SimpleNamespace(
        text=[" first LM transcript ", "second LM transcript"]
    )

    class FakePool:
        def __init__(self) -> None:
            self.closed = False
            self.joined = False

        def close(self) -> None:
            self.closed = True

        def join(self) -> None:
            self.joined = True

    pool = FakePool()
    pool_calls: list[int] = []

    def create_pool(workers: int) -> FakePool:
        pool_calls.append(workers)
        return pool

    monkeypatch.setattr(ctc, "_create_decoder_pool", create_pool)
    backend = ctc.TransformersCTCBackend(
        _bookbot_lm_profile(decoder_workers=1),
        model_path,
        device=state.torch.device("cpu"),
    )
    decoded = backend._decode(
        [np.zeros(4, dtype=np.float32), np.ones(5, dtype=np.float32)]
    )

    assert state.lm_processor_calls == [
        (
            (str(model_path),),
            {"local_files_only": True, "trust_remote_code": False},
        )
    ]
    assert pool_calls == [1]
    assert state.argmax_calls == []
    assert logits.detach_calls == 1
    assert logits.to_calls == [{"device": "cpu", "dtype": state.torch.float32}]
    assert state.lm_processor_return.decode_calls == [
        (
            logits_values,
            {
                "beam_width": 100,
                "alpha": 0.5,
                "beta": 1.5,
                "unk_score_offset": -10.0,
                "lm_score_boundary": True,
                "n_best": 1,
                "pool": pool,
            },
        )
    ]
    assert decoded == ["first LM transcript", "second LM transcript"]

    metadata = backend.metadata()
    assert metadata["strategy"] == "beam_search"
    assert metadata["processor_class"] == "Wav2Vec2ProcessorWithLM"
    assert metadata["decoder_class"] == "SimpleNamespace"
    assert metadata["language_model_path"] == str(
        model_path / "language_model/5gram_mms.bin"
    )
    assert metadata["language_model_size_bytes"] == 2
    assert metadata["beam_width"] == 100
    assert metadata["alpha"] == 0.5
    assert metadata["beta"] == 1.5
    assert metadata["unk_score_offset"] == -10.0
    assert metadata["lm_score_boundary"] is True
    assert metadata["n_best"] == 1
    assert metadata["decoder_workers"] == 1
    assert metadata["pyctcdecode_version"] == "0.5.0"
    assert metadata["kenlm_version"] == "0.3.0"
    assert metadata["device"] == "cpu"
    assert metadata["effective_torch_dtype"] == "float32"

    backend.close()
    assert pool.closed is True
    assert pool.joined is True


def test_bookbot_lm_returns_one_ordered_result_per_input(
    fake_transformers_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    ctc = importlib.import_module("inference.transformers.adapters.ctc")
    model_path = tmp_path / "bookbot-lm"
    _create_lm_snapshot(model_path)
    state.auto_ctc_model_return.logits = _FakeLogits(
        np.zeros((2, 3, 4), dtype=np.float32)
    )
    state.lm_processor_return.decoded = types.SimpleNamespace(text=["first", "second"])
    backend = ctc.TransformersCTCBackend(
        _bookbot_lm_profile(),
        model_path,
        device=state.torch.device("cpu"),
    )

    def fake_load(path: str, target_sr: int):
        value = 1.0 if path == "first.wav" else 2.0
        return np.full(4, value, dtype=np.float32), target_sr, value, False

    monkeypatch.setattr(ctc, "load_audio_and_resample", fake_load)
    rows = backend.transcribe_batch(["first.wav", "second.wav"])

    assert [(row.audio_filepath, row.pred_text) for row in rows] == [
        ("first.wav", "first"),
        ("second.wav", "second"),
    ]


def test_bookbot_lm_missing_files_fail_before_processor_or_audio(
    fake_transformers_runtime: Any,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    ctc = importlib.import_module("inference.transformers.adapters.ctc")
    model_path = tmp_path / "bookbot-lm"
    model_path.mkdir()

    with pytest.raises(RuntimeError, match="decoder files are missing"):
        ctc.TransformersCTCBackend(
            _bookbot_lm_profile(),
            model_path,
            device=state.torch.device("cpu"),
        )

    assert state.lm_processor_calls == []
    assert state.auto_ctc_model_calls == []


def test_bookbot_lm_missing_package_fails_before_processor_or_audio(
    fake_transformers_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    ctc = importlib.import_module("inference.transformers.adapters.ctc")
    model_path = tmp_path / "bookbot-lm"
    _create_lm_snapshot(model_path)
    real_import_module = ctc.importlib.import_module

    def import_without_kenlm(name: str) -> Any:
        if name == "kenlm":
            raise ModuleNotFoundError("No module named 'kenlm'", name="kenlm")
        return real_import_module(name)

    monkeypatch.setattr(ctc.importlib, "import_module", import_without_kenlm)
    with pytest.raises(RuntimeError, match="missing package: kenlm"):
        ctc.TransformersCTCBackend(
            _bookbot_lm_profile(),
            model_path,
            device=state.torch.device("cpu"),
        )

    assert state.lm_processor_calls == []
    assert state.auto_ctc_model_calls == []


def test_bookbot_lm_rejects_non_lm_processor_and_alphabet_mismatch(
    fake_transformers_runtime: Any,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    ctc = importlib.import_module("inference.transformers.adapters.ctc")
    model_path = tmp_path / "bookbot-lm"
    _create_lm_snapshot(model_path)
    state.lm_processor_return = _FakeProcessor()

    with pytest.raises(RuntimeError, match="did not load an LM-aware"):
        ctc.TransformersCTCBackend(
            _bookbot_lm_profile(),
            model_path,
            device=state.torch.device("cpu"),
        )

    state.lm_processor_return = state.transformers.Wav2Vec2ProcessorWithLM()
    state.lm_processor_return.decoder._alphabet.labels = [" ", "b", "⁇", ""]
    with pytest.raises(RuntimeError, match="must match in length and order"):
        ctc.TransformersCTCBackend(
            _bookbot_lm_profile(),
            model_path,
            device=state.torch.device("cpu"),
        )


def test_speech_seq2seq_generate_receives_language_task_and_no_timestamps(
    fake_transformers_runtime: Any,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    seq2seq = importlib.import_module(
        "inference.transformers.adapters.speech_seq2seq"
    )
    model_path = tmp_path / "whisper-large-v2"
    model_path.mkdir()
    processor = _FakeProcessor()
    processor.encoded = {"input_features": _FakeTensor()}
    processor.decoded = [" habari "]
    state.auto_processor_return = processor
    state.auto_seq2seq_model_return = _FakeModel(state.torch.float32)
    profile = _profile(
        adapter="speech_seq2seq",
        profile_id="whisper-large-v2-sw",
        artifact="whisper-large-v2",
        language="sw",
        generation_kwargs={"num_beams": 3},
    )

    backend = seq2seq.TransformersSpeechSeq2SeqBackend(
        profile,
        model_path,
        device=state.torch.device("cpu"),
    )
    decoded = backend._decode([np.zeros(8, dtype=np.float32)])

    assert state.auto_seq2seq_model_calls == [
        (
            (str(model_path),),
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "torch_dtype": state.torch.float32,
            },
        )
    ]
    assert state.auto_seq2seq_model_return.generate_calls == [
        {
            "input_features": processor.encoded["input_features"],
            "num_beams": 3,
            "return_timestamps": False,
            "task": "transcribe",
            "language": "sw",
        }
    ]
    assert processor.decode_calls == [
        (state.auto_seq2seq_model_return.generated_ids, {"skip_special_tokens": True})
    ]
    assert decoded == ["habari"]


def test_speech_seq2seq_long_form_uses_untruncated_timestamp_path(
    fake_transformers_runtime: Any,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    seq2seq = importlib.import_module(
        "inference.transformers.adapters.speech_seq2seq"
    )
    model_path = tmp_path / "whisper-large"
    model_path.mkdir()
    processor = _FakeProcessor()
    processor.encoded = {
        "input_features": _FakeTensor(),
        "attention_mask": _FakeTensor(floating=False),
    }
    processor.decoded = [" long transcription "]
    state.auto_processor_return = processor
    state.auto_seq2seq_model_return = _FakeModel(state.torch.float32)
    profile = _profile(
        adapter="speech_seq2seq",
        profile_id="whisper-large-sw",
        artifact="whisper-large",
        language="sw",
        generation_kwargs={"do_sample": False, "return_timestamps": False},
        long_form={"strategy": "timestamp", "threshold_seconds": 30.0},
    )
    backend = seq2seq.TransformersSpeechSeq2SeqBackend(
        profile,
        model_path,
        device=state.torch.device("cpu"),
    )

    decoded = backend._decode(
        [np.zeros(31 * 16000, dtype=np.float32)],
        long_form=True,
    )

    assert processor.encode_calls[0][1] == {
        "sampling_rate": 16000,
        "return_tensors": "pt",
        "truncation": False,
        "padding": "longest",
        "return_attention_mask": True,
    }
    assert state.auto_seq2seq_model_return.generate_calls == [
        {
            "input_features": processor.encoded["input_features"],
            "attention_mask": processor.encoded["attention_mask"],
            "do_sample": False,
            "return_timestamps": True,
            "task": "transcribe",
            "language": "sw",
        }
    ]
    assert decoded == ["long transcription"]


def test_speech_seq2seq_routes_mixed_batch_and_preserves_original_order(
    fake_transformers_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    seq2seq = importlib.import_module(
        "inference.transformers.adapters.speech_seq2seq"
    )
    model_path = tmp_path / "whisper-large-v2"
    model_path.mkdir()
    profile = _profile(
        adapter="speech_seq2seq",
        profile_id="whisper-large-v2-sw",
        artifact="whisper-large-v2",
        language="sw",
        generation_kwargs={"do_sample": False, "return_timestamps": False},
        long_form={"strategy": "timestamp", "threshold_seconds": 30.0},
    )
    backend = seq2seq.TransformersSpeechSeq2SeqBackend(
        profile,
        model_path,
        device=state.torch.device("cpu"),
    )
    paths = ["long.wav", "short.wav", "exactly-30.wav", "longer.wav"]
    markers = {path: index + 1 for index, path in enumerate(paths)}
    durations = {
        "long.wav": 30.001,
        "short.wav": 4.0,
        "exactly-30.wav": 30.0,
        "longer.wav": 58.635,
    }
    monkeypatch.setattr(
        seq2seq,
        "load_audio_and_resample",
        lambda path, target_sr: (
            np.full(8, markers[path], dtype=np.float32),
            target_sr,
            durations[path],
            False,
        ),
    )
    decode_calls: list[tuple[bool, list[int]]] = []

    def fake_decode(audio_batch: list[np.ndarray], *, long_form: bool = False):
        marker_values = [int(audio[0]) for audio in audio_batch]
        decode_calls.append((long_form, marker_values))
        prefix = "long" if long_form else "short"
        return [f"{prefix}-{marker}" for marker in marker_values]

    monkeypatch.setattr(backend, "_decode", fake_decode)

    rows = backend.transcribe_batch(paths)

    assert decode_calls == [(False, [2, 3]), (True, [1, 4])]
    assert [row.audio_filepath for row in rows] == paths
    assert [row.pred_text for row in rows] == [
        "long-1",
        "short-2",
        "short-3",
        "long-4",
    ]
    assert backend.metadata()["long_form"] == {
        "strategy": "timestamp",
        "threshold_seconds": 30.0,
    }



def test_speech_seq2seq_chunks_profile_scoped_long_audio_in_memory(
    fake_transformers_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    seq2seq = importlib.import_module(
        "inference.transformers.adapters.speech_seq2seq"
    )
    model_path = tmp_path / "paza-whisper"
    model_path.mkdir()
    profile = _profile(
        adapter="speech_seq2seq",
        profile_id="paza-whisper",
        artifact="paza-whisper",
        language="sw",
        generation_kwargs={"do_sample": False, "return_timestamps": False},
        audio={
            "maximum_seconds": 30,
            "long_audio_strategy": "sequential_chunks",
            "chunk_seconds": 30,
            "overlap_seconds": 0,
        },
    )
    backend = seq2seq.TransformersSpeechSeq2SeqBackend(
        profile, model_path, device=state.torch.device("cpu")
    )
    monkeypatch.setattr(
        seq2seq,
        "load_audio_and_resample",
        lambda path, target_sr: (
            np.zeros(65 * target_sr, dtype=np.float32),
            target_sr,
            65.0,
            False,
        ),
    )
    decode_calls: list[tuple[bool, list[int]]] = []

    def fake_decode(audio_batch: list[np.ndarray], *, long_form: bool = False):
        decode_calls.append((long_form, [len(audio) for audio in audio_batch]))
        return ["moja", "mbili", "tatu"]

    monkeypatch.setattr(backend, "_decode", fake_decode)

    rows = backend.transcribe_batch(["long.wav"])

    assert decode_calls == [(False, [30 * 16000, 30 * 16000, 5 * 16000])]
    assert rows[0].pred_text == "moja mbili tatu"
    assert backend.metadata()["chunking"] == {
        "maximum_seconds": 30.0,
        "long_audio_strategy": "sequential_chunks",
        "chunk_seconds": 30.0,
        "overlap_seconds": 0.0,
    }
    assert backend.metadata()["chunking_stats"] == {
        "long_audio_files": 1,
        "chunks_generated": 3,
    }

@pytest.mark.parametrize(
    ("adapter", "module_name", "class_name"),
    [
        (
            "ctc",
            "inference.transformers.adapters.ctc",
            "TransformersCTCBackend",
        ),
        (
            "speech_seq2seq",
            "inference.transformers.adapters.speech_seq2seq",
            "TransformersSpeechSeq2SeqBackend",
        ),
    ],
)
def test_transformers_batch_failure_returns_one_error_row_per_input(
    fake_transformers_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter: str,
    module_name: str,
    class_name: str,
) -> None:
    state = fake_transformers_runtime
    module = importlib.import_module(module_name)
    model_path = tmp_path / f"{adapter}-model"
    model_path.mkdir()
    profile = _profile(
        adapter=adapter,
        artifact=model_path.name,
        language="sw" if adapter == "speech_seq2seq" else None,
    )
    backend_class = getattr(module, class_name)
    backend = backend_class(profile, model_path, device=state.torch.device("cpu"))

    def fake_load(path: str, target_sr: int):
        if path == "unreadable.wav":
            raise RuntimeError("cannot decode")
        return np.zeros(8, dtype=np.float32), target_sr, 0.5, False

    def fail_decode(audio_batch, *, long_form=False):
        raise RuntimeError("model failed")

    monkeypatch.setattr(module, "load_audio_and_resample", fake_load)
    monkeypatch.setattr(backend, "_decode", fail_decode)

    rows = backend.transcribe_batch(
        ["first.wav", "unreadable.wav", "second.wav"]
    )

    assert [row.audio_filepath for row in rows] == [
        "first.wav",
        "unreadable.wav",
        "second.wav",
    ]
    assert rows[0].error == "inference_failed: model failed"
    assert rows[1].error == "failed_to_read_audio: cannot decode"
    assert rows[2].error == "inference_failed: model failed"



def test_whisper_guard_caps_wps_and_preserves_raw_prediction(
    fake_transformers_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = fake_transformers_runtime
    seq2seq = importlib.import_module(
        "inference.transformers.adapters.speech_seq2seq"
    )
    model_path = tmp_path / "whisper-large"
    model_path.mkdir()
    processor = _FakeProcessor()
    processor.encoded = {"input_features": _FakeTensor()}
    processor.decoded = [" ".join(f"word{index}" for index in range(20))]
    state.auto_processor_return = processor
    profile = _profile(
        adapter="speech_seq2seq",
        profile_id="whisper-large-sw",
        artifact="whisper-large",
        language="sw",
        hallucination_guard={
            "max_words_per_second": 8.0,
            "repeated_phrase_min_words": 5,
            "repeated_phrase_max_words": 8,
            "repeated_phrase_repetitions": 3,
        },
    )
    backend = seq2seq.TransformersSpeechSeq2SeqBackend(
        profile,
        model_path,
        device=state.torch.device("cpu"),
    )
    monkeypatch.setattr(
        seq2seq,
        "load_audio_and_resample",
        lambda path, target_sr: (np.zeros(8, dtype=np.float32), target_sr, 1.0, False),
    )

    row = backend.transcribe_batch(["sample.wav"])[0]

    assert row.pred_text == " ".join(f"word{index}" for index in range(8))
    assert row.raw_pred_text == processor.decoded[0]
    assert row.to_row()["raw_pred_text"] == processor.decoded[0]
    assert backend.metadata()["hallucination_guard"]["max_words_per_second"] == 8.0


def test_whisper_guard_stops_consecutive_repeated_phrase(
    fake_transformers_runtime: Any,
) -> None:
    seq2seq = importlib.import_module(
        "inference.transformers.adapters.speech_seq2seq"
    )
    profile = _profile(
        adapter="speech_seq2seq",
        hallucination_guard={
            "max_words_per_second": 8.0,
            "repeated_phrase_min_words": 5,
            "repeated_phrase_max_words": 8,
            "repeated_phrase_repetitions": 3,
        },
    )
    phrase = "one two three four five"
    text = f"intro {phrase} {phrase} {phrase} trailing words"

    guarded, changed = seq2seq.apply_hallucination_guard(
        text,
        duration=10.0,
        config=profile.decoding.hallucination_guard,
    )

    assert changed is True
    assert guarded == f"intro {phrase}"
