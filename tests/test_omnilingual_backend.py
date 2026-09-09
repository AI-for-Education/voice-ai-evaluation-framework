from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from inference.omnilingual import backend as backend_module
from inference.omnilingual.backend import OmnilingualBackend
from inference.omnilingual.cache import (
    configure_model_cache,
    prepared_model_identity,
    write_prepared_marker,
)
from inference.profile import load_profile, parse_profile


REPO_ROOT = Path(__file__).resolve().parents[1]


def _profile(adapter: str):
    return parse_profile(
        {
            "profile_schema_version": 2,
            "inference_setup_id": f"omnilingual-{adapter}-test",
            "inference_library": "omnilingual",
            "adapter": adapter,
            "artifact": (
                "omniASR_LLM_3B_v2" if adapter == "llm" else "omniASR_CTC_3B_v2"
            ),
            "language": "swh_Latn",
            "task": "transcribe",
            "output_units": "orthographic",
            "loader": {
                "processor_mode": "auto",
                "local_files_only": True,
                "trust_remote_code": False,
                "torch_dtype": "bfloat16",
            },
            "decoding": {
                "strategy": (
                    "llm_beam_search" if adapter == "llm" else "ctc_greedy"
                ),
                "generation_kwargs": {},
            },
            "audio": {
                "maximum_seconds": 1,
                "long_audio_strategy": "sequential_chunks",
                "chunk_seconds": 1,
                "overlap_seconds": 0,
            },
        }
    )


class _Pipeline:
    def __init__(self, **kwargs) -> None:
        self.init_kwargs = kwargs
        self.calls: list[tuple[list[dict], dict]] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return [f"chunk-{index + 1}" for index in range(len(audio))]


def test_llm_backend_chunks_and_passes_language_conditioning(monkeypatch) -> None:
    pipeline = _Pipeline()
    monkeypatch.setattr(
        backend_module,
        "load_audio_and_resample",
        lambda path, sample_rate: (
            np.ones(40000, dtype=np.float32),
            sample_rate,
            2.5,
            True,
        ),
    )
    backend = OmnilingualBackend(
        _profile("llm"),
        batch_size=1,
        device=torch.device("cpu"),
        pipeline_factory=lambda **kwargs: (
            pipeline.__dict__.update(init_kwargs=kwargs) or pipeline
        ),
    )

    rows = backend.transcribe_batch(["audio.wav"])

    assert rows[0].pred_text == "chunk-1 chunk-2 chunk-3"
    audio, kwargs = pipeline.calls[0]
    assert [len(item["waveform"]) for item in audio] == [16000, 16000, 8000]
    assert kwargs == {"lang": ["swh_Latn"] * 3, "batch_size": 1}
    assert pipeline.init_kwargs["model_card"] == "omniASR_LLM_3B_v2"
    assert pipeline.init_kwargs["dtype"] is torch.bfloat16
    assert backend.metadata()["statistics"] == {
        "files_seen": 1,
        "chunks_generated": 3,
        "long_audio_files": 1,
        "resampled_files": 1,
    }


def test_ctc_backend_does_not_pass_language_conditioning(monkeypatch) -> None:
    pipeline = _Pipeline()
    monkeypatch.setattr(
        backend_module,
        "load_audio_and_resample",
        lambda path, sample_rate: (
            np.ones(8000, dtype=np.float32),
            sample_rate,
            0.5,
            False,
        ),
    )
    backend = OmnilingualBackend(
        _profile("ctc"),
        device=torch.device("cpu"),
        pipeline_factory=lambda **kwargs: pipeline,
    )

    rows = backend.transcribe_batch(["audio.wav"])

    assert rows[0].pred_text == "chunk-1"
    assert pipeline.calls[0][1]["lang"] is None
    assert backend.metadata()["language_conditioning"]["enabled"] is False


def test_owner_pipeline_loader_uses_fairseq2_mmap(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def load_model(card, **kwargs):
        calls["model"] = (card, kwargs)
        return "model"

    def load_tokenizer(card):
        calls["tokenizer"] = card
        return "tokenizer"

    def pipeline_factory(**kwargs):
        calls["pipeline"] = kwargs
        return "pipeline"

    monkeypatch.setattr(
        backend_module,
        "_owner_components",
        lambda: (load_model, load_tokenizer, pipeline_factory),
    )

    pipeline = backend_module.load_owner_pipeline(_profile("llm"), torch.device("cpu"))

    assert pipeline == "pipeline"
    assert calls["model"] == (
        "omniASR_LLM_3B_v2",
        {"device": torch.device("cpu"), "dtype": torch.bfloat16, "mmap": True},
    )
    assert calls["tokenizer"] == "omniASR_LLM_3B_v2"
    assert calls["pipeline"] == {
        "model_card": None,
        "model": "model",
        "tokenizer": "tokenizer",
        "device": torch.device("cpu"),
        "dtype": torch.bfloat16,
    }


def test_prepared_cache_marker_is_model_specific(monkeypatch, tmp_path: Path) -> None:
    profile = _profile("ctc")
    monkeypatch.setenv("ASR_MODEL_ROOT", str(tmp_path))
    cache_path = configure_model_cache(profile, create=True)
    (cache_path / "asset.pt").write_bytes(b"checkpoint")

    marker = write_prepared_marker(profile, cache_path)
    identity = prepared_model_identity(profile, cache_path)

    assert marker.is_file()
    assert identity["kind"] == "fairseq2_asset_card"
    assert identity["artifact"] == "omniASR_CTC_3B_v2"
    assert identity["prepared_cache"]["cache_inventory"]["file_count"] == 1

    write_prepared_marker(profile, cache_path)
    identity = prepared_model_identity(profile, cache_path)
    assert identity["prepared_cache"]["cache_inventory"]["file_count"] == 1


def test_tracked_omnilingual_profiles_validate() -> None:
    profiles = [
        REPO_ROOT
        / "inference/omnilingual/profiles/omniasr-llm-3b-v2-swh-latn.yaml",
        REPO_ROOT / "inference/omnilingual/profiles/omniasr-ctc-3b-v2-swh.yaml",
    ]

    loaded = [
        load_profile(path, expected_inference_library="omnilingual")
        for path in profiles
    ]

    assert [profile.adapter for profile in loaded] == ["llm", "ctc"]
