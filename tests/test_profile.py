from __future__ import annotations

import copy
from pathlib import Path

import pytest

from inference.profile import (
    ProfileError,
    load_profile,
    parse_profile,
    resolve_model_path,
)


def _sequential_chunking() -> dict[str, object]:
    return {
        "maximum_seconds": 30,
        "long_audio_strategy": "sequential_chunks",
        "chunk_seconds": 30,
        "overlap_seconds": 0,
    }


BASE_PROFILE = {
    "id": "example",
    "framework": "transformers",
    "adapter": "ctc",
    "artifact": "example-model",
    "language": "sw",
    "task": "transcribe",
    "output_units": "orthographic",
    "loader": {
        "processor_mode": "auto",
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": "auto",
    },
    "decoding": {"strategy": "greedy", "generation_kwargs": {}},
}

MULTIMODAL_PROFILE = {
    "id": "gemma-test",
    "framework": "multimodal",
    "adapter": "gemma4_audio",
    "artifact": "gemma-model",
    "language": "sw",
    "task": "transcribe",
    "output_units": "orthographic",
    "prompt": "Transcribe the speech in Swahili. Output only the transcription.",
    "loader": {
        "processor_mode": "auto",
        "local_files_only": True,
        "trust_remote_code": False,
        "torch_dtype": "bfloat16",
    },
    "decoding": {
        "strategy": "generate",
        "generation_kwargs": {"do_sample": False, "max_new_tokens": 32},
    },
    "audio": _sequential_chunking(),
}


def _lm_profile() -> dict[str, object]:
    profile = copy.deepcopy(BASE_PROFILE)
    profile["loader"]["processor_mode"] = "wav2vec2_with_lm"
    profile["decoding"] = {
        "strategy": "beam_search",
        "ctc_lm_kwargs": {
            "beam_width": 100,
            "alpha": 0.5,
            "beta": 1.5,
            "unk_score_offset": -10.0,
            "lm_score_boundary": True,
            "n_best": 1,
        },
    }
    return profile


def _transducer_profile(
    strategy: str = "greedy_search",
) -> dict[str, object]:
    decoding: dict[str, object] = {"strategy": strategy}
    if strategy == "modified_beam_search":
        decoding["transducer_search_kwargs"] = {"max_active_paths": 4}
    return {
        "id": f"zipformer-{strategy}",
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


# Core profile schema


def test_parse_profile_rejects_unknown_keys() -> None:
    profile = dict(BASE_PROFILE, surprise=True)
    with pytest.raises(ProfileError, match="Unknown profile key"):
        parse_profile(profile)


@pytest.mark.parametrize(
    "artifact",
    [
        "/absolute/model",
        r"C:\absolute\model",
        r"C:model",
        "../outside",
        "models/../../outside",
    ],
)
def test_parse_profile_rejects_unsafe_artifacts(artifact: str) -> None:
    profile = dict(BASE_PROFILE, artifact=artifact)
    with pytest.raises(ProfileError, match="artifact must be a relative path"):
        parse_profile(profile)


def test_profile_rejects_adapter_framework_mismatch() -> None:
    profile = dict(BASE_PROFILE, framework="nemo")
    with pytest.raises(ProfileError, match="Unsupported adapter"):
        parse_profile(profile)


def test_plain_processor_mode_is_ctc_only() -> None:
    profile = dict(
        BASE_PROFILE,
        adapter="speech_seq2seq",
        loader=dict(BASE_PROFILE["loader"], processor_mode="wav2vec2_plain"),
        decoding={"strategy": "generate"},
    )
    with pytest.raises(ProfileError, match="only valid for transformers/ctc"):
        parse_profile(profile)


# CTC decoding profiles


def test_valid_ctc_lm_profile_has_typed_defaults() -> None:
    profile = parse_profile(_lm_profile())

    assert profile.loader.processor_mode == "wav2vec2_with_lm"
    assert profile.decoding.strategy == "beam_search"
    assert profile.decoding.generation_kwargs == {}
    assert profile.decoding.ctc_lm_kwargs is not None
    assert profile.decoding.ctc_lm_kwargs.beam_width == 100
    assert profile.decoding.ctc_lm_kwargs.alpha == 0.5
    assert profile.decoding.ctc_lm_kwargs.decoder_workers == 1
    assert profile.to_dict()["decoding"]["ctc_lm_kwargs"]["beam_width"] == 100


def test_greedy_profile_metadata_shape_remains_unchanged() -> None:
    profile = parse_profile(BASE_PROFILE)

    assert "ctc_lm_kwargs" not in profile.to_dict()["decoding"]
    assert "output_notation" not in profile.to_dict()
    assert "output_inventory" not in profile.to_dict()
    assert "parameter_evidence" not in profile.to_dict()
    assert "attention_implementation" not in profile.to_dict()["loader"]


def test_phoneme_profile_requires_explicit_ipa_inventory() -> None:
    missing = dict(BASE_PROFILE, output_units="phoneme")
    with pytest.raises(ProfileError, match="output_notation: ipa"):
        parse_profile(missing)

    profile = parse_profile(
        dict(
            BASE_PROFILE,
            output_units="phoneme",
            output_notation="ipa",
            output_inventory="bookbot_gruut_sw_v1",
        )
    )
    assert profile.output_notation == "ipa"
    assert profile.output_inventory == "bookbot_gruut_sw_v1"
    assert profile.to_dict()["output_inventory"] == "bookbot_gruut_sw_v1"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("beam_width", 0, "positive integer"),
        ("beam_width", True, "positive integer"),
        ("alpha", float("inf"), "finite number"),
        ("beta", "1.5", "finite number"),
        ("unk_score_offset", float("nan"), "finite number"),
        ("lm_score_boundary", 1, "boolean"),
        ("n_best", 0, "positive integer"),
        ("decoder_workers", -1, "non-negative integer"),
        ("decoder_workers", False, "non-negative integer"),
    ],
)
def test_ctc_lm_kwargs_are_typed_before_model_loading(
    key: str,
    value: object,
    message: str,
) -> None:
    profile = _lm_profile()
    profile["decoding"]["ctc_lm_kwargs"][key] = value

    with pytest.raises(ProfileError, match=message):
        parse_profile(profile)


def test_ctc_lm_configuration_cannot_leak_into_greedy_profile() -> None:
    profile = _lm_profile()
    profile["loader"]["processor_mode"] = "wav2vec2_plain"

    with pytest.raises(ProfileError, match="requires loader.processor_mode"):
        parse_profile(profile)

    profile = _lm_profile()
    profile["decoding"]["strategy"] = "greedy"
    with pytest.raises(ProfileError, match="only valid for CTC beam_search"):
        parse_profile(profile)


# Transducer search profiles


def test_modified_beam_search_has_typed_transducer_config() -> None:
    profile = parse_profile(_transducer_profile("modified_beam_search"))

    assert profile.decoding.strategy == "modified_beam_search"
    assert profile.decoding.transducer_search_kwargs is not None
    assert profile.decoding.transducer_search_kwargs.max_active_paths == 4
    assert profile.to_dict()["decoding"]["transducer_search_kwargs"] == {
        "max_active_paths": 4
    }


@pytest.mark.parametrize("value", [0, -1, True, "4"])
def test_modified_beam_search_rejects_invalid_max_active_paths(
    value: object,
) -> None:
    profile = _transducer_profile("modified_beam_search")
    profile["decoding"]["transducer_search_kwargs"]["max_active_paths"] = value

    with pytest.raises(ProfileError, match="positive integer"):
        parse_profile(profile)


def test_transducer_search_config_is_required_and_cannot_leak() -> None:
    profile = _transducer_profile("modified_beam_search")
    del profile["decoding"]["transducer_search_kwargs"]
    with pytest.raises(ProfileError, match="is required"):
        parse_profile(profile)

    profile = _transducer_profile()
    profile["decoding"]["transducer_search_kwargs"] = {"max_active_paths": 4}
    with pytest.raises(ProfileError, match="only valid"):
        parse_profile(profile)


# Speech-seq2seq and long-audio profiles


def test_timestamp_generation_is_disabled_for_initial_delivery() -> None:
    profile = dict(
        BASE_PROFILE,
        adapter="speech_seq2seq",
        decoding={
            "strategy": "generate",
            "generation_kwargs": {"return_timestamps": True},
        },
    )
    with pytest.raises(ProfileError, match="return_timestamps must be false"):
        parse_profile(profile)


def test_whisper_long_form_profile_is_explicit_and_typed() -> None:
    profile = parse_profile(
        dict(
            BASE_PROFILE,
            adapter="speech_seq2seq",
            decoding={
                "strategy": "generate",
                "generation_kwargs": {
                    "do_sample": False,
                    "return_timestamps": False,
                },
                "long_form": {
                    "strategy": "timestamp",
                    "threshold_seconds": 30,
                },
            },
        )
    )

    assert profile.decoding.long_form is not None
    assert profile.decoding.long_form.strategy == "timestamp"
    assert profile.decoding.long_form.threshold_seconds == 30.0
    assert profile.to_dict()["decoding"]["long_form"] == {
        "strategy": "timestamp",
        "threshold_seconds": 30.0,
    }


def test_speech_seq2seq_profile_accepts_explicit_audio_chunking() -> None:
    data = dict(
        BASE_PROFILE,
        adapter="speech_seq2seq",
        decoding={
            "strategy": "generate",
            "generation_kwargs": {"do_sample": False},
        },
        audio=_sequential_chunking(),
    )

    profile = parse_profile(data)

    assert profile.audio is not None
    assert profile.audio.maximum_seconds == 30.0
    assert profile.audio.chunk_seconds == 30.0


def test_speech_seq2seq_rejects_two_long_audio_strategies() -> None:
    data = dict(
        BASE_PROFILE,
        adapter="speech_seq2seq",
        decoding={
            "strategy": "generate",
            "generation_kwargs": {"do_sample": False},
            "long_form": {"strategy": "timestamp", "threshold_seconds": 30},
        },
        audio=_sequential_chunking(),
    )

    with pytest.raises(ProfileError, match="cannot be enabled together"):
        parse_profile(data)


def test_ctc_profile_still_rejects_audio_chunking() -> None:
    data = dict(
        BASE_PROFILE,
        audio=_sequential_chunking(),
    )

    with pytest.raises(ProfileError, match="audio is only valid"):
        parse_profile(data)


@pytest.mark.parametrize(
    ("long_form", "message"),
    [
        ({"strategy": "chunks", "threshold_seconds": 30}, "must be 'timestamp'"),
        ({"strategy": "timestamp", "threshold_seconds": 0}, "greater than zero"),
        ({"strategy": "timestamp", "threshold_seconds": True}, "finite number"),
    ],
)
def test_whisper_long_form_profile_rejects_invalid_values(
    long_form: dict[str, object],
    message: str,
) -> None:
    profile = dict(
        BASE_PROFILE,
        adapter="speech_seq2seq",
        decoding={
            "strategy": "generate",
            "generation_kwargs": {"do_sample": False},
            "long_form": long_form,
        },
    )

    with pytest.raises(ProfileError, match=message):
        parse_profile(profile)


def test_whisper_long_form_rejects_sampling() -> None:
    profile = dict(
        BASE_PROFILE,
        adapter="speech_seq2seq",
        decoding={
            "strategy": "generate",
            "generation_kwargs": {"do_sample": True},
            "long_form": {"strategy": "timestamp", "threshold_seconds": 30},
        },
    )

    with pytest.raises(ProfileError, match="do_sample to be false"):
        parse_profile(profile)


# Multimodal generation profiles


def test_multimodal_profile_has_frozen_prompt_and_audio_policy() -> None:
    profile = parse_profile(MULTIMODAL_PROFILE)

    assert profile.prompt == MULTIMODAL_PROFILE["prompt"]
    assert profile.audio is not None
    assert profile.audio.long_audio_strategy == "sequential_chunks"
    assert profile.audio.chunk_seconds == 30.0
    assert profile.decoding.generation_kwargs["do_sample"] is False
    assert profile.to_dict()["audio"]["overlap_seconds"] == 0.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"prompt": ""}, "non-empty prompt"),
        ({"audio": None}, "require an audio section"),
        (
            {"audio": dict(MULTIMODAL_PROFILE["audio"], overlap_seconds=1)},
            "overlap_seconds must be 0",
        ),
        (
            {"loader": dict(MULTIMODAL_PROFILE["loader"], torch_dtype="auto")},
            "requires loader.torch_dtype: bfloat16",
        ),
    ],
)
def test_multimodal_profile_rejects_unfrozen_behavior(
    change: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ProfileError, match=message):
        parse_profile(dict(MULTIMODAL_PROFILE, **change))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("num_beams", 0, "positive integer"),
        ("num_beams", "four", "positive integer"),
        ("max_new_tokens", 1.5, "positive integer"),
        ("do_sample", "no", "boolean"),
        ("early_stopping", "yes", "boolean"),
        ("length_penalty", "high", "finite number"),
        ("suppress_tokens", [1, "2"], "non-negative integers"),
    ],
)
def test_generation_kwargs_are_typed_before_model_loading(
    key: str,
    value: object,
    message: str,
) -> None:
    profile = dict(
        BASE_PROFILE,
        adapter="speech_seq2seq",
        decoding={"strategy": "generate", "generation_kwargs": {key: value}},
    )
    with pytest.raises(ProfileError, match=message):
        parse_profile(profile)


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("id", ["model"], "id is required"),
        (
            "framework",
            {"name": "transformers"},
            "inference_library is required",
        ),
        ("language", ["sw"], "language must be a string or null"),
        ("task", ["transcribe"], "task must be a string"),
    ],
)
def test_profile_rejects_non_scalar_schema_values(
    section: str,
    value: object,
    message: str,
) -> None:
    profile = dict(BASE_PROFILE, **{section: value})
    with pytest.raises(ProfileError, match=message):
        parse_profile(profile)


def test_loader_rejects_non_scalar_mode() -> None:
    profile = dict(
        BASE_PROFILE,
        loader=dict(BASE_PROFILE["loader"], processor_mode=["auto"]),
    )
    with pytest.raises(ProfileError, match="processor_mode must be a string"):
        parse_profile(profile)


# Parameter evidence and reproducibility


def test_parameter_evidence_is_optional_typed_and_serialized() -> None:
    profile_data = copy.deepcopy(BASE_PROFILE)
    profile_data["parameter_evidence"] = [
        {
            "applies_to": ["decoding.strategy"],
            "level": "exact",
            "source": "https://example.test/model-card",
            "rationale": "The model card defines greedy CTC decoding.",
        }
    ]

    profile = parse_profile(profile_data)

    assert profile.parameter_evidence[0].applies_to == ("decoding.strategy",)
    assert profile.to_dict()["parameter_evidence"] == profile_data["parameter_evidence"]


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        ([], "non-empty list"),
        (
            [
                {
                    "applies_to": ["decoding.missing"],
                    "level": "exact",
                    "source": "https://example.test/model-card",
                    "rationale": "Invalid path.",
                }
            ],
            "missing profile field",
        ),
        (
            [
                {
                    "applies_to": ["decoding.strategy"],
                    "level": "authoritative",
                    "source": "https://example.test/model-card",
                    "rationale": "Invalid level.",
                }
            ],
            "level must be one of",
        ),
        (
            [
                {
                    "applies_to": ["decoding.strategy"],
                    "level": "exact",
                    "source": "model-card",
                    "rationale": "Invalid source.",
                }
            ],
            "must use https",
        ),
    ],
)
def test_parameter_evidence_rejects_invalid_records(
    evidence: object,
    message: str,
) -> None:
    with pytest.raises(ProfileError, match=message):
        parse_profile(dict(BASE_PROFILE, parameter_evidence=evidence))


# Artifact resolution and tracked-profile integration


def test_resolve_model_path_stays_below_root(tmp_path: Path) -> None:
    artifact = tmp_path / "example-model"
    artifact.mkdir()
    profile = parse_profile(BASE_PROFILE)
    assert resolve_model_path(profile, model_root=tmp_path) == artifact.resolve()


def _result_affecting_profile_paths(payload: dict[str, object]) -> set[str]:
    paths = {"decoding.strategy"}

    def add_leaves(value: object, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                add_leaves(child, f"{prefix}.{key}")
        elif value is not None:
            paths.add(prefix)

    decoding = payload["decoding"]
    assert isinstance(decoding, dict)
    for section in (
        "generation_kwargs",
        "ctc_lm_kwargs",
        "transducer_search_kwargs",
        "long_form",
    ):
        value = decoding.get(section)
        if value:
            add_leaves(value, f"decoding.{section}")

    for section in ("audio", "hardware"):
        value = payload.get(section)
        if value:
            add_leaves(value, section)

    loader = payload["loader"]
    assert isinstance(loader, dict)
    if loader.get("torch_dtype") != "auto":
        paths.add("loader.torch_dtype")
    if loader.get("attention_implementation") is not None:
        paths.add("loader.attention_implementation")

    for field in ("prompt", "output_notation", "output_inventory"):
        if field in payload:
            paths.add(field)
    return paths


def test_all_tracked_profiles_are_valid_and_unique() -> None:
    repo = Path(__file__).resolve().parents[1]
    profile_paths = sorted((repo / "inference").glob("*/profiles/*.yaml"))
    assert len(profile_paths) == 23

    ids: set[str] = set()
    for path in profile_paths:
        profile = load_profile(path)
        payload = profile.to_dict()
        assert payload["profile_schema_version"] == 2
        assert "id" not in payload
        assert "framework" not in payload
        assert payload["inference_setup_id"] == profile.inference_setup_id
        assert payload["inference_library"] == profile.inference_library
        assert profile.framework == path.parents[1].name
        assert profile.pipeline_contract is not None
        assert profile.parameter_evidence
        for item in profile.parameter_evidence:
            if item.source.startswith("repo:"):
                source_path = repo / item.source.removeprefix("repo:")
                assert source_path.is_file(), (
                    f"{path} references missing evidence source: {source_path}"
                )
        assert "parameter_evidence" in profile.to_dict()
        covered = {
            field for item in profile.parameter_evidence for field in item.applies_to
        }
        missing = _result_affecting_profile_paths(profile.to_dict()) - covered
        assert not missing, f"{path} lacks parameter evidence for: {sorted(missing)}"
        assert profile.id not in ids
        ids.add(profile.id)


def test_v1_profile_normalizes_to_v2_and_conflicts_fail() -> None:
    profile = parse_profile(BASE_PROFILE)
    payload = profile.to_dict()

    assert payload["profile_schema_version"] == 2
    assert payload["inference_setup_id"] == BASE_PROFILE["id"]
    assert payload["inference_library"] == BASE_PROFILE["framework"]
    assert profile.id == profile.inference_setup_id
    assert profile.framework == profile.inference_library

    conflicting = dict(
        BASE_PROFILE,
        profile_schema_version=2,
        inference_setup_id="different",
    )
    with pytest.raises(ProfileError, match="Conflicting profile fields"):
        parse_profile(conflicting)


@pytest.mark.parametrize(
    ("baseline_name", "beam_name"),
    [
        ("whisper-large-sw.yaml", "whisper-large-sw-beam5.yaml"),
        ("whisper-large-v2-sw.yaml", "whisper-large-v2-sw-beam5.yaml"),
        (
            "paza-whisper-large-v3-turbo-sw.yaml",
            "paza-whisper-large-v3-turbo-sw-beam5.yaml",
        ),
    ],
)
def test_whisper_decoder_pairs_preserve_model_contract(
    baseline_name: str,
    beam_name: str,
) -> None:
    profiles_root = (
        Path(__file__).resolve().parents[1] / "inference/transformers/profiles"
    )
    baseline = load_profile(profiles_root / baseline_name)
    beam = load_profile(profiles_root / beam_name)

    contract = (
        "framework",
        "adapter",
        "artifact",
        "language",
        "task",
        "output_units",
        "output_notation",
        "output_inventory",
    )
    assert tuple(getattr(baseline, key) for key in contract) == tuple(
        getattr(beam, key) for key in contract
    )
    assert baseline.id != beam.id
    assert baseline.decoding.generation_kwargs["num_beams"] == 1
    assert beam.decoding.generation_kwargs["num_beams"] == 5
    assert baseline.decoding.generation_kwargs["do_sample"] is False
    assert beam.decoding.generation_kwargs["do_sample"] is False


def test_zipformer_decoder_pair_preserves_model_contract() -> None:
    profiles_root = (
        Path(__file__).resolve().parents[1] / "inference/sherpa_onnx/profiles"
    )
    greedy = load_profile(profiles_root / "zipformer-streaming-robust-sw-v4.yaml")
    beam = load_profile(
        profiles_root / "zipformer-streaming-robust-sw-v4-modified-beam4.yaml"
    )

    assert (
        greedy.framework,
        greedy.adapter,
        greedy.artifact,
        greedy.language,
        greedy.task,
        greedy.output_units,
        greedy.output_notation,
        greedy.output_inventory,
    ) == (
        beam.framework,
        beam.adapter,
        beam.artifact,
        beam.language,
        beam.task,
        beam.output_units,
        beam.output_notation,
        beam.output_inventory,
    )
    assert greedy.id != beam.id
    assert greedy.decoding.strategy == "greedy_search"
    assert beam.decoding.strategy == "modified_beam_search"
    assert beam.decoding.transducer_search_kwargs is not None
    assert beam.decoding.transducer_search_kwargs.max_active_paths == 4


def test_every_local_transformers_artifact_has_a_profile() -> None:
    repo = Path(__file__).resolve().parents[1]
    models_root = repo / "inference/transformers/models"
    local_artifacts = {
        path.name
        for path in models_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    profiled_artifacts = {
        load_profile(path).artifact
        for path in (repo / "inference/transformers/profiles").glob("*.yaml")
    }

    assert local_artifacts <= profiled_artifacts, (
        "Local Transformers model folder(s) have no tracked profile: "
        + ", ".join(sorted(local_artifacts - profiled_artifacts))
    )


# Post-decoding guardrails and dedicated multimodal requirements


def test_post_decoding_guard_is_rejected_for_seq2seq() -> None:
    profile = dict(
        BASE_PROFILE,
        adapter="speech_seq2seq",
        decoding={
            "strategy": "generate",
            "generation_kwargs": {"return_timestamps": False},
            "hallucination_guard": {
                "max_words_per_second": 8.0,
                "repeated_phrase_min_words": 5,
                "repeated_phrase_max_words": 8,
                "repeated_phrase_repetitions": 3,
            },
        },
    )

    with pytest.raises(ProfileError, match="Unknown decoding key.*hallucination_guard"):
        parse_profile(profile)


def test_hallucination_guard_is_rejected_for_ctc() -> None:
    profile = copy.deepcopy(BASE_PROFILE)
    profile["decoding"]["hallucination_guard"] = {
        "max_words_per_second": 8.0,
        "repeated_phrase_min_words": 5,
        "repeated_phrase_max_words": 8,
        "repeated_phrase_repetitions": 3,
    }

    with pytest.raises(ProfileError, match="Unknown decoding key.*hallucination_guard"):
        parse_profile(profile)


def test_phi4_profile_scopes_trusted_code_and_offload_policy() -> None:
    profile = parse_profile(
        {
            "id": "phi4-test",
            "framework": "multimodal",
            "adapter": "phi4_audio",
            "artifact": "phi4-model",
            "language": "sw",
            "task": "transcribe",
            "output_units": "orthographic",
            "prompt": "Transcribe to Swahili.",
            "loader": {
                "processor_mode": "auto",
                "local_files_only": True,
                "trust_remote_code": True,
                "torch_dtype": "bfloat16",
            },
            "decoding": {
                "strategy": "generate",
                "generation_kwargs": {"max_new_tokens": 64},
            },
            "audio": {
                "maximum_seconds": 40,
                "long_audio_strategy": "sequential_chunks",
                "chunk_seconds": 40,
                "overlap_seconds": 0,
            },
            "hardware": {
                "memory_strategy": "cpu_disk_offload",
                "minimum_gpu_memory_gib": 12,
                "gpu_max_memory_gib": 13,
                "cpu_max_memory_gib": 12,
                "output_mode": "text_only",
            },
        }
    )

    assert profile.loader.trust_remote_code is True
    assert profile.hardware is not None
    assert profile.hardware.memory_strategy == "cpu_disk_offload"
    assert profile.to_dict()["hardware"]["gpu_max_memory_gib"] == 13.0


def test_trusted_code_remains_rejected_outside_phi4() -> None:
    profile = copy.deepcopy(BASE_PROFILE)
    profile["loader"]["trust_remote_code"] = True
    with pytest.raises(ProfileError, match="only be enabled"):
        parse_profile(profile)


def test_qwen_profile_requires_large_gpu_and_text_only() -> None:
    profile = copy.deepcopy(MULTIMODAL_PROFILE)
    profile.update(
        {
            "id": "qwen-test",
            "adapter": "qwen_omni_audio",
            "artifact": "qwen-model",
            "hardware": {
                "memory_strategy": "large_gpu_only",
                "minimum_gpu_memory_gib": 40,
                "output_mode": "text_only",
            },
        }
    )
    parsed = parse_profile(profile)
    assert parsed.hardware is not None
    assert parsed.hardware.minimum_gpu_memory_gib == 40.0

    profile["hardware"]["minimum_gpu_memory_gib"] = 16
    with pytest.raises(ProfileError, match="at least 38 GiB"):
        parse_profile(profile)
