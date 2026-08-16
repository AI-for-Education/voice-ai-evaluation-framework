"""Strict, portable YAML model profiles for offline inference."""

from __future__ import annotations

import os
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in incomplete environments
    yaml = None


_TOP_LEVEL_KEYS = {
    "id",
    "framework",
    "adapter",
    "artifact",
    "language",
    "task",
    "output_units",
    "output_notation",
    "output_inventory",
    "prompt",
    "audio",
    "hardware",
    "loader",
    "decoding",
    "parameter_evidence",
}
_LOADER_KEYS = {
    "processor_mode",
    "local_files_only",
    "trust_remote_code",
    "torch_dtype",
    "attention_implementation",
}
_DECODING_KEYS = {
    "strategy",
    "generation_kwargs",
    "ctc_lm_kwargs",
    "transducer_search_kwargs",
    "hallucination_guard",
    "long_form",
}
_HALLUCINATION_GUARD_KEYS = {
    "max_words_per_second",
    "repeated_phrase_min_words",
    "repeated_phrase_max_words",
    "repeated_phrase_repetitions",
}
_LONG_FORM_KEYS = {
    "strategy",
    "threshold_seconds",
}
_AUDIO_KEYS = {
    "maximum_seconds",
    "long_audio_strategy",
    "chunk_seconds",
    "overlap_seconds",
}
_HARDWARE_KEYS = {
    "memory_strategy",
    "minimum_gpu_memory_gib",
    "gpu_max_memory_gib",
    "cpu_max_memory_gib",
    "output_mode",
}
_GENERATION_KEYS = {
    "do_sample",
    "early_stopping",
    "length_penalty",
    "max_new_tokens",
    "num_beams",
    "return_timestamps",
    "suppress_tokens",
}
_CTC_LM_KEYS = {
    "beam_width",
    "alpha",
    "beta",
    "unk_score_offset",
    "lm_score_boundary",
    "n_best",
    "decoder_workers",
}
_TRANSDUCER_SEARCH_KEYS = {"max_active_paths"}
_PARAMETER_EVIDENCE_KEYS = {
    "applies_to",
    "level",
    "source",
    "rationale",
}
_PARAMETER_EVIDENCE_LEVELS = {
    "exact",
    "supported",
    "project",
    "unvalidated",
}
_ADAPTERS = {
    "multimodal": {"gemma4_audio", "phi4_audio", "qwen_omni_audio"},
    "nemo": {"nemo"},
    "onnxruntime": {"android_ctc", "ctc"},
    "sherpa_onnx": {"online_transducer"},
    "transformers": {"ctc", "speech_seq2seq"},
}


class ProfileError(ValueError):
    """Raised when a model profile is invalid."""


@dataclass(frozen=True)
class LoaderConfig:
    processor_mode: str = "auto"
    local_files_only: bool = True
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    attention_implementation: str | None = None


@dataclass(frozen=True)
class CTCLMConfig:
    beam_width: int
    alpha: float
    beta: float
    unk_score_offset: float
    lm_score_boundary: bool
    n_best: int
    decoder_workers: int = 1


@dataclass(frozen=True)
class TransducerSearchConfig:
    max_active_paths: int


@dataclass(frozen=True)
class HallucinationGuardConfig:
    max_words_per_second: float
    repeated_phrase_min_words: int
    repeated_phrase_max_words: int
    repeated_phrase_repetitions: int


@dataclass(frozen=True)
class WhisperLongFormConfig:
    strategy: str
    threshold_seconds: float


@dataclass(frozen=True)
class DecodingConfig:
    strategy: str
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    ctc_lm_kwargs: CTCLMConfig | None = None
    transducer_search_kwargs: TransducerSearchConfig | None = None
    hallucination_guard: HallucinationGuardConfig | None = None
    long_form: WhisperLongFormConfig | None = None



@dataclass(frozen=True)
class AudioConfig:
    maximum_seconds: float
    long_audio_strategy: str
    chunk_seconds: float
    overlap_seconds: float = 0.0

@dataclass(frozen=True)
class HardwareConfig:
    memory_strategy: str
    minimum_gpu_memory_gib: float
    gpu_max_memory_gib: float | None
    cpu_max_memory_gib: float | None
    output_mode: str

@dataclass(frozen=True)
class ParameterEvidence:
    applies_to: tuple[str, ...]
    level: str
    source: str
    rationale: str




@dataclass(frozen=True)
class ModelProfile:
    id: str
    framework: str
    adapter: str
    artifact: str
    language: str | None
    task: str
    output_units: str
    output_notation: str | None
    output_inventory: str | None
    prompt: str | None
    audio: AudioConfig | None
    hardware: HardwareConfig | None
    loader: LoaderConfig
    decoding: DecodingConfig

    parameter_evidence: tuple[ParameterEvidence, ...] = ()
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["loader"]["attention_implementation"] is None:
            # Profiles created before parameter provenance was introduced keep
            # their existing serialized shape and effective adapter defaults.
            del payload["loader"]["attention_implementation"]
        if not payload["parameter_evidence"]:
            del payload["parameter_evidence"]
        else:
            payload["parameter_evidence"] = [
                {
                    **item,
                    "applies_to": list(item["applies_to"]),
                }
                for item in payload["parameter_evidence"]
            ]
        if payload["decoding"]["ctc_lm_kwargs"] is None:
            # Keep existing greedy/seq2seq metadata stable; the dedicated LM
            # block appears only for the new beam-search profile instance.
            del payload["decoding"]["ctc_lm_kwargs"]
        if payload["decoding"]["transducer_search_kwargs"] is None:
            del payload["decoding"]["transducer_search_kwargs"]
        if payload["output_notation"] is None:
            del payload["output_notation"]
        if payload["output_inventory"] is None:
            del payload["output_inventory"]
        if payload["prompt"] is None:
            del payload["prompt"]
        if payload["audio"] is None:
            del payload["audio"]
        if payload["hardware"] is None:
            del payload["hardware"]

        if payload["decoding"]["hallucination_guard"] is None:
            del payload["decoding"]["hallucination_guard"]
        if payload["decoding"]["long_form"] is None:
            del payload["decoding"]["long_form"]
        return payload


def _unknown_keys(data: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ProfileError(f"Unknown {section} key(s): {', '.join(unknown)}")


def _require_mapping(value: Any, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProfileError(f"{section} must be a mapping")
    return value


def _validate_artifact(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError("artifact is required")
    artifact = value.strip()
    posix_path = PurePosixPath(artifact)
    windows_path = PureWindowsPath(artifact)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part == ".." for part in posix_path.parts)
        or any(part == ".." for part in windows_path.parts)
    ):
        raise ProfileError("artifact must be a relative path without '..'")
    return artifact


def _build_loader(
    data: dict[str, Any],
    *,
    allow_trusted_local_code: bool = False,
) -> LoaderConfig:
    _unknown_keys(data, _LOADER_KEYS, "loader")
    processor_mode = data.get("processor_mode", "auto")
    local_files_only = data.get("local_files_only", True)
    trust_remote_code = data.get("trust_remote_code", False)
    torch_dtype = data.get("torch_dtype", "auto")
    attention_implementation = data.get("attention_implementation")
    if not isinstance(processor_mode, str):
        raise ProfileError("loader.processor_mode must be a string")
    if type(local_files_only) is not bool:
        raise ProfileError("loader.local_files_only must be a boolean")
    if type(trust_remote_code) is not bool:
        raise ProfileError("loader.trust_remote_code must be a boolean")
    if not isinstance(torch_dtype, str):
        raise ProfileError("loader.torch_dtype must be a string")
    if attention_implementation is not None and not isinstance(
        attention_implementation, str
    ):
        raise ProfileError("loader.attention_implementation must be a string or null")
    config = LoaderConfig(
        processor_mode=processor_mode,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        attention_implementation=attention_implementation,
    )
    if config.processor_mode not in {
        "auto",
        "wav2vec2_plain",
        "wav2vec2_with_lm",
    }:
        raise ProfileError(
            "loader.processor_mode must be 'auto', 'wav2vec2_plain', or "
            "'wav2vec2_with_lm'"
        )
    if config.local_files_only is not True:
        raise ProfileError("loader.local_files_only must remain true for offline inference")
    if config.trust_remote_code and not allow_trusted_local_code:
        raise ProfileError(
            "loader.trust_remote_code may only be enabled for the bundled "
            "multimodal/phi4_audio implementation"
        )
    if config.torch_dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ProfileError(
            "loader.torch_dtype must be auto, float32, float16, or bfloat16"
        )
    if config.attention_implementation not in {
        None,
        "eager",
        "sdpa",
        "flash_attention_2",
    }:
        raise ProfileError(
            "loader.attention_implementation must be eager, sdpa, "
            "flash_attention_2, or null"
        )
    return config




def _build_parameter_evidence(
    value: Any,
    *,
    profile_data: dict[str, Any],
) -> tuple[ParameterEvidence, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise ProfileError("parameter_evidence must be a non-empty list")

    evidence: list[ParameterEvidence] = []
    for index, raw_item in enumerate(value):
        section = f"parameter_evidence[{index}]"
        item = _require_mapping(raw_item, section)
        _unknown_keys(item, _PARAMETER_EVIDENCE_KEYS, section)
        missing = sorted(_PARAMETER_EVIDENCE_KEYS - set(item))
        if missing:
            raise ProfileError(
                f"{section} is missing required key(s): " + ", ".join(missing)
            )

        applies_to = item["applies_to"]
        if not isinstance(applies_to, list) or not applies_to:
            raise ProfileError(f"{section}.applies_to must be a non-empty list")
        if any(not isinstance(path, str) or not path.strip() for path in applies_to):
            raise ProfileError(
                f"{section}.applies_to must contain non-empty dotted paths"
            )
        normalized_paths = tuple(path.strip() for path in applies_to)
        if len(set(normalized_paths)) != len(normalized_paths):
            raise ProfileError(f"{section}.applies_to cannot contain duplicates")

        for path in normalized_paths:
            current: Any = profile_data
            for component in path.split("."):
                if not isinstance(current, dict) or component not in current:
                    raise ProfileError(
                        f"{section}.applies_to references missing profile field: {path}"
                    )
                current = current[component]

        level = item["level"]
        if level not in _PARAMETER_EVIDENCE_LEVELS:
            raise ProfileError(
                f"{section}.level must be one of: "
                + ", ".join(sorted(_PARAMETER_EVIDENCE_LEVELS))
            )
        source = item["source"]
        if not isinstance(source, str) or not source.strip():
            raise ProfileError(f"{section}.source must be a non-empty string")
        source = source.strip()
        if not source.startswith(("https://", "artifact:", "repo:")):
            raise ProfileError(
                f"{section}.source must use https://, artifact:, or repo:"
            )
        rationale = item["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ProfileError(f"{section}.rationale must be a non-empty string")

        evidence.append(
            ParameterEvidence(
                applies_to=normalized_paths,
                level=level,
                source=source,
                rationale=rationale.strip(),
            )
        )
    return tuple(evidence)
def _validate_generation_kwargs(generation_kwargs: dict[str, Any]) -> None:
    for key in ("max_new_tokens", "num_beams"):
        if key in generation_kwargs and (
            type(generation_kwargs[key]) is not int or generation_kwargs[key] < 1
        ):
            raise ProfileError(f"decoding.generation_kwargs.{key} must be a positive integer")

    for key in ("do_sample", "early_stopping"):
        if key in generation_kwargs and type(generation_kwargs[key]) is not bool:
            raise ProfileError(
                f"decoding.generation_kwargs.{key} must be a boolean"
            )

    if "length_penalty" in generation_kwargs:
        value = generation_kwargs["length_penalty"]
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise ProfileError(
                "decoding.generation_kwargs.length_penalty must be a finite number"
            )

    if "suppress_tokens" in generation_kwargs:
        value = generation_kwargs["suppress_tokens"]
        if not isinstance(value, list) or any(
            type(token) is not int or token < 0 for token in value
        ):
            raise ProfileError(
                "decoding.generation_kwargs.suppress_tokens must be a list of "
                "non-negative integers"
            )

    if (
        "return_timestamps" in generation_kwargs
        and generation_kwargs["return_timestamps"] is not False
    ):
        raise ProfileError(
            "decoding.generation_kwargs.return_timestamps must be false in this delivery"
        )



def _build_hallucination_guard(
    data: dict[str, Any],
) -> HallucinationGuardConfig:
    _unknown_keys(data, _HALLUCINATION_GUARD_KEYS, "hallucination_guard")
    required = set(_HALLUCINATION_GUARD_KEYS)
    missing = sorted(required - set(data))
    if missing:
        raise ProfileError(
            "decoding.hallucination_guard is missing required key(s): "
            + ", ".join(missing)
        )

    max_wps = data["max_words_per_second"]
    if type(max_wps) not in {int, float} or not math.isfinite(float(max_wps)):
        raise ProfileError(
            "decoding.hallucination_guard.max_words_per_second must be a finite number"
        )
    if float(max_wps) <= 0:
        raise ProfileError(
            "decoding.hallucination_guard.max_words_per_second must be greater than zero"
        )

    integer_keys = (
        "repeated_phrase_min_words",
        "repeated_phrase_max_words",
        "repeated_phrase_repetitions",
    )
    for key in integer_keys:
        if type(data[key]) is not int or data[key] < 1:
            raise ProfileError(
                f"decoding.hallucination_guard.{key} must be a positive integer"
            )
    if data["repeated_phrase_max_words"] < data["repeated_phrase_min_words"]:
        raise ProfileError(
            "decoding.hallucination_guard.repeated_phrase_max_words cannot be "
            "less than repeated_phrase_min_words"
        )
    if data["repeated_phrase_repetitions"] < 2:
        raise ProfileError(
            "decoding.hallucination_guard.repeated_phrase_repetitions must be at least 2"
        )

    return HallucinationGuardConfig(
        max_words_per_second=float(max_wps),
        repeated_phrase_min_words=data["repeated_phrase_min_words"],
        repeated_phrase_max_words=data["repeated_phrase_max_words"],
        repeated_phrase_repetitions=data["repeated_phrase_repetitions"],
    )


def _build_whisper_long_form(data: dict[str, Any]) -> WhisperLongFormConfig:
    _unknown_keys(data, _LONG_FORM_KEYS, "long_form")
    required = set(_LONG_FORM_KEYS)
    missing = sorted(required - set(data))
    if missing:
        raise ProfileError(
            "decoding.long_form is missing required key(s): " + ", ".join(missing)
        )

    strategy = data["strategy"]
    if strategy != "timestamp":
        raise ProfileError("decoding.long_form.strategy must be 'timestamp'")

    threshold = data["threshold_seconds"]
    if type(threshold) not in {int, float} or not math.isfinite(float(threshold)):
        raise ProfileError(
            "decoding.long_form.threshold_seconds must be a finite number"
        )
    if float(threshold) <= 0:
        raise ProfileError(
            "decoding.long_form.threshold_seconds must be greater than zero"
        )

    return WhisperLongFormConfig(
        strategy=strategy,
        threshold_seconds=float(threshold),
    )


def _build_ctc_lm_kwargs(data: dict[str, Any]) -> CTCLMConfig:
    _unknown_keys(data, _CTC_LM_KEYS, "ctc_lm_kwargs")
    required = {
        "beam_width",
        "alpha",
        "beta",
        "unk_score_offset",
        "lm_score_boundary",
        "n_best",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ProfileError(
            "decoding.ctc_lm_kwargs is missing required key(s): "
            + ", ".join(missing)
        )

    for key in ("beam_width", "n_best"):
        value = data[key]
        if type(value) is not int or value < 1:
            raise ProfileError(
                f"decoding.ctc_lm_kwargs.{key} must be a positive integer"
            )

    decoder_workers = data.get("decoder_workers", 1)
    if type(decoder_workers) is not int or decoder_workers < 0:
        raise ProfileError(
            "decoding.ctc_lm_kwargs.decoder_workers must be a non-negative integer"
        )

    for key in ("alpha", "beta", "unk_score_offset"):
        value = data[key]
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise ProfileError(
                f"decoding.ctc_lm_kwargs.{key} must be a finite number"
            )

    if type(data["lm_score_boundary"]) is not bool:
        raise ProfileError(
            "decoding.ctc_lm_kwargs.lm_score_boundary must be a boolean"
        )

    return CTCLMConfig(
        beam_width=data["beam_width"],
        alpha=float(data["alpha"]),
        beta=float(data["beta"]),
        unk_score_offset=float(data["unk_score_offset"]),
        lm_score_boundary=data["lm_score_boundary"],
        n_best=data["n_best"],
        decoder_workers=decoder_workers,
    )


def _build_transducer_search_kwargs(
    data: dict[str, Any],
) -> TransducerSearchConfig:
    _unknown_keys(
        data,
        _TRANSDUCER_SEARCH_KEYS,
        "transducer_search_kwargs",
    )
    if "max_active_paths" not in data:
        raise ProfileError(
            "decoding.transducer_search_kwargs is missing required key: "
            "max_active_paths"
        )
    max_active_paths = data["max_active_paths"]
    if type(max_active_paths) is not int or max_active_paths < 1:
        raise ProfileError(
            "decoding.transducer_search_kwargs.max_active_paths must be a "
            "positive integer"
        )
    return TransducerSearchConfig(max_active_paths=max_active_paths)


def _build_audio(data: dict[str, Any]) -> AudioConfig:
    _unknown_keys(data, _AUDIO_KEYS, "audio")
    required = {"maximum_seconds", "long_audio_strategy", "chunk_seconds"}
    missing = sorted(required - set(data))
    if missing:
        raise ProfileError("audio is missing required key(s): " + ", ".join(missing))

    numeric: dict[str, float] = {}
    for key in ("maximum_seconds", "chunk_seconds", "overlap_seconds"):
        value = data.get(key, 0.0)
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise ProfileError(f"audio.{key} must be a finite number")
        numeric[key] = float(value)

    if numeric["maximum_seconds"] <= 0:
        raise ProfileError("audio.maximum_seconds must be greater than zero")
    if numeric["chunk_seconds"] <= 0:
        raise ProfileError("audio.chunk_seconds must be greater than zero")
    if numeric["chunk_seconds"] > numeric["maximum_seconds"]:
        raise ProfileError("audio.chunk_seconds cannot exceed audio.maximum_seconds")
    if numeric["overlap_seconds"] != 0:
        raise ProfileError("audio.overlap_seconds must be 0 in this delivery")

    strategy = data["long_audio_strategy"]
    if strategy != "sequential_chunks":
        raise ProfileError(
            "audio.long_audio_strategy must be 'sequential_chunks' in this delivery"
        )
    return AudioConfig(
        maximum_seconds=numeric["maximum_seconds"],
        long_audio_strategy=strategy,
        chunk_seconds=numeric["chunk_seconds"],
        overlap_seconds=numeric["overlap_seconds"],
    )


def _build_hardware(data: dict[str, Any]) -> HardwareConfig:
    _unknown_keys(data, _HARDWARE_KEYS, "hardware")
    required = {"memory_strategy", "minimum_gpu_memory_gib", "output_mode"}
    missing = sorted(required - set(data))
    if missing:
        raise ProfileError("hardware is missing required key(s): " + ", ".join(missing))

    memory_strategy = data["memory_strategy"]
    if memory_strategy not in {"cpu_disk_offload", "large_gpu_only"}:
        raise ProfileError(
            "hardware.memory_strategy must be cpu_disk_offload or large_gpu_only"
        )
    if data["output_mode"] != "text_only":
        raise ProfileError("hardware.output_mode must be text_only")

    numeric: dict[str, float | None] = {}
    for key in (
        "minimum_gpu_memory_gib",
        "gpu_max_memory_gib",
        "cpu_max_memory_gib",
    ):
        value = data.get(key)
        if value is None:
            numeric[key] = None
            continue
        if type(value) not in {int, float} or not math.isfinite(float(value)):
            raise ProfileError(f"hardware.{key} must be a finite number")
        if float(value) <= 0:
            raise ProfileError(f"hardware.{key} must be greater than zero")
        numeric[key] = float(value)

    if numeric["minimum_gpu_memory_gib"] is None:
        raise ProfileError("hardware.minimum_gpu_memory_gib is required")
    if memory_strategy == "cpu_disk_offload" and (
        numeric["gpu_max_memory_gib"] is None
        or numeric["cpu_max_memory_gib"] is None
    ):
        raise ProfileError(
            "cpu_disk_offload requires hardware.gpu_max_memory_gib and "
            "hardware.cpu_max_memory_gib"
        )
    if memory_strategy == "large_gpu_only" and (
        numeric["gpu_max_memory_gib"] is not None
        or numeric["cpu_max_memory_gib"] is not None
    ):
        raise ProfileError(
            "large_gpu_only cannot set gpu_max_memory_gib or cpu_max_memory_gib"
        )
    return HardwareConfig(
        memory_strategy=memory_strategy,
        minimum_gpu_memory_gib=float(numeric["minimum_gpu_memory_gib"]),
        gpu_max_memory_gib=numeric["gpu_max_memory_gib"],
        cpu_max_memory_gib=numeric["cpu_max_memory_gib"],
        output_mode="text_only",
    )


def _build_decoding(data: dict[str, Any], framework: str, adapter: str) -> DecodingConfig:
    _unknown_keys(data, _DECODING_KEYS, "decoding")
    if "strategy" not in data:
        raise ProfileError("decoding.strategy is required")
    generation_kwargs = _require_mapping(data.get("generation_kwargs"), "generation_kwargs")
    _unknown_keys(generation_kwargs, _GENERATION_KEYS, "generation_kwargs")
    if not isinstance(data["strategy"], str) or not data["strategy"].strip():
        raise ProfileError("decoding.strategy must be a non-empty string")
    _validate_generation_kwargs(generation_kwargs)
    strategy = data["strategy"].strip()
    ctc_lm_kwargs = None
    if "ctc_lm_kwargs" in data:
        ctc_lm_kwargs = _build_ctc_lm_kwargs(
            _require_mapping(data["ctc_lm_kwargs"], "ctc_lm_kwargs")
        )
    transducer_search_kwargs = None
    if "transducer_search_kwargs" in data:
        transducer_search_kwargs = _build_transducer_search_kwargs(
            _require_mapping(
                data["transducer_search_kwargs"],
                "transducer_search_kwargs",
            )
        )
    hallucination_guard = None
    if "hallucination_guard" in data:
        hallucination_guard = _build_hallucination_guard(
            _require_mapping(data["hallucination_guard"], "hallucination_guard")
        )
    long_form = None
    if "long_form" in data:
        long_form = _build_whisper_long_form(
            _require_mapping(data["long_form"], "long_form")
        )

    expected = {
        ("multimodal", "gemma4_audio"): {"generate"},
        ("multimodal", "phi4_audio"): {"generate"},
        ("multimodal", "qwen_omni_audio"): {"generate"},
        ("transformers", "ctc"): {"beam_search", "greedy"},
        ("transformers", "speech_seq2seq"): {"generate"},
        ("onnxruntime", "ctc"): {"greedy"},
        ("onnxruntime", "android_ctc"): {"greedy"},
        ("sherpa_onnx", "online_transducer"): {
            "greedy_search",
            "modified_beam_search",
        },
        ("nemo", "nemo"): {"ctc", "rnnt"},
    }[(framework, adapter)]
    if strategy not in expected:
        raise ProfileError(
            f"decoding.strategy '{strategy}' is invalid for {framework}/{adapter}; "
            f"expected one of: {', '.join(sorted(expected))}"
        )
    if adapter not in {
        "speech_seq2seq",
        "gemma4_audio",
        "phi4_audio",
        "qwen_omni_audio",
    } and generation_kwargs:
        raise ProfileError(
            "decoding.generation_kwargs is only valid for generative adapters"
        )
    if strategy == "beam_search" and ctc_lm_kwargs is None:
        raise ProfileError(
            "decoding.ctc_lm_kwargs is required for CTC beam_search decoding"
        )
    if strategy != "beam_search" and ctc_lm_kwargs is not None:
        raise ProfileError(
            "decoding.ctc_lm_kwargs is only valid for CTC beam_search decoding"
        )
    if strategy == "modified_beam_search" and transducer_search_kwargs is None:
        raise ProfileError(
            "decoding.transducer_search_kwargs is required for "
            "modified_beam_search decoding"
        )
    if strategy != "modified_beam_search" and transducer_search_kwargs is not None:
        raise ProfileError(
            "decoding.transducer_search_kwargs is only valid for "
            "modified_beam_search decoding"
        )
    if hallucination_guard is not None and (
        framework,
        adapter,
    ) != ("transformers", "speech_seq2seq"):
        raise ProfileError(
            "decoding.hallucination_guard is only valid for transformers/speech_seq2seq"
        )
    if long_form is not None and (
        framework,
        adapter,
    ) != ("transformers", "speech_seq2seq"):
        raise ProfileError(
            "decoding.long_form is only valid for transformers/speech_seq2seq"
        )
    if long_form is not None and generation_kwargs.get("do_sample", False) is not False:
        raise ProfileError(
            "decoding.long_form requires generation_kwargs.do_sample to be false"
        )
    return DecodingConfig(
        strategy=strategy,
        generation_kwargs=dict(generation_kwargs),
        ctc_lm_kwargs=ctc_lm_kwargs,
        transducer_search_kwargs=transducer_search_kwargs,
        hallucination_guard=hallucination_guard,
        long_form=long_form,
    )


def parse_profile(data: Any) -> ModelProfile:
    """Validate parsed YAML and return an immutable model profile."""
    if not isinstance(data, dict):
        raise ProfileError("Model profile must contain a YAML mapping")
    _unknown_keys(data, _TOP_LEVEL_KEYS, "profile")

    for key in ("id", "framework", "adapter", "artifact"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ProfileError(f"{key} is required")

    framework = data["framework"].strip()
    adapter = data["adapter"].strip()
    if framework not in _ADAPTERS:
        raise ProfileError(f"Unsupported framework: {framework}")
    if adapter not in _ADAPTERS[framework]:
        raise ProfileError(f"Unsupported adapter '{adapter}' for framework '{framework}'")

    task_value = data.get("task", "transcribe")
    if not isinstance(task_value, str):
        raise ProfileError("task must be a string")
    task = task_value.strip()
    if task != "transcribe":
        raise ProfileError("Only task: transcribe is supported in this delivery")
    output_units_value = data.get("output_units", "orthographic")
    if not isinstance(output_units_value, str):
        raise ProfileError("output_units must be a string")
    output_units = output_units_value.strip()
    if output_units not in {"orthographic", "phoneme"}:
        raise ProfileError("output_units must be orthographic or phoneme")

    output_notation = data.get("output_notation")
    output_inventory = data.get("output_inventory")
    if output_units == "phoneme":
        if output_notation != "ipa":
            raise ProfileError("phoneme output requires output_notation: ipa")
        if not isinstance(output_inventory, str) or not output_inventory.strip():
            raise ProfileError("phoneme output requires output_inventory")
        output_inventory = output_inventory.strip()
    elif output_notation is not None or output_inventory is not None:
        raise ProfileError(
            "output_notation and output_inventory are only valid for phoneme output"
        )


    loader = _build_loader(
        _require_mapping(data.get("loader"), "loader"),
        allow_trusted_local_code=(framework, adapter) == ("multimodal", "phi4_audio"),
    )
    if loader.processor_mode in {"wav2vec2_plain", "wav2vec2_with_lm"} and (
        framework,
        adapter,
    ) != ("transformers", "ctc"):
        raise ProfileError(
            f"loader.processor_mode '{loader.processor_mode}' is only valid for "
            "transformers/ctc"
        )
    if framework == "nemo" and loader.torch_dtype != "auto":
        raise ProfileError("loader.torch_dtype must be auto for the NeMo backend")

    prompt_value = data.get("prompt")
    audio_value = data.get("audio")
    if framework == "multimodal":
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            raise ProfileError("multimodal profiles require a non-empty prompt")
        prompt = prompt_value.strip()
        if audio_value is None:
            raise ProfileError("multimodal profiles require an audio section")
        audio = _build_audio(_require_mapping(audio_value, "audio"))
        if loader.processor_mode != "auto":
            raise ProfileError(f"{adapter} requires loader.processor_mode: auto")
        if loader.torch_dtype != "bfloat16":
            raise ProfileError(f"{adapter} requires loader.torch_dtype: bfloat16")
        if adapter == "phi4_audio" and loader.trust_remote_code is not True:
            raise ProfileError(
                "phi4_audio requires loader.trust_remote_code: true for its bundled code"
            )
    else:
        if prompt_value is not None or audio_value is not None:
            raise ProfileError("prompt and audio are only valid for multimodal profiles")
        prompt = None
        audio = None
    hardware_value = data.get("hardware")
    hardware = (
        _build_hardware(_require_mapping(hardware_value, "hardware"))
        if hardware_value is not None
        else None
    )
    if hardware is not None and framework != "multimodal":
        raise ProfileError("hardware is only valid for multimodal profiles")
    if adapter == "phi4_audio":
        if hardware is None or hardware.memory_strategy != "cpu_disk_offload":
            raise ProfileError(
                "phi4_audio requires hardware.memory_strategy: cpu_disk_offload"
            )
    elif adapter == "qwen_omni_audio":
        if hardware is None or hardware.memory_strategy != "large_gpu_only":
            raise ProfileError(
                "qwen_omni_audio requires hardware.memory_strategy: large_gpu_only"
            )
        if hardware.minimum_gpu_memory_gib < 38:
            raise ProfileError(
                "qwen_omni_audio requires at least 38 GiB in "
                "hardware.minimum_gpu_memory_gib"
            )
    elif hardware is not None:
        raise ProfileError(
            "hardware is only used by phi4_audio and qwen_omni_audio profiles"
        )
    decoding = _build_decoding(
        _require_mapping(data.get("decoding"), "decoding"), framework, adapter
    )
    if decoding.strategy == "beam_search" and loader.processor_mode != "wav2vec2_with_lm":
        raise ProfileError(
            "CTC beam_search decoding requires loader.processor_mode "
            "'wav2vec2_with_lm'"
        )
    if (
        loader.processor_mode == "wav2vec2_with_lm"
        and decoding.strategy != "beam_search"
    ):
        raise ProfileError(
            "loader.processor_mode 'wav2vec2_with_lm' requires CTC beam_search decoding"
        )

    language_value = data.get("language")
    if language_value is not None and not isinstance(language_value, str):
        raise ProfileError("language must be a string or null")
    parameter_evidence = _build_parameter_evidence(
        data.get("parameter_evidence"),
        profile_data=data,
    )

    return ModelProfile(
        id=data["id"].strip(),
        framework=framework,
        adapter=adapter,
        artifact=_validate_artifact(data["artifact"]),
        language=(language_value.strip() if language_value else None),
        task=task,
        output_units=output_units,
        output_notation=output_notation,
        output_inventory=output_inventory,
        prompt=prompt,
        audio=audio,
        hardware=hardware,
        loader=loader,
        decoding=decoding,
        parameter_evidence=parameter_evidence,
    )


def load_profile(path: str | Path, *, expected_framework: str | None = None) -> ModelProfile:
    """Load and validate one tracked YAML model profile."""
    if yaml is None:
        raise ProfileError("PyYAML is required to load --model_config profiles")
    profile_path = Path(path)
    if not profile_path.is_file():
        raise ProfileError(f"Model profile not found: {profile_path}")
    try:
        data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileError(f"Invalid YAML in {profile_path}: {exc}") from exc
    profile = parse_profile(data)
    if expected_framework and profile.framework != expected_framework:
        raise ProfileError(
            f"Profile framework is '{profile.framework}', expected '{expected_framework}'"
        )
    return profile


def default_model_root(framework: str) -> Path:
    """Return a framework-local model root, overridable inside containers."""
    override = os.environ.get("ASR_MODEL_ROOT", "").strip()
    if override:
        return Path(override)
    if framework not in _ADAPTERS:
        raise ProfileError(f"Unsupported framework: {framework}")
    return Path(__file__).resolve().parent / framework / "models"


def resolve_model_path(
    profile: ModelProfile,
    *,
    model_root: str | Path | None = None,
    require_exists: bool = True,
) -> Path:
    """Resolve a profile artifact below its framework model root."""
    root = Path(model_root) if model_root is not None else default_model_root(profile.framework)
    root = root.resolve()
    candidate = (root / profile.artifact).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProfileError(f"Model artifact escaped its model root: {profile.artifact}") from exc
    if require_exists and not candidate.exists():
        raise ProfileError(
            f"Model artifact not found for profile '{profile.id}': {candidate}"
        )
    return candidate
