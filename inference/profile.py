"""Strict, portable YAML inference profiles for offline inference."""

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
    "profile_schema_version",
    "inference_setup_id",
    "inference_library",
    "adapter",
    "artifact",
    "model",
    "api",
    "request",
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
    "pipeline_contract",
    "parameter_evidence",
}
_OPENROUTER_MODEL_KEYS = {"slug"}
_OPENROUTER_API_KEYS = {
    "base_url",
    "routing_region",
    "endpoint_kind",
    "api_key_environment_variable",
}
_OPENROUTER_REQUEST_KEYS = {
    "stream",
    "provider",
    "temperature",
    "max_tokens",
    "seed",
    "reasoning",
}
_OPENROUTER_PROVIDER_KEYS = {"zdr", "data_collection"}
_OPENROUTER_REASONING_KEYS = {"effort", "exclude"}
_OPENROUTER_GLOBAL_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_EU_BASE_URL = "https://eu.openrouter.ai/api/v1"
_OPENROUTER_BASE_URLS = {
    "global": _OPENROUTER_GLOBAL_BASE_URL,
    "eu": _OPENROUTER_EU_BASE_URL,
}
_OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
_LOADER_KEYS = {
    "processor_mode",
    "local_files_only",
    "trust_remote_code",
    "torch_dtype",
    "attention_implementation",
    "artifact_format",
    "artifact_precision",
}
_DECODING_KEYS = {
    "strategy",
    "generation_kwargs",
    "ctc_lm_kwargs",
    "transducer_search_kwargs",
    # Post-decoding text guards are intentionally unsupported.
    "long_form",
}
# Model hypotheses are evaluated without text postprocessing.
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
_PIPELINE_CONTRACT_V2_KEYS = {
    "schema_version",
    "audio_preparation",
    "model_artifact",
    "input_processing",
    "execution_stack",
    "chunking",
    "evaluation",
    "observation_policy",
}
_ADAPTERS = {
    "multimodal": {"gemma4_audio", "phi4_audio", "qwen_omni_audio"},
    "nemo": {"nemo"},
    "omnilingual": {"ctc", "llm"},
    "onnxruntime": {"android_ctc", "ctc"},
    "openrouter": {"chat_audio", "stt"},
    "sherpa_onnx": {"online_transducer"},
    "torch": {"streaming_transducer"},
    "transformers": {"ctc", "speech_seq2seq"},
}


class ProfileError(ValueError):
    """Raised when an inference profile is invalid."""


@dataclass(frozen=True)
class LoaderConfig:
    processor_mode: str = "auto"
    local_files_only: bool = True
    trust_remote_code: bool = False
    torch_dtype: str = "auto"
    attention_implementation: str | None = None
    artifact_format: str = "auto"
    artifact_precision: str = "auto"


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


# No post-decoding mutation config is exposed by inference profiles.


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
    # pred_text is always the direct inference-adapter hypothesis.
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
class OpenRouterModelConfig:
    slug: str


@dataclass(frozen=True)
class OpenRouterAPIConfig:
    base_url: str
    routing_region: str
    endpoint_kind: str
    api_key_environment_variable: str


@dataclass(frozen=True)
class OpenRouterProviderConfig:
    zdr: bool
    data_collection: str


@dataclass(frozen=True)
class OpenRouterReasoningConfig:
    effort: str
    exclude: bool


@dataclass(frozen=True)
class OpenRouterRequestConfig:
    stream: bool
    provider: OpenRouterProviderConfig
    temperature: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    reasoning: OpenRouterReasoningConfig | None = None


@dataclass(frozen=True)
class ParameterEvidence:
    applies_to: tuple[str, ...]
    level: str
    source: str
    rationale: str


@dataclass(frozen=True)
class PipelineContract:
    schema_version: int
    audio_preparation: str
    model_artifact: str
    input_processing: str
    execution_stack: str
    chunking: str
    evaluation: str
    observation_policy: str


@dataclass(frozen=True)
class InferenceProfile:
    inference_setup_id: str
    inference_library: str
    adapter: str
    artifact: str | None
    model: OpenRouterModelConfig | None
    api: OpenRouterAPIConfig | None
    request: OpenRouterRequestConfig | None
    language: str | None
    task: str
    output_units: str
    output_notation: str | None
    output_inventory: str | None
    prompt: str | None
    audio: AudioConfig | None
    hardware: HardwareConfig | None
    loader: LoaderConfig | None
    decoding: DecodingConfig

    pipeline_contract: PipelineContract | None = None
    parameter_evidence: tuple[ParameterEvidence, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload = {"profile_schema_version": 2, **payload}
        if payload["artifact"] is None:
            del payload["artifact"]
        if payload["model"] is None:
            del payload["model"]
        if payload["api"] is None:
            del payload["api"]
        if payload["request"] is None:
            del payload["request"]
        else:
            for optional_request_key in (
                "temperature",
                "max_tokens",
                "seed",
                "reasoning",
            ):
                if payload["request"][optional_request_key] is None:
                    del payload["request"][optional_request_key]
        if payload["loader"] is None:
            del payload["loader"]
        elif payload["loader"]["attention_implementation"] is None:
            # Profiles created before parameter provenance was introduced keep
            # their existing serialized shape and effective adapter defaults.
            del payload["loader"]["attention_implementation"]
        if "loader" in payload:
            for defaulted_loader_key in ("artifact_format", "artifact_precision"):
                if payload["loader"][defaulted_loader_key] == "auto":
                    del payload["loader"][defaulted_loader_key]
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
        if payload["pipeline_contract"] is None:
            del payload["pipeline_contract"]
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

        # No postprocessing field is serialized.
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


def _build_openrouter_model(value: Any) -> OpenRouterModelConfig:
    data = _require_mapping(value, "model")
    _unknown_keys(data, _OPENROUTER_MODEL_KEYS, "model")
    slug = data.get("slug")
    if not isinstance(slug, str) or not slug.strip() or any(ch.isspace() for ch in slug):
        raise ProfileError("model.slug must be a non-empty OpenRouter model slug")
    return OpenRouterModelConfig(slug=slug.strip())


def _build_openrouter_api(
    value: Any, *, adapter: str
) -> OpenRouterAPIConfig:
    data = _require_mapping(value, "api")
    _unknown_keys(data, _OPENROUTER_API_KEYS, "api")
    missing = sorted(_OPENROUTER_API_KEYS - set(data))
    if missing:
        raise ProfileError("api is missing required key(s): " + ", ".join(missing))

    routing_region = data["routing_region"]
    if routing_region not in _OPENROUTER_BASE_URLS:
        raise ProfileError(
            "OpenRouter api.routing_region must be global or eu"
        )
    base_url = data["base_url"]
    expected_base_url = _OPENROUTER_BASE_URLS[routing_region]
    if base_url != expected_base_url:
        raise ProfileError(
            "OpenRouter api.base_url must exactly match api.routing_region: "
            f"{expected_base_url}"
        )
    endpoint_kind = data["endpoint_kind"]
    if endpoint_kind != adapter:
        raise ProfileError(
            "OpenRouter api.endpoint_kind must match the selected adapter"
        )
    key_environment = data["api_key_environment_variable"]
    if key_environment != _OPENROUTER_API_KEY_ENV:
        raise ProfileError(
            "OpenRouter api.api_key_environment_variable must be "
            f"{_OPENROUTER_API_KEY_ENV}"
        )
    return OpenRouterAPIConfig(
        base_url=base_url,
        routing_region=routing_region,
        endpoint_kind=endpoint_kind,
        api_key_environment_variable=key_environment,
    )


def _build_openrouter_request(
    value: Any,
    *,
    adapter: str,
) -> OpenRouterRequestConfig:
    data = _require_mapping(value, "request")
    _unknown_keys(data, _OPENROUTER_REQUEST_KEYS, "request")
    if data.get("stream") is not False:
        raise ProfileError("OpenRouter request.stream must be false")

    provider_data = _require_mapping(data.get("provider"), "request.provider")
    _unknown_keys(provider_data, _OPENROUTER_PROVIDER_KEYS, "request.provider")
    if provider_data.get("zdr") is not True:
        raise ProfileError("OpenRouter request.provider.zdr must be true")
    if provider_data.get("data_collection") != "deny":
        raise ProfileError(
            "OpenRouter request.provider.data_collection must be deny"
        )
    provider = OpenRouterProviderConfig(zdr=True, data_collection="deny")

    temperature = data.get("temperature")
    if temperature is not None:
        if type(temperature) not in {int, float} or not math.isfinite(
            float(temperature)
        ):
            raise ProfileError("request.temperature must be a finite number")
        if not 0 <= float(temperature) <= 1:
            raise ProfileError("request.temperature must be between 0 and 1")
        temperature = float(temperature)

    max_tokens = data.get("max_tokens")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise ProfileError("request.max_tokens must be a positive integer")
    seed = data.get("seed")
    if seed is not None and type(seed) is not int:
        raise ProfileError("request.seed must be an integer")

    reasoning = None
    if "reasoning" in data:
        reasoning_data = _require_mapping(data["reasoning"], "request.reasoning")
        _unknown_keys(
            reasoning_data, _OPENROUTER_REASONING_KEYS, "request.reasoning"
        )
        if reasoning_data.get("effort") not in {
            "minimal",
            "low",
            "medium",
            "high",
        }:
            raise ProfileError(
                "request.reasoning.effort must be minimal, low, medium, or high"
            )
        if reasoning_data.get("exclude") is not True:
            raise ProfileError("request.reasoning.exclude must be true")
        reasoning = OpenRouterReasoningConfig(
            effort=reasoning_data["effort"], exclude=True
        )

    if adapter == "chat_audio" and max_tokens is None:
        raise ProfileError("chat_audio requests require request.max_tokens")
    if adapter == "stt":
        if max_tokens is not None:
            raise ProfileError("stt requests cannot define request.max_tokens")
        if seed is not None:
            raise ProfileError("stt requests cannot define request.seed")
        if reasoning is not None:
            raise ProfileError("stt requests cannot define request.reasoning")

    return OpenRouterRequestConfig(
        stream=False,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        reasoning=reasoning,
    )


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
    artifact_format = data.get("artifact_format", "auto")
    artifact_precision = data.get("artifact_precision", "auto")
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
    if not isinstance(artifact_format, str):
        raise ProfileError("loader.artifact_format must be a string")
    if not isinstance(artifact_precision, str):
        raise ProfileError("loader.artifact_precision must be a string")
    config = LoaderConfig(
        processor_mode=processor_mode,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        attention_implementation=attention_implementation,
        artifact_format=artifact_format,
        artifact_precision=artifact_precision,
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
        raise ProfileError(
            "loader.local_files_only must remain true for offline inference"
        )
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
    if config.artifact_format not in {"auto", "onnx", "ort", "torchscript"}:
        raise ProfileError(
            "loader.artifact_format must be auto, onnx, ort, or torchscript"
        )
    if config.artifact_precision not in {"auto", "fp32", "int8"}:
        raise ProfileError(
            "loader.artifact_precision must be auto, fp32, or int8"
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


def _build_pipeline_contract(value: Any) -> PipelineContract | None:
    if value is None:
        return None
    data = _require_mapping(value, "pipeline_contract")
    if data.get("schema_version") != 2:
        raise ProfileError("pipeline_contract.schema_version must be 2")
    _unknown_keys(data, _PIPELINE_CONTRACT_V2_KEYS, "pipeline_contract")
    missing = sorted(_PIPELINE_CONTRACT_V2_KEYS - set(data))
    if missing:
        raise ProfileError(
            "pipeline_contract is missing required key(s): " + ", ".join(missing)
        )
    normalized = dict(data)

    for key in sorted(_PIPELINE_CONTRACT_V2_KEYS - {"schema_version"}):
        item = normalized[key]
        if not isinstance(item, str) or not item.strip():
            raise ProfileError(
                f"pipeline_contract.{key} must be a non-empty contract reference"
            )
        normalized[key] = item.strip()
    contract = PipelineContract(
        **normalized,
    )
    # Keep profile references fail-closed against the tracked registry while
    # leaving the registry definitions out of each compact model YAML.
    from inference.pipeline_provenance import resolve_contract_references

    resolved = resolve_contract_references(asdict(contract))

    resolved_sections = [
        resolved["audio_preparation"],
        resolved["model_artifact"],
        resolved["input_processing"],
        resolved["execution_stack"],
        resolved["chunking"],
        resolved["evaluation"],
        resolved["observation_policy"],
    ]
    if any(
        isinstance(section, dict) and section.get("status") == "not_available"
        for section in resolved_sections
    ):
        raise ProfileError(
            "pipeline_contract contains a reference missing from "
            "inference/pipeline_contracts.json"
        )
    return contract


def _validate_generation_kwargs(generation_kwargs: dict[str, Any]) -> None:
    for key in ("max_new_tokens", "num_beams"):
        if key in generation_kwargs and (
            type(generation_kwargs[key]) is not int or generation_kwargs[key] < 1
        ):
            raise ProfileError(
                f"decoding.generation_kwargs.{key} must be a positive integer"
            )

    for key in ("do_sample", "early_stopping"):
        if key in generation_kwargs and type(generation_kwargs[key]) is not bool:
            raise ProfileError(f"decoding.generation_kwargs.{key} must be a boolean")

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


# Profiles reject any post-decoding text mutation setting as an unknown key.


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
            "decoding.ctc_lm_kwargs is missing required key(s): " + ", ".join(missing)
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
            raise ProfileError(f"decoding.ctc_lm_kwargs.{key} must be a finite number")

    if type(data["lm_score_boundary"]) is not bool:
        raise ProfileError("decoding.ctc_lm_kwargs.lm_score_boundary must be a boolean")

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
        numeric["gpu_max_memory_gib"] is None or numeric["cpu_max_memory_gib"] is None
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


def _build_decoding(
    data: dict[str, Any], inference_library: str, adapter: str
) -> DecodingConfig:
    _unknown_keys(data, _DECODING_KEYS, "decoding")
    if "strategy" not in data:
        raise ProfileError("decoding.strategy is required")
    generation_kwargs = _require_mapping(
        data.get("generation_kwargs"), "generation_kwargs"
    )
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
    # No post-decoding mutation settings are accepted.
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
        ("openrouter", "chat_audio"): {"generate"},
        ("openrouter", "stt"): {"transcribe"},
        ("sherpa_onnx", "online_transducer"): {
            "greedy_search",
            "modified_beam_search",
        },
        ("torch", "streaming_transducer"): {
            "greedy_search",
            "modified_beam_search",
        },
        ("nemo", "nemo"): {"ctc", "rnnt"},
        ("omnilingual", "ctc"): {"ctc_greedy"},
        ("omnilingual", "llm"): {"llm_beam_search"},
    }[(inference_library, adapter)]
    if strategy not in expected:
        raise ProfileError(
            "decoding.strategy "
            f"'{strategy}' is invalid for {inference_library}/{adapter}; "
            f"expected one of: {', '.join(sorted(expected))}"
        )
    if (
        adapter
        not in {
            "speech_seq2seq",
            "gemma4_audio",
            "phi4_audio",
            "qwen_omni_audio",
        }
        and generation_kwargs
    ):
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
    # Decoder output is returned directly for every inference library.
    if long_form is not None and (
        inference_library,
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
        # No post-decoding mutation configuration.
        long_form=long_form,
    )


def parse_profile(data: Any) -> InferenceProfile:
    """Validate parsed YAML and return an immutable inference profile."""
    if not isinstance(data, dict):
        raise ProfileError("Inference profile must contain a YAML mapping")
    _unknown_keys(data, _TOP_LEVEL_KEYS, "profile")

    if data.get("profile_schema_version") != 2:
        raise ProfileError("profile_schema_version must be 2")
    inference_setup_id = data.get("inference_setup_id")
    inference_library = data.get("inference_library")
    required_values = {
        "inference_setup_id": inference_setup_id,
        "inference_library": inference_library,
        "adapter": data.get("adapter"),
    }
    for key, value in required_values.items():
        if not isinstance(value, str) or not value.strip():
            raise ProfileError(f"{key} is required")

    inference_library = inference_library.strip()
    adapter = data["adapter"].strip()
    if inference_library not in _ADAPTERS:
        raise ProfileError(f"Unsupported inference library: {inference_library}")
    if adapter not in _ADAPTERS[inference_library]:
        raise ProfileError(
            f"Unsupported adapter '{adapter}' for inference library "
            f"'{inference_library}'"
        )

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

    is_openrouter = inference_library == "openrouter"
    if is_openrouter:
        if "artifact" in data:
            raise ProfileError("OpenRouter profiles use model.slug, not artifact")
        if "loader" in data:
            raise ProfileError("OpenRouter profiles cannot define a local loader")
        artifact = None
        loader = None
    else:
        artifact = _validate_artifact(data.get("artifact"))
        if any(key in data for key in ("model", "api", "request")):
            raise ProfileError(
                "model, api, and request are only valid for OpenRouter profiles"
            )
        loader = _build_loader(
            _require_mapping(data.get("loader"), "loader"),
            allow_trusted_local_code=(inference_library, adapter)
            == ("multimodal", "phi4_audio"),
        )

    if loader is not None and loader.processor_mode in {
        "wav2vec2_plain",
        "wav2vec2_with_lm",
    } and (
        inference_library,
        adapter,
    ) != ("transformers", "ctc"):
        raise ProfileError(
            f"loader.processor_mode '{loader.processor_mode}' is only valid for "
            "transformers/ctc"
        )
    if inference_library == "nemo" and loader is not None and loader.torch_dtype != "auto":
        raise ProfileError("loader.torch_dtype must be auto for the NeMo backend")
    explicit_artifact_loader = (
        loader is not None
        and (loader.artifact_format != "auto" or loader.artifact_precision != "auto")
    )
    if explicit_artifact_loader and inference_library not in {"sherpa_onnx", "torch"}:
        raise ProfileError(
            "loader.artifact_format and loader.artifact_precision are only valid "
            "for sherpa_onnx and torch inference"
        )
    if inference_library == "sherpa_onnx":
        if loader.artifact_format == "torchscript":
            raise ProfileError(
                "sherpa_onnx profiles cannot load torchscript artifacts"
            )
    if inference_library == "torch":
        if loader.artifact_format != "torchscript":
            raise ProfileError(
                "torch/streaming_transducer requires "
                "loader.artifact_format: torchscript"
            )
        if loader.artifact_precision != "fp32":
            raise ProfileError(
                "torch/streaming_transducer requires "
                "loader.artifact_precision: fp32"
            )

    prompt_value = data.get("prompt")
    audio_value = data.get("audio")
    if inference_library == "multimodal":
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
    elif is_openrouter:
        if audio_value is not None:
            raise ProfileError(
                "OpenRouter sends each source WAV directly and cannot define audio chunking"
            )
        audio = None
        if adapter == "chat_audio":
            if not isinstance(prompt_value, str) or not prompt_value.strip():
                raise ProfileError("OpenRouter chat_audio profiles require a prompt")
            prompt = prompt_value.strip()
        else:
            if prompt_value is not None:
                raise ProfileError("OpenRouter stt profiles cannot define a prompt")
            prompt = None
    elif inference_library == "omnilingual":
        if prompt_value is not None:
            raise ProfileError("Omnilingual ASR profiles cannot define a prompt")
        prompt = None
        if audio_value is None:
            raise ProfileError("Omnilingual ASR profiles require an audio section")
        audio = _build_audio(_require_mapping(audio_value, "audio"))
        if loader.processor_mode != "auto":
            raise ProfileError("Omnilingual ASR requires loader.processor_mode: auto")
        if loader.local_files_only is not True:
            raise ProfileError("Omnilingual ASR inference must be local-files-only")
        if loader.trust_remote_code is not False:
            raise ProfileError("Omnilingual ASR cannot enable trust_remote_code")
        if loader.torch_dtype != "bfloat16":
            raise ProfileError("Omnilingual ASR requires loader.torch_dtype: bfloat16")
    else:
        if prompt_value is not None:
            raise ProfileError("prompt is only valid for multimodal profiles")
        prompt = None
        if audio_value is not None:
            if (inference_library, adapter) != (
                "transformers",
                "speech_seq2seq",
            ):
                raise ProfileError(
                    "audio is only valid for multimodal profiles or "
                    "transformers/speech_seq2seq"
                )
            audio = _build_audio(_require_mapping(audio_value, "audio"))
        else:
            audio = None
    hardware_value = data.get("hardware")
    hardware = (
        _build_hardware(_require_mapping(hardware_value, "hardware"))
        if hardware_value is not None
        else None
    )
    if hardware is not None and inference_library != "multimodal":
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
    elif adapter == "gemma4_audio":
        if hardware is not None and hardware.memory_strategy != "large_gpu_only":
            raise ProfileError(
                "gemma4_audio hardware.memory_strategy must be large_gpu_only"
            )
    elif hardware is not None:
        raise ProfileError(
            "hardware is only used by gemma4_audio, phi4_audio, and "
            "qwen_omni_audio profiles"
        )
    decoding = _build_decoding(
        _require_mapping(data.get("decoding"), "decoding"),
        inference_library,
        adapter,
    )
    if audio is not None and decoding.long_form is not None:
        raise ProfileError(
            "audio chunking and decoding.long_form cannot be enabled together"
        )
    if (
        loader is not None
        and decoding.strategy == "beam_search"
        and loader.processor_mode != "wav2vec2_with_lm"
    ):
        raise ProfileError(
            "CTC beam_search decoding requires loader.processor_mode 'wav2vec2_with_lm'"
        )
    if (
        loader is not None
        and loader.processor_mode == "wav2vec2_with_lm"
        and decoding.strategy != "beam_search"
    ):
        raise ProfileError(
            "loader.processor_mode 'wav2vec2_with_lm' requires CTC beam_search decoding"
        )

    language_value = data.get("language")
    if language_value is not None and not isinstance(language_value, str):
        raise ProfileError("language must be a string or null")
    language = language_value.strip() if language_value else None
    if inference_library == "omnilingual" and adapter == "llm" and language is None:
        raise ProfileError("Omnilingual LLM profiles require a language code")
    if is_openrouter:
        model = _build_openrouter_model(data.get("model"))
        api = _build_openrouter_api(data.get("api"), adapter=adapter)
        request = _build_openrouter_request(data.get("request"), adapter=adapter)
    else:
        model = None
        api = None
        request = None

    parameter_evidence = _build_parameter_evidence(
        data.get("parameter_evidence"),
        profile_data=data,
    )
    pipeline_contract = _build_pipeline_contract(data.get("pipeline_contract"))

    return InferenceProfile(
        inference_setup_id=inference_setup_id.strip(),
        inference_library=inference_library,
        adapter=adapter,
        artifact=artifact,
        model=model,
        api=api,
        request=request,
        language=language,
        task=task,
        output_units=output_units,
        output_notation=output_notation,
        output_inventory=output_inventory,
        prompt=prompt,
        audio=audio,
        hardware=hardware,
        loader=loader,
        decoding=decoding,
        pipeline_contract=pipeline_contract,
        parameter_evidence=parameter_evidence,
    )


def load_profile(
    path: str | Path,
    *,
    expected_inference_library: str | None = None,
) -> InferenceProfile:
    """Load and validate one tracked YAML inference profile."""
    if yaml is None:
        raise ProfileError("PyYAML is required to load inference profiles")
    profile_path = Path(path)
    if not profile_path.is_file():
        raise ProfileError(f"Inference profile not found: {profile_path}")
    try:
        data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileError(f"Invalid YAML in {profile_path}: {exc}") from exc
    profile = parse_profile(data)
    if (
        expected_inference_library
        and profile.inference_library != expected_inference_library
    ):
        raise ProfileError(
            "Profile inference library is "
            f"'{profile.inference_library}', expected '{expected_inference_library}'"
        )
    return profile


def default_model_root(inference_library: str) -> Path:
    """Return an inference-library model root, overridable in containers."""
    override = os.environ.get("ASR_MODEL_ROOT", "").strip()
    if override:
        return Path(override)
    if inference_library not in _ADAPTERS:
        raise ProfileError(f"Unsupported inference library: {inference_library}")
    return Path(__file__).resolve().parent / inference_library / "models"


def resolve_model_path(
    profile: InferenceProfile,
    *,
    model_root: str | Path | None = None,
    require_exists: bool = True,
) -> Path:
    """Resolve a model artifact below its inference-library model root."""
    if profile.artifact is None:
        raise ProfileError(
            f"Inference setup '{profile.inference_setup_id}' uses a remote model "
            "and has no local model artifact path"
        )
    root = (
        Path(model_root)
        if model_root is not None
        else default_model_root(profile.inference_library)
    )
    root = root.resolve()
    candidate = (root / profile.artifact).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProfileError(
            f"Model artifact escaped its model root: {profile.artifact}"
        ) from exc
    if require_exists and not candidate.exists():
        raise ProfileError(
            "Model artifact not found for inference setup "
            f"'{profile.inference_setup_id}': {candidate}"
        )
    return candidate
