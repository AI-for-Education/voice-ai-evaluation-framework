"""Normalized three-stage provenance for inference and evaluation artifacts."""

from __future__ import annotations

import json
import os
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from inference.provenance import canonical_json_sha256, file_identity


PIPELINE_PROVENANCE_SCHEMA_VERSION = 1
CONTRACT_REGISTRY_PATH = Path(__file__).with_name("pipeline_contracts.json")


def not_available(reason: str) -> dict[str, str]:
    """Return the one machine-readable representation of unavailable history."""
    return {"status": "not_available", "reason": reason}


def not_applicable(reason: str) -> dict[str, str]:
    """Represent a field that does not apply, rather than an unknown value."""
    return {"status": "not_applicable", "reason": reason}


def runtime_launch_context_from_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read the launcher/Compose/image identity passed into the container."""
    values = os.environ if environ is None else environ
    keys = {
        "orchestrator": "PIPELINE_LAUNCH_ORCHESTRATOR",
        "launcher": "PIPELINE_LAUNCHER",
        "compose_service": "PIPELINE_COMPOSE_SERVICE",
        "image_reference": "PIPELINE_IMAGE_REFERENCE",
        "image_id": "PIPELINE_IMAGE_ID",
        "repo_digests": "PIPELINE_IMAGE_REPO_DIGESTS_JSON",
    }
    observed = {
        field: values.get(environment_key, "").strip()
        for field, environment_key in keys.items()
    }
    if not any(observed.values()):
        return not_available(
            "The process was not started by a launcher that records container identity"
        )

    def value_or_missing(field: str, reason: str) -> Any:
        return observed[field] or not_available(reason)

    raw_repo_digests = observed["repo_digests"]
    repo_digests: Any
    if not raw_repo_digests:
        repo_digests = not_available(
            "The local image has no recorded repository digest"
        )
    else:
        try:
            parsed = json.loads(raw_repo_digests)
        except json.JSONDecodeError:
            repo_digests = not_available(
                "The launcher supplied malformed image repository-digest JSON"
            )
        else:
            repo_digests = (
                parsed
                if isinstance(parsed, list)
                and parsed
                and all(isinstance(item, str) for item in parsed)
                else not_available(
                    "The launcher did not supply a non-empty image repository-digest list"
                )
            )

    return {
        "status": "observed",
        "orchestrator": value_or_missing(
            "orchestrator", "The launch orchestrator was not recorded"
        ),
        "launcher": value_or_missing(
            "launcher", "The repository launcher was not recorded"
        ),
        "compose_service": value_or_missing(
            "compose_service", "The Compose service was not recorded"
        ),
        "image": {
            "reference": value_or_missing(
                "image_reference", "The container image reference was not recorded"
            ),
            "id": value_or_missing(
                "image_id", "The local container image ID was not available"
            ),
            "repo_digests": repo_digests,
        },
    }


def load_contract_registry(path: str | Path = CONTRACT_REGISTRY_PATH) -> dict[str, Any]:
    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported pipeline contract registry: {registry_path}")
    return payload


def contract_registry_identity(
    path: str | Path = CONTRACT_REGISTRY_PATH,
) -> dict[str, Any]:
    identity = file_identity(path)
    identity["schema_version"] = 1
    return identity


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _contract_refs(value: Any) -> dict[str, Any] | None:
    raw = _mapping(value)
    inference = _mapping(raw.get("inference"))
    if raw.get("schema_version") != 1:
        return None
    required = ("artifact", "frontend", "runtime", "chunking")
    if not isinstance(raw.get("audio_preparation"), str):
        return None
    if any(not isinstance(inference.get(key), str) for key in required):
        return None
    if not isinstance(raw.get("evaluation"), str):
        return None
    if not isinstance(raw.get("runtime_resolution"), str):
        return None
    return {
        "schema_version": 1,
        "audio_preparation": raw["audio_preparation"],
        "inference": {key: inference[key] for key in required},
        "evaluation": raw["evaluation"],
        "runtime_resolution": raw["runtime_resolution"],
    }


def infer_contract_references(profile: Mapping[str, Any]) -> dict[str, Any] | None:
    """Recover contract references from an embedded legacy profile when possible."""
    explicit = _contract_refs(profile.get("pipeline_contract"))
    if explicit is not None:
        return explicit

    framework = profile.get("framework")
    adapter = profile.get("adapter")
    artifact = str(profile.get("artifact") or "")
    profile_id = str(profile.get("id") or "")

    if (framework, adapter) == ("onnxruntime", "android_ctc"):
        runtime = (
            "onnxruntime_shared_image"
            if "published-int8-android-frontend" in profile_id
            else "android_pinned_runtime_proxy"
        )
        return _references(
            audio="android_pcm16_linear_16khz",
            artifact="published_deployment_artifact",
            frontend="deployment_parity_android_frontend",
            runtime=runtime,
            chunking="none",
        )
    if (framework, adapter) == ("onnxruntime", "ctc"):
        artifact_role = (
            "published_deployment_artifact"
            if "swahili-exp41-ctc-android/" in artifact
            else "project_export"
        )
        return _references(
            audio="shared_soundfile_librosa_16khz",
            artifact=artifact_role,
            frontend="compatible_onnx_asr_nemo_frontend",
            runtime="onnxruntime_shared_image",
            chunking="none",
        )
    if (framework, adapter) == ("nemo", "nemo"):
        return _references(
            audio="shared_soundfile_librosa_16khz",
            artifact="canonical_checkpoint",
            frontend="canonical_nemo_model_preprocessor",
            runtime="nemo_image",
            chunking="none",
        )
    if (framework, adapter) == ("sherpa_onnx", "online_transducer"):
        return _references(
            audio="shared_soundfile_librosa_16khz",
            artifact="canonical_checkpoint",
            frontend="canonical_sherpa_online_frontend",
            runtime="sherpa_onnx_image",
            chunking="none",
        )
    if framework == "transformers" and adapter in {"ctc", "speech_seq2seq"}:
        chunking = "none"
        if _mapping(profile.get("decoding")).get("long_form"):
            chunking = "whisper_timestamp_long_form"
        elif profile.get("audio"):
            chunking = "profile_sequential_zero_overlap"
        return _references(
            audio="shared_soundfile_librosa_16khz",
            artifact="canonical_checkpoint",
            frontend="canonical_transformers_auto_processor",
            runtime="shared_transformers_image",
            chunking=chunking,
        )
    if framework == "multimodal" and adapter in {
        "gemma4_audio",
        "phi4_audio",
        "qwen_omni_audio",
    }:
        return _references(
            audio="shared_soundfile_librosa_16khz",
            artifact="canonical_checkpoint",
            frontend="canonical_transformers_auto_processor",
            runtime="multimodal_image",
            chunking=(
                "profile_sequential_zero_overlap" if profile.get("audio") else "none"
            ),
        )
    return None


def _references(
    *,
    audio: str,
    artifact: str,
    frontend: str,
    runtime: str,
    chunking: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audio_preparation": audio,
        "inference": {
            "artifact": artifact,
            "frontend": frontend,
            "runtime": runtime,
            "chunking": chunking,
        },
        "evaluation": "direct_pred_text_by_output_units",
        "runtime_resolution": "observed_effective_values_v1",
    }


def resolve_contract_references(
    references: Mapping[str, Any] | None,
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if references is None:
        missing = not_available("No supported pipeline contract could be recovered")
        return {
            "audio_preparation": missing,
            "inference": {
                "artifact": missing,
                "frontend": missing,
                "runtime": missing,
                "chunking": missing,
            },
            "evaluation": missing,
            "runtime_resolution": missing,
        }
    registry_payload = dict(registry or load_contract_registry())
    inference_registry = _mapping(registry_payload.get("inference"))
    inference_refs = _mapping(references.get("inference"))

    def lookup(section: Mapping[str, Any], key: Any, label: str) -> Any:
        if isinstance(key, str) and key in section:
            return section[key]
        return not_available(f"Unknown {label} contract reference: {key!r}")

    return {
        "audio_preparation": lookup(
            _mapping(registry_payload.get("audio_preparation")),
            references.get("audio_preparation"),
            "audio_preparation",
        ),
        "inference": {
            "artifact": lookup(
                _mapping(inference_registry.get("artifacts")),
                inference_refs.get("artifact"),
                "inference.artifact",
            ),
            "frontend": lookup(
                _mapping(inference_registry.get("frontends")),
                inference_refs.get("frontend"),
                "inference.frontend",
            ),
            "runtime": lookup(
                _mapping(inference_registry.get("runtimes")),
                inference_refs.get("runtime"),
                "inference.runtime",
            ),
            "chunking": lookup(
                _mapping(inference_registry.get("chunking")),
                inference_refs.get("chunking"),
                "inference.chunking",
            ),
        },
        "evaluation": lookup(
            _mapping(registry_payload.get("evaluation")),
            references.get("evaluation"),
            "evaluation",
        ),
        "runtime_resolution": lookup(
            _mapping(registry_payload.get("runtime_resolution")),
            references.get("runtime_resolution"),
            "runtime_resolution",
        ),
    }


def summarize_wav_headers(audio_paths: Iterable[str]) -> dict[str, Any]:
    """Inspect WAV headers only; no samples are decoded or sent to a model."""
    rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    widths: Counter[int] = Counter()
    compression: Counter[str] = Counter()
    inspected = 0
    errors = 0
    for raw_path in audio_paths:
        try:
            with wave.open(str(raw_path), "rb") as wav:
                rates[int(wav.getframerate())] += 1
                channels[int(wav.getnchannels())] += 1
                widths[int(wav.getsampwidth())] += 1
                compression[str(wav.getcomptype())] += 1
                inspected += 1
        except (OSError, EOFError, wave.Error):
            errors += 1
    if inspected == 0:
        return not_available("No readable WAV headers were available")
    return {
        "status": "observed",
        "files_inspected": inspected,
        "inspection_errors": errors,
        "sample_rate_hz_counts": {str(k): v for k, v in sorted(rates.items())},
        "channel_counts": {str(k): v for k, v in sorted(channels.items())},
        "sample_width_byte_counts": {str(k): v for k, v in sorted(widths.items())},
        "compression_type_counts": dict(sorted(compression.items())),
    }


def _first(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if metadata.get(key) is not None:
            return metadata[key]
    return not_available("The backend did not record this effective value")


def _frontend_configuration_identity(
    model_identity: Mapping[str, Any] | None,
    backend: Mapping[str, Any],
) -> Any:
    identity = _mapping(model_identity)
    configuration_files = identity.get("configuration_files")
    if configuration_files:
        return configuration_files
    if backend.get("frontend_reference_revision") is not None:
        return {
            "implementation": backend.get("frontend"),
            "reference_revision": backend["frontend_reference_revision"],
        }
    if backend.get("onnx_asr_version") is not None:
        return {
            "implementation": "onnx-asr",
            "version": backend["onnx_asr_version"],
            "source": "runtime_package",
        }
    if identity.get("kind") == "file" and identity.get("sha256"):
        return {
            "status": "contained_in_artifact",
            "artifact_sha256": identity["sha256"],
            "configuration_path": "model_config.yaml",
        }
    artifact_files = identity.get("artifact_files")
    if artifact_files:
        return {
            "status": "identified_by_artifact_manifest",
            "artifact_manifest_sha256": identity.get("artifact_manifest_sha256"),
            "files": artifact_files,
        }
    return not_available(
        "No processor/frontend configuration identity was recoverable"
    )


def _resampling_observation(
    input_audio_summary: Mapping[str, Any] | None,
    target_sample_rate: Any,
) -> Any:
    summary = _mapping(input_audio_summary)
    counts = summary.get("sample_rate_hz_counts")
    if not isinstance(counts, dict) or not isinstance(target_sample_rate, int):
        return not_available(
            "Input sample-rate counts or the effective target rate were unavailable"
        )
    required = 0
    unchanged = 0
    for raw_rate, raw_count in counts.items():
        try:
            rate = int(raw_rate)
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if rate == target_sample_rate:
            unchanged += count
        else:
            required += count
    return {
        "target_sample_rate_hz": target_sample_rate,
        "files_requiring_resampling": required,
        "files_already_at_target_rate": unchanged,
    }


def _chunking_effective(profile: Mapping[str, Any], backend: Mapping[str, Any]) -> Any:
    if backend.get("chunking") is not None:
        return {"configuration": backend["chunking"], "statistics": backend.get("chunking_stats", not_available("Chunk statistics were not recorded"))}
    if backend.get("long_audio") is not None:
        return {
            "configuration": backend["long_audio"],
            "statistics": {
                key: backend[key]
                for key in ("files_seen", "long_audio_files", "chunks_generated")
                if key in backend
            },
        }
    decoding = _mapping(profile.get("decoding"))
    if decoding.get("long_form") is not None:
        return {"configuration": decoding["long_form"], "statistics": not_available("Historical chunk statistics were not recorded")}
    if profile.get("audio") is not None:
        return {"configuration": profile["audio"], "statistics": not_available("Historical chunk statistics were not recorded")}
    return {"configuration": "none"}


def build_pipeline_provenance(
    *,
    profile: Mapping[str, Any],
    backend: Mapping[str, Any],
    runtime: Mapping[str, Any],
    model_identity: Mapping[str, Any] | None,
    profile_link: Mapping[str, Any] | None,
    run_metadata_path: str | Path,
    transcriptions_path: str | Path,
    input_audio_summary: Mapping[str, Any] | None,
    recording_mode: str,
    limitations: Iterable[str] = (),
) -> dict[str, Any]:
    references = infer_contract_references(profile)
    resolved = resolve_contract_references(references)
    backend_payload = dict(backend)
    runtime_payload = dict(runtime)
    transcript_path = Path(transcriptions_path)
    run_path = Path(run_metadata_path)
    profile_output_units = profile.get("output_units", "orthographic")
    framework = profile.get("framework")
    valid_views = (
        ["ipa"] if profile_output_units == "phoneme" else ["orthographic", "ipa"]
    )
    processor_config_identity = _frontend_configuration_identity(
        model_identity, backend_payload
    )
    selected_frontend = {
        "processor_class": _first(backend_payload, "processor_class", "frontend"),
        "feature_configuration_identity": processor_config_identity,
        "target_sample_rate_hz": _first(
            backend_payload, "sampling_rate", "sample_rate", "target_sample_rate"
        ),
    }
    decoder_strategy = _first(
        backend_payload,
        "effective_decoder_type",
        "decoding_strategy",
        "strategy",
        "decoder_type",
    )
    if isinstance(decoder_strategy, dict) and decoder_strategy.get("status") == "not_available":
        profile_strategy = _mapping(profile.get("decoding")).get("strategy")
        if isinstance(profile_strategy, str):
            decoder_strategy = {
                "value": profile_strategy,
                "source": "embedded_profile_selected_by_completed_run",
            }
    decoder_observed = {
        "profile": _mapping(profile.get("decoding")),
        "effective_strategy": decoder_strategy,
        "backend_call": _first(
            backend_payload,
            "effective_decoding_config",
            "generation",
            "recognizer_call",
            "decoder_call",
        ),
    }
    links = {
        "profile": dict(profile_link) if profile_link else not_available("The source profile path or identity was not recorded"),
        "contract_registry": contract_registry_identity(),
        "run_metadata": {"path": str(run_path), "relation": "self"},
        "transcriptions": file_identity(transcript_path),
        "expected_evaluation_root": str(
            run_path.parent.parent.parent / "evaluations" / run_path.parent.name
            if run_path.parent.parent.name == "transcripts"
            else run_path.parent / "evaluations"
        ),
    }
    return {
        "schema_version": PIPELINE_PROVENANCE_SCHEMA_VERSION,
        "recording": {
            "mode": recording_mode,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "limitations": list(limitations),
        },
        "contract": {
            "references": references or not_available("The embedded profile is not recognized by the contract registry"),
            "resolved": resolved,
            "references_sha256": canonical_json_sha256(references) if references else None,
            "runtime_resolution": resolved["runtime_resolution"],
        },
        "stages": {
            "audio_preparation": {
                "contract": resolved["audio_preparation"],
                "input_audio": dict(input_audio_summary) if input_audio_summary else not_available("Input audio headers were not recorded or recoverable"),
                "effective_target_sample_rate_hz": selected_frontend["target_sample_rate_hz"],
                "observed_resampling_need": _resampling_observation(
                    input_audio_summary,
                    selected_frontend["target_sample_rate_hz"],
                ),
            },
            "inference": {
                "artifact": {
                    "contract": resolved["inference"]["artifact"],
                    "identity": dict(model_identity) if model_identity else not_available("The selected artifact identity was not recorded"),
                },
                "frontend": {
                    "contract": resolved["inference"]["frontend"],
                    "observed": selected_frontend,
                },
                "runtime": {
                    "contract": resolved["inference"]["runtime"],
                    "observed": {
                        "launch_context": runtime_payload.get(
                            "launch_context",
                            not_available("Container launch identity was not recorded"),
                        ),
                        "python": runtime_payload.get("python", not_available("Python version was not recorded")),
                        "platform": runtime_payload.get("platform", not_available("Platform was not recorded")),
                        "batch_size": runtime_payload.get("batch_size", not_available("Batch size was not recorded")),
                        "packages": runtime_payload.get("packages", not_available("Package versions were not recorded")),
                        "argv": runtime_payload.get("argv", not_available("Invocation arguments were not recorded")),
                        "device": backend_payload.get("device", not_available("Device was not recorded")),
                        "provider": backend_payload.get(
                            "provider",
                            (
                                not_applicable(
                                    "This backend records a device rather than an execution provider"
                                )
                                if framework in {"transformers", "multimodal", "nemo"}
                                else not_available(
                                    "Execution provider was not recorded"
                                )
                            ),
                        ),
                        "requested_torch_dtype": backend_payload.get(
                            "requested_torch_dtype",
                            _mapping(profile.get("loader")).get(
                                "torch_dtype",
                                not_available("Requested dtype was not recorded"),
                            ),
                        ),
                        "effective_torch_dtype": backend_payload.get(
                            "effective_torch_dtype",
                            (
                                not_applicable(
                                    "This backend does not execute a PyTorch model"
                                )
                                if framework in {"onnxruntime", "sherpa_onnx"}
                                else not_available("Effective dtype was not recorded")
                            ),
                        ),
                        "attention_implementation": backend_payload.get(
                            "attention_implementation",
                            (
                                not_applicable(
                                    "This backend has no Transformers attention implementation"
                                )
                                if framework in {"nemo", "onnxruntime", "sherpa_onnx"}
                                else not_available(
                                    "Effective attention implementation was not recorded"
                                )
                            ),
                        ),
                        "threads_or_workers": {
                            key: backend_payload[key]
                            for key in (
                                "num_threads",
                                "num_workers",
                                "decoder_workers",
                                "batch_size_required",
                            )
                            if key in backend_payload
                        }
                        or not_available("Thread or worker controls were not recorded"),
                        "backend_versions": {
                            key: value
                            for key, value in backend_payload.items()
                            if "version" in key and value is not None
                        }
                        or not_available("Backend-specific versions were not recorded"),
                    },
                },
                "chunking": {
                    "contract": resolved["inference"]["chunking"],
                    "observed": _chunking_effective(profile, backend_payload),
                },
                "decoder": decoder_observed,
            },
            "evaluation": {
                "contract": resolved["evaluation"],
                "handoff": {
                    "hypothesis_field": "pred_text",
                    "text_postprocessing": "none",
                    "model_output_units": profile_output_units,
                    "model_output_notation": profile.get(
                        "output_notation",
                        not_applicable(
                            "Orthographic output has no separate notation field"
                        ),
                    ),
                    "compatible_scoring_views": valid_views,
                },
            },
        },
        "links": links,
    }


def evaluation_source_run_metadata_path(manifest_in: str | Path) -> Path | None:
    manifest_path = Path(manifest_in)
    if manifest_path.parent.name != "manifests":
        return None
    run_dir = manifest_path.parent.parent
    if run_dir.parent.name != "evaluations":
        return None
    return run_dir.parent.parent / "transcripts" / run_dir.name / "run_metadata.json"


def build_evaluation_provenance(
    *,
    base: str | Path,
    manifest_in: str | Path,
    requested_representation: str,
    scoring_units: str,
    namespace: str,
    reference_metadata_path: str | Path | None = None,
    recording_mode: str = "evaluation_time",
    limitations: Iterable[str] = (),
    run_metadata_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base_path = Path(base)
    run_metadata_path = evaluation_source_run_metadata_path(manifest_in)
    run_metadata: dict[str, Any] = dict(run_metadata_override or {})
    if (
        not run_metadata
        and run_metadata_path is not None
        and run_metadata_path.is_file()
    ):
        try:
            loaded = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                run_metadata = loaded
        except (OSError, json.JSONDecodeError):
            pass
    run_pipeline = _mapping(run_metadata.get("pipeline_provenance"))
    run_evaluation_stage = _mapping(_mapping(run_pipeline.get("stages")).get("evaluation"))
    profile = _mapping(run_metadata.get("profile"))
    source_run_link: Any
    if run_metadata_path is not None and run_metadata_path.is_file():
        source_run_link = file_identity(run_metadata_path)
    else:
        source_run_link = not_available("No standard source run metadata was found")
    profile_link = _mapping(_mapping(run_pipeline.get("links")).get("profile"))
    if not profile_link:
        recorded_profile = run_metadata.get("profile")
        if isinstance(recorded_profile, dict):
            profile_link = {
                "embedded_profile_sha256": canonical_json_sha256(recorded_profile),
                "recorded_path": run_metadata.get(
                    "profile_path",
                    not_available("The source profile path was not recorded"),
                ),
                "recorded_identity": run_metadata.get(
                    "profile_identity",
                    not_available("The source profile file identity was not recorded"),
                ),
            }
        else:
            profile_link = not_available(
                "The source run has no recoverable profile link"
            )
    reference_link: Any = not_available(
        "Reference-view metadata path was not recorded for this evaluation"
    )
    if reference_metadata_path is not None:
        reference_link = file_identity(reference_metadata_path)
    return {
        "schema_version": PIPELINE_PROVENANCE_SCHEMA_VERSION,
        "recording": {
            "mode": recording_mode,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "limitations": list(limitations),
        },
        "stage": "evaluation",
        "contract": run_evaluation_stage.get(
            "contract",
            resolve_contract_references(infer_contract_references(profile))["evaluation"],
        ),
        "effective": {
            "requested_scoring_representation": requested_representation,
            "effective_scoring_units": scoring_units,
            "output_namespace": namespace,
            "hypothesis_field": "pred_text",
            "text_postprocessing": "none",
            "representation_compatible": requested_representation != "legacy_orthographic",
            "reference_integrity_enforced": requested_representation != "legacy_orthographic",
        },
        "links": {
            "source_manifest": file_identity(manifest_in),
            "source_run_metadata": source_run_link,
            "source_profile": profile_link,
            "reference_view_metadata": reference_link,
            "detailed_scores": file_identity(base_path / "egra_eval_detailed.csv"),
            "summary": file_identity(base_path / "egra_eval_summary.txt"),
        },
    }
