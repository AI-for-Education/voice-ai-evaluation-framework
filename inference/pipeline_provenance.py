"""Normalized provenance for inference runs and their evaluations."""

from __future__ import annotations

import json
import os
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from inference.provenance import canonical_json_sha256, file_identity


PIPELINE_PROVENANCE_SCHEMA_VERSION = 2
CONTRACT_REGISTRY_PATH = Path(__file__).with_name("pipeline_contracts.json")


def not_available(reason: str) -> dict[str, str]:
    """Return the one machine-readable representation of unavailable history."""
    return {"status": "not_available", "reason": reason}


def not_applicable(reason: str) -> dict[str, str]:
    """Represent a field that does not apply, rather than an unknown value."""
    return {"status": "not_applicable", "reason": reason}


def execution_environment_from_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read the launcher, container, and image identity for this process."""
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
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError(f"Unsupported pipeline contract registry: {registry_path}")
    return payload


def contract_registry_identity(
    path: str | Path = CONTRACT_REGISTRY_PATH,
) -> dict[str, Any]:
    identity = file_identity(path)
    identity["schema_version"] = 2
    return identity


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def inference_profile_from_run_metadata(
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Read current or historical embedded profiles into current field names."""
    raw = run_metadata.get("inference_profile")
    if not isinstance(raw, Mapping):
        raw = run_metadata.get("profile")
    profile = _mapping(raw)
    if "inference_setup_id" not in profile and isinstance(profile.get("id"), str):
        profile["inference_setup_id"] = profile["id"]
    if "inference_library" not in profile and isinstance(
        profile.get("framework"), str
    ):
        profile["inference_library"] = profile["framework"]
    return profile


def _contract_refs(value: Any) -> dict[str, Any] | None:
    raw = _mapping(value)
    if raw.get("schema_version") != 2:
        return None
    required = (
        "audio_preparation",
        "model_artifact",
        "input_processing",
        "execution_stack",
        "chunking",
        "evaluation",
        "observation_policy",
    )
    if any(not isinstance(raw.get(key), str) for key in required):
        return None
    return {"schema_version": 2, **{key: raw[key] for key in required}}


def infer_contract_references(profile: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read schema-v2 contract references from an inference profile."""
    return _contract_refs(profile.get("pipeline_contract"))


def resolve_contract_references(
    references: Mapping[str, Any] | None,
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if references is None:
        missing = not_available("No supported pipeline contract could be recovered")
        return {
            "audio_preparation": missing,
            "model_artifact": missing,
            "input_processing": missing,
            "execution_stack": missing,
            "chunking": missing,
            "evaluation": missing,
            "observation_policy": missing,
        }
    registry_payload = dict(registry or load_contract_registry())
    inference_registry = _mapping(registry_payload.get("inference"))

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
        "model_artifact": lookup(
            _mapping(inference_registry.get("model_artifacts")),
            references.get("model_artifact"),
            "model_artifact",
        ),
        "input_processing": lookup(
            _mapping(inference_registry.get("input_processing")),
            references.get("input_processing"),
            "input_processing",
        ),
        "execution_stack": lookup(
            _mapping(inference_registry.get("execution_stacks")),
            references.get("execution_stack"),
            "execution_stack",
        ),
        "chunking": lookup(
            _mapping(inference_registry.get("chunking")),
            references.get("chunking"),
            "chunking",
        ),
        "evaluation": lookup(
            _mapping(registry_payload.get("evaluation")),
            references.get("evaluation"),
            "evaluation",
        ),
        "observation_policy": lookup(
            _mapping(registry_payload.get("observation_policies")),
            references.get("observation_policy"),
            "observation_policy",
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
    return not_available("The inference adapter did not record this effective value")


def _frontend_configuration_identity(
    model_artifact_identity: Mapping[str, Any] | None,
    adapter_metadata: Mapping[str, Any],
) -> Any:
    identity = _mapping(model_artifact_identity)
    configuration_files = identity.get("configuration_files")
    if configuration_files:
        return configuration_files
    if adapter_metadata.get("frontend_reference_revision") is not None:
        return {
            "implementation": adapter_metadata.get("frontend"),
            "reference_revision": adapter_metadata["frontend_reference_revision"],
        }
    if adapter_metadata.get("onnx_asr_version") is not None:
        return {
            "implementation": "onnx-asr",
            "version": adapter_metadata["onnx_asr_version"],
            "source": "execution_stack_package",
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


def _chunking_effective(
    profile: Mapping[str, Any],
    adapter_metadata: Mapping[str, Any],
) -> Any:
    if adapter_metadata.get("chunking") is not None:
        return {
            "configuration": adapter_metadata["chunking"],
            "statistics": adapter_metadata.get(
                "chunking_stats",
                not_available("Chunk statistics were not recorded"),
            ),
        }
    if adapter_metadata.get("long_audio") is not None:
        return {
            "configuration": adapter_metadata["long_audio"],
            "statistics": {
                key: adapter_metadata[key]
                for key in ("files_seen", "long_audio_files", "chunks_generated")
                if key in adapter_metadata
            },
        }
    decoding = _mapping(profile.get("decoding"))
    if decoding.get("long_form") is not None:
        return {
            "configuration": decoding["long_form"],
            "statistics": not_available(
                "Historical chunk statistics were not recorded"
            ),
        }
    if profile.get("audio") is not None:
        return {
            "configuration": profile["audio"],
            "statistics": not_available(
                "Historical chunk statistics were not recorded"
            ),
        }
    return {"configuration": "none"}


def build_pipeline_provenance(
    *,
    profile: Mapping[str, Any],
    adapter_metadata: Mapping[str, Any],
    execution_stack: Mapping[str, Any],
    model_artifact_identity: Mapping[str, Any] | None,
    profile_link: Mapping[str, Any] | None,
    run_metadata_path: str | Path,
    transcriptions_path: str | Path,
    input_audio_summary: Mapping[str, Any] | None,
    recording_mode: str,
    published_transcriptions_path: str | Path | None = None,
    limitations: Iterable[str] = (),
) -> dict[str, Any]:
    references = infer_contract_references(profile)
    resolved = resolve_contract_references(references)
    adapter_payload = dict(adapter_metadata)
    execution_stack_payload = dict(execution_stack)
    transcript_path = Path(transcriptions_path)
    published_transcript_path = Path(
        published_transcriptions_path or transcriptions_path
    )
    run_path = Path(run_metadata_path)
    profile_output_units = profile.get("output_units", "orthographic")
    inference_library = profile.get("inference_library")
    valid_views = (
        ["ipa"] if profile_output_units == "phoneme" else ["orthographic", "ipa"]
    )
    processor_config_identity = _frontend_configuration_identity(
        model_artifact_identity,
        adapter_payload,
    )
    selected_frontend = {
        "processor_class": _first(adapter_payload, "processor_class", "frontend"),
        "feature_configuration_identity": processor_config_identity,
        "target_sample_rate_hz": _first(
            adapter_payload, "sampling_rate", "sample_rate", "target_sample_rate"
        ),
    }
    decoder_strategy = _first(
        adapter_payload,
        "effective_decoder_type",
        "decoding_strategy",
        "strategy",
        "decoder_type",
    )
    if (
        isinstance(decoder_strategy, dict)
        and decoder_strategy.get("status") == "not_available"
    ):
        profile_strategy = _mapping(profile.get("decoding")).get("strategy")
        if isinstance(profile_strategy, str):
            decoder_strategy = {
                "value": profile_strategy,
                "source": "embedded_profile_selected_by_completed_run",
            }
    decoder_observed = {
        "profile": _mapping(profile.get("decoding")),
        "effective_strategy": decoder_strategy,
        "adapter_call": _first(
            adapter_payload,
            "effective_decoding_config",
            "generation",
            "recognizer_call",
            "decoder_call",
        ),
    }
    links = {
        "profile": (
            dict(profile_link)
            if profile_link
            else not_available(
                "The source profile path or identity was not recorded"
            )
        ),
        "contract_registry": contract_registry_identity(),
        "run_metadata": {"path": str(run_path), "relation": "self"},
        "transcriptions": {
            **file_identity(transcript_path),
            "path": str(published_transcript_path),
        },
        "expected_evaluation_root": str(
            run_path.parent.parent.parent / "evaluations" / run_path.parent.name
            if run_path.parent.parent.name == "transcripts"
            else run_path.parent / "evaluations"
        ),
    }
    audio_preparation = {
        "contract": resolved["audio_preparation"],
        "input_audio": (
            dict(input_audio_summary)
            if input_audio_summary
            else not_available("Input audio headers were not recorded or recoverable")
        ),
        "effective_target_sample_rate_hz": selected_frontend[
            "target_sample_rate_hz"
        ],
        "observed_resampling_need": _resampling_observation(
            input_audio_summary,
            selected_frontend["target_sample_rate_hz"],
        ),
    }
    model_artifact = {
        "contract": resolved["model_artifact"],
        "identity": (
            dict(model_artifact_identity)
            if model_artifact_identity
            else not_available("The selected model-artifact identity was not recorded")
        ),
    }
    input_processing = {
        "contract": resolved["input_processing"],
        "observed": selected_frontend,
    }
    chunking = {
        "contract": resolved["chunking"],
        "observed": _chunking_effective(profile, adapter_payload),
    }
    execution_stack_observed = {
        "runner": {
            "argv": execution_stack_payload.get(
                "argv", not_available("Invocation arguments were not recorded")
            ),
            "adapter": adapter_payload.get(
                "adapter", not_available("Inference adapter was not recorded")
            ),
            "batch_size": execution_stack_payload.get(
                "batch_size", not_available("Batch size was not recorded")
            ),
        },
        "inference_library": {
            "name": inference_library
            or not_available("Inference library was not recorded"),
            "packages": execution_stack_payload.get(
                "packages", not_available("Package versions were not recorded")
            ),
        },
        "inference_engine": {
            "name": _mapping(resolved["execution_stack"]).get(
                "inference_engine",
                not_available("Inference engine was not recorded"),
            ),
            "provider": adapter_payload.get(
                "provider",
                (
                    not_applicable(
                        "This inference adapter records a device rather than an execution provider"
                    )
                    if inference_library in {
                        "transformers",
                        "multimodal",
                        "nemo",
                        "torch",
                    }
                    else not_available("Execution provider was not recorded")
                ),
            ),
            "versions": {
                key: value
                for key, value in adapter_payload.items()
                if "version" in key and value is not None
            }
            or not_available("Engine or adapter versions were not recorded"),
        },
        "environment": {
            "launch": execution_stack_payload.get(
                "launch_context",
                not_available("Container launch identity was not recorded"),
            ),
            "python": execution_stack_payload.get(
                "python", not_available("Python version was not recorded")
            ),
            "platform": execution_stack_payload.get(
                "platform", not_available("Operating platform was not recorded")
            ),
        },
        "hardware": {
            "device": adapter_payload.get(
                "device", not_available("Device was not recorded")
            ),
            "requested_torch_dtype": adapter_payload.get(
                "requested_torch_dtype",
                _mapping(profile.get("loader")).get(
                    "torch_dtype", not_available("Requested dtype was not recorded")
                ),
            ),
            "effective_torch_dtype": adapter_payload.get(
                "effective_torch_dtype",
                (
                    not_applicable("This setup does not execute a PyTorch model")
                    if inference_library in {"onnxruntime", "sherpa_onnx"}
                    else not_available("Effective dtype was not recorded")
                ),
            ),
            "attention_implementation": adapter_payload.get(
                "attention_implementation",
                (
                    not_applicable(
                        "This setup has no Transformers attention implementation"
                    )
                    if inference_library
                    in {"nemo", "onnxruntime", "sherpa_onnx", "torch"}
                    else not_available(
                        "Effective attention implementation was not recorded"
                    )
                ),
            ),
            "threads_or_workers": {
                key: adapter_payload[key]
                for key in (
                    "num_threads",
                    "num_workers",
                    "decoder_workers",
                    "batch_size_required",
                )
                if key in adapter_payload
            }
            or not_available("Thread or worker controls were not recorded"),
        },
    }
    execution_stack = {
        "contract": resolved["execution_stack"],
        "observed": execution_stack_observed,
    }
    evaluation = {
        "contract": resolved["evaluation"],
        "handoff": {
            "hypothesis_field": "pred_text",
            "text_postprocessing": "none",
            "model_output_units": profile_output_units,
            "model_output_notation": profile.get(
                "output_notation",
                not_applicable("Orthographic output has no separate notation field"),
            ),
            "compatible_scoring_views": valid_views,
        },
    }
    return {
        "schema_version": PIPELINE_PROVENANCE_SCHEMA_VERSION,
        "recording": {
            "mode": recording_mode,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "limitations": list(limitations),
        },
        "contract": {
            "references": references
            or not_available(
                "The embedded inference profile is not recognized by the contract registry"
            ),
            "resolved": resolved,
            "references_sha256": (
                canonical_json_sha256(references) if references else None
            ),
            "observation_policy": resolved["observation_policy"],
        },
        "model_artifact": model_artifact,
        "inference_setup": {
            "audio_preparation": audio_preparation,
            "input_processing": input_processing,
            "chunking": chunking,
            "decoding": decoder_observed,
        },
        "execution_stack": execution_stack,
        "evaluation": evaluation,
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


def _mutable_reference_index_link(path: str | Path) -> dict[str, Any]:
    """Link a shared mutable index without treating its current bytes as identity."""
    index_path = Path(path)
    return {
        "path": str(index_path),
        "exists": index_path.is_file(),
        "identity_scope": "mutable_shared_index",
        "content_hash_recorded": False,
    }


def build_evaluation_provenance(
    *,
    base: str | Path,
    manifest_in: str | Path,
    requested_representation: str,
    scoring_units: str,
    namespace: str,
    reference_metadata_path: str | Path | None = None,
    reference_view_path: str | Path | None = None,
    aligned_manifest_path: str | Path | None = None,
    g2p_system: Mapping[str, Any] | None = None,
    hypothesis_route: str = "native orthographic",
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
    run_evaluation_stage = _mapping(run_pipeline.get("evaluation"))
    profile = inference_profile_from_run_metadata(run_metadata)
    source_run_link: Any
    if run_metadata_path is not None and run_metadata_path.is_file():
        source_run_link = file_identity(run_metadata_path)
    else:
        source_run_link = not_available("No standard source run metadata was found")
    profile_link = _mapping(_mapping(run_pipeline.get("links")).get("profile"))
    if not profile_link:
        recorded_profile = run_metadata.get("inference_profile")
        if not isinstance(recorded_profile, dict):
            recorded_profile = run_metadata.get("profile")
        if isinstance(recorded_profile, dict):
            profile_link = {
                "embedded_profile_sha256": canonical_json_sha256(recorded_profile),
                "recorded_path": run_metadata.get(
                    "inference_profile_path",
                    not_available("The source profile path was not recorded"),
                ),
                "recorded_identity": run_metadata.get(
                    "inference_profile_identity",
                    not_available(
                        "The source profile file identity was not recorded"
                    ),
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
        reference_link = _mutable_reference_index_link(reference_metadata_path)
    reference_view_link: Any = not_available(
        "No exact G2P reference view applies to this evaluation"
    )
    if reference_view_path is not None:
        reference_view_link = file_identity(reference_view_path)
    aligned_manifest_link: Any = not_available(
        "No aligned IPA manifest applies to this evaluation"
    )
    if aligned_manifest_path is not None:
        aligned_manifest_link = file_identity(aligned_manifest_path)
    source_manifest_link = file_identity(manifest_in)
    if g2p_system is not None:
        reference_identity: dict[str, Any] = {
            "kind": "exact_g2p_reference_view",
            "system_id": g2p_system.get(
                "system_id", not_available("G2P system ID was not recorded")
            ),
            "g2p_identity_sha256": g2p_system.get(
                "identity_sha256",
                not_available("G2P identity hash was not recorded"),
            ),
            "inventory": g2p_system.get(
                "inventory", not_available("G2P inventory was not recorded")
            ),
            "reference_view": reference_view_link,
        }
    else:
        reference_identity = {
            "kind": "evaluation_source_manifest",
            "source_manifest": source_manifest_link,
        }
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
            "text_postprocessing": (
                hypothesis_route if g2p_system is not None else "none"
            ),
            "representation_compatible": True,
            "reference_integrity_enforced": True,
            "hypothesis_route": hypothesis_route,
            "reference_identity": reference_identity,
            "g2p_system": (
                dict(g2p_system)
                if g2p_system is not None
                else not_applicable("Orthographic scoring does not use G2P")
            ),
        },
        "links": {
            "source_manifest": source_manifest_link,
            "source_run_metadata": source_run_link,
            "source_profile": profile_link,
            "reference_view_metadata": reference_link,
            "reference_view": reference_view_link,
            "aligned_ipa_manifest": aligned_manifest_link,
            "detailed_scores": file_identity(base_path / "egra_eval_detailed.csv"),
            "summary": file_identity(base_path / "egra_eval_summary.txt"),
        },
    }
