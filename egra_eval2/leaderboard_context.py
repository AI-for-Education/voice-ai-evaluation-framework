"""Normalize inference provenance into concise leaderboard context fields.

This module deliberately contains no ranking or file-discovery logic.  It turns
the profile and completed-run metadata already selected by the leaderboard into
stable, human-readable descriptions of the artifact, preprocessing, runtime,
platform, and decoder.
"""

from __future__ import annotations

import re
from typing import Any

from inference.pipeline_provenance import infer_contract_references


def nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def decoding_label(profile: dict[str, Any]) -> str:
    """Return a concise, presentation-ready decoder description."""
    decoding = profile.get("decoding")
    if not isinstance(decoding, dict):
        if profile.get("adapter") == "nemo":
            return "native CTC decoder"
        return "not recorded"

    strategy = decoding.get("strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        return "not recorded"
    strategy = strategy.strip().lower()

    generation = decoding.get("generation_kwargs")
    generation = generation if isinstance(generation, dict) else {}
    beams = nonnegative_int(generation.get("num_beams"))
    if strategy == "generate":
        return f"{beams}-beam search" if beams > 1 else "greedy generation"

    if strategy == "beam_search":
        language_model = decoding.get("ctc_lm_kwargs")
        language_model = language_model if isinstance(language_model, dict) else {}
        width = nonnegative_int(language_model.get("beam_width"))
        return f"5-gram LM beam, width {width}" if width else "beam search"

    if strategy == "modified_beam_search":
        search = decoding.get("transducer_search_kwargs")
        search = search if isinstance(search, dict) else {}
        paths = nonnegative_int(search.get("max_active_paths"))
        return (
            f"modified beam search, {paths} paths" if paths else "modified beam search"
        )

    if strategy in {"greedy", "greedy_search"}:
        return "greedy"
    if strategy == "ctc":
        return "native CTC decoder"
    return strategy.replace("_", " ")


def contract_references(
    profile: dict[str, Any],
    run_metadata: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return the best available pipeline contract and its evidence level."""
    pipeline = run_metadata.get("pipeline_provenance")
    pipeline = pipeline if isinstance(pipeline, dict) else {}
    contract = pipeline.get("contract")
    contract = contract if isinstance(contract, dict) else {}
    references = contract.get("references")
    if isinstance(references, dict) and references.get("schema_version") == 1:
        recording = pipeline.get("recording")
        recording = recording if isinstance(recording, dict) else {}
        mode = str(recording.get("mode") or "recording mode not available")
        return references, f"pipeline contract ({mode.replace('_', ' ')})"

    recovered = infer_contract_references(profile)
    if recovered is not None:
        return recovered, "profile-derived contract fallback"
    return {}, "legacy profile/backend fallback"


def _inference_reference(references: dict[str, Any], field: str) -> str:
    inference = references.get("inference")
    inference = inference if isinstance(inference, dict) else {}
    value = inference.get(field)
    return value if isinstance(value, str) else ""


def artifact_context(
    profile: dict[str, Any],
    run_metadata: dict[str, Any],
    references: dict[str, Any],
) -> str:
    artifact = _inference_reference(references, "artifact")
    labels = {
        "canonical_checkpoint": "canonical model-owner artifact",
        "project_export": "project-exported ONNX artifact",
        "published_deployment_artifact": "published Android deployment artifact",
    }
    if artifact in labels:
        return labels[artifact]

    backend = run_metadata.get("backend")
    backend = backend if isinstance(backend, dict) else {}
    if backend.get("artifact_target") == "packaged_android_reference":
        return "published Android deployment artifact"
    if profile.get("framework") == "onnxruntime":
        return "ONNX artifact (role not recorded)"
    if profile.get("artifact"):
        return "selected artifact (role not recorded)"
    return "artifact not recorded"


def preprocessing_context(references: dict[str, Any]) -> str:
    audio = references.get("audio_preparation")
    audio = audio if isinstance(audio, str) else ""
    frontend = _inference_reference(references, "frontend")
    chunking = _inference_reference(references, "chunking")

    audio_labels = {
        "shared_soundfile_librosa_16khz": "shared SoundFile/librosa 16 kHz audio",
        "android_pcm16_linear_16khz": "Android PCM16/linear 16 kHz audio",
    }
    frontend_labels = {
        "canonical_transformers_auto_processor": "canonical Transformers frontend",
        "canonical_nemo_model_preprocessor": "canonical NeMo frontend",
        "compatible_onnx_asr_nemo_frontend": "compatible onnx-asr NeMo frontend",
        "deployment_parity_android_frontend": "Android deployment-parity frontend",
        "canonical_sherpa_online_frontend": "canonical Sherpa-ONNX frontend",
    }
    chunking_labels = {
        "none": "no chunking",
        "profile_sequential_zero_overlap": (
            "profile-owned sequential zero-overlap chunking"
        ),
        "whisper_timestamp_long_form": "Whisper timestamp long-form chunking",
    }
    parts = [
        audio_labels.get(
            audio, audio.replace("_", " ") if audio else "audio policy not recorded"
        ),
        frontend_labels.get(
            frontend,
            frontend.replace("_", " ") if frontend else "frontend not recorded",
        ),
        chunking_labels.get(
            chunking,
            chunking.replace("_", " ") if chunking else "chunking not recorded",
        ),
    ]
    return "; ".join(parts)


def runtime_version(run_metadata: dict[str, Any]) -> tuple[str, str]:
    runtime = run_metadata.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    packages = runtime.get("packages")
    packages = packages if isinstance(packages, dict) else {}
    backend = run_metadata.get("backend")
    backend = backend if isinstance(backend, dict) else {}
    observed = (
        packages.get("onnxruntime")
        or packages.get("onnxruntime-gpu")
        or backend.get("onnxruntime_version")
    )
    required = backend.get("onnxruntime_version_required")
    return str(observed or ""), str(required or "")


def runtime_context(
    profile: dict[str, Any],
    run_metadata: dict[str, Any],
    references: dict[str, Any],
) -> tuple[str, bool]:
    """Describe runtime independently from the selected artifact and frontend."""
    runtime = run_metadata.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    launch = runtime.get("launch_context")
    launch = launch if isinstance(launch, dict) else {}
    observed_version, required_version = runtime_version(run_metadata)
    if profile.get("framework") != "onnxruntime":
        observed_version = ""
        required_version = ""

    if launch.get("status") == "observed":
        service = launch.get("compose_service")
        image = launch.get("image")
        image = image if isinstance(image, dict) else {}
        image_reference = image.get("reference")
        service_labels = {
            "nemo-asr": "shared ASR image / NeMo service",
            "transformers-asr": "shared ASR image / Transformers service",
            "sherpa-onnx-asr": "shared ASR image / Sherpa-ONNX service",
            "onnxruntime-asr": "shared ASR image / ONNX Runtime service",
            "onnxruntime-android-asr": (
                "dedicated Android-parity image / ONNX Runtime service (PC proxy)"
            ),
            "multimodal-asr": "shared ASR image / multimodal service",
            "phi4-multimodal-asr": "dedicated Phi-4 multimodal image",
            "qwen-omni-asr": "dedicated Qwen Omni image",
        }
        label = service_labels.get(
            service,
            (
                f"Compose service {service}"
                if isinstance(service, str)
                else "observed container launch"
            ),
        )
        details = []
        if isinstance(image_reference, str):
            details.append(image_reference)
        if observed_version:
            details.append(f"ONNX Runtime {observed_version}")
        elif required_version:
            details.append(f"ONNX Runtime {required_version} required")
        return (f"{label} ({'; '.join(details)})" if details else label), True

    runtime_reference = _inference_reference(references, "runtime")
    labels = {
        "shared_transformers_image": "shared ASR image / Transformers runtime",
        "multimodal_image": "model-family multimodal image",
        "nemo_image": "shared ASR image / NeMo runtime",
        "onnxruntime_shared_image": "shared ASR image / ONNX Runtime",
        "android_pinned_runtime_proxy": "dedicated Android-parity runtime (PC proxy)",
        "sherpa_onnx_image": "shared ASR image / Sherpa-ONNX runtime",
    }
    if runtime_reference in labels:
        label = labels[runtime_reference]
        if observed_version:
            label += f" (ONNX Runtime {observed_version} observed)"
        elif required_version:
            label += f" (ONNX Runtime {required_version} required)"
        return label, False

    framework = profile.get("framework")
    adapter = profile.get("adapter")
    if framework == "onnxruntime":
        backend = run_metadata.get("backend")
        backend = backend if isinstance(backend, dict) else {}
        quantization = backend.get("quantization")
        precision = "INT8" if quantization == "int8" else "FP32"
        return f"ONNX {precision} runtime (image not recorded)", False
    if adapter == "phi4_audio":
        return "Phi-4 multimodal runtime (image not recorded)", False
    if adapter == "qwen_omni_audio":
        return "Qwen multimodal runtime (image not recorded)", False
    if adapter == "gemma4_audio":
        return "Gemma multimodal runtime (image not recorded)", False

    names = {
        "nemo": "NeMo runtime (image not recorded)",
        "transformers": "Transformers runtime (image not recorded)",
        "sherpa_onnx": "Sherpa-ONNX runtime (image not recorded)",
    }
    if isinstance(framework, str) and framework:
        return names.get(framework, framework.replace("_", " ")), False
    return "runtime not recorded", False


def platform_label(
    profile: dict[str, Any],
    run_metadata: dict[str, Any],
    references: dict[str, Any],
) -> str:
    """Return the normalized execution platform used in presentation labels."""
    runtime = run_metadata.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    launch = runtime.get("launch_context")
    launch = launch if isinstance(launch, dict) else {}
    service = launch.get("compose_service")
    service_labels = {
        "nemo-asr": "nemo",
        "transformers-asr": "transformers",
        "sherpa-onnx-asr": "sherpa-onnx",
        "onnxruntime-asr": "onnxruntime-desktop",
        "onnxruntime-android-asr": "onnxruntime-android-proxy",
        "multimodal-asr": "multimodal-shared",
        "phi4-multimodal-asr": "multimodal-phi4",
        "qwen-omni-asr": "multimodal-qwen-omni",
    }
    if service in service_labels:
        return service_labels[service]

    runtime_reference = _inference_reference(references, "runtime")
    runtime_labels = {
        "shared_transformers_image": "transformers",
        "nemo_image": "nemo",
        "onnxruntime_shared_image": "onnxruntime-desktop",
        "android_pinned_runtime_proxy": "onnxruntime-android-proxy",
        "sherpa_onnx_image": "sherpa-onnx",
    }
    if runtime_reference in runtime_labels:
        return runtime_labels[runtime_reference]

    adapter = profile.get("adapter")
    multimodal_labels = {
        "gemma4_audio": "multimodal-gemma",
        "phi4_audio": "multimodal-phi4",
        "qwen_omni_audio": "multimodal-qwen-omni",
    }
    if adapter in multimodal_labels:
        return multimodal_labels[adapter]

    framework = profile.get("framework")
    if isinstance(framework, str) and framework:
        return framework.strip().lower().replace("_", "-")
    return "platform-not-recorded"


def decoder_slug(label: str) -> str:
    """Normalize a readable decoder description for legacy display labels."""
    normalized = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return normalized or "decoder-not-recorded"
