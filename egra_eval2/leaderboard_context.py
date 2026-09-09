"""Normalize inference provenance into concise leaderboard context fields.

This module deliberately contains no ranking or file-discovery logic.  It turns
the profile and completed-run metadata already selected by the leaderboard into
stable, human-readable descriptions of the model artifact, inference setup,
execution stack, and decoding.
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
    if isinstance(references, dict) and references.get("schema_version") == 2:
        recording = pipeline.get("recording")
        recording = recording if isinstance(recording, dict) else {}
        mode = str(recording.get("mode") or "recording mode not available")
        return references, f"pipeline contract ({mode.replace('_', ' ')})"
    if isinstance(references, dict) and references.get("schema_version") == 1:
        inference = references.get("inference")
        if isinstance(inference, dict):
            recording = pipeline.get("recording")
            recording = recording if isinstance(recording, dict) else {}
            mode = str(recording.get("mode") or "recording mode not available")
            return {
                "schema_version": 2,
                "audio_preparation": references.get("audio_preparation", ""),
                "model_artifact": inference.get("artifact", ""),
                "input_processing": inference.get("frontend", ""),
                "execution_stack": inference.get("runtime", ""),
                "chunking": inference.get("chunking", ""),
                "evaluation": references.get("evaluation", ""),
                "observation_policy": references.get("runtime_resolution", ""),
            }, f"pipeline contract ({mode.replace('_', ' ')}; schema 1)"

    recovered = infer_contract_references(profile)
    if recovered is not None:
        return recovered, "inference-profile contract"
    return {}, "pipeline contract not recorded"


def _inference_reference(references: dict[str, Any], field: str) -> str:
    value = references.get(field)
    return value if isinstance(value, str) else ""


def model_artifact_label(
    profile: dict[str, Any],
    run_metadata: dict[str, Any],
    references: dict[str, Any],
) -> str:
    artifact = _inference_reference(references, "model_artifact")
    labels = {
        "canonical_checkpoint": "canonical model-owner artifact",
        "project_export": "project-exported ONNX artifact",
        "published_deployment_artifact": "published Android deployment artifact",
        "openrouter_remote_model_slug": "OpenRouter-hosted model slug",
        "fairseq2_asset_card_checkpoint": "Fairseq2 model-owner asset card",
    }
    if artifact in labels:
        return labels[artifact]

    backend = run_metadata.get("inference_adapter")
    backend = backend if isinstance(backend, dict) else {}
    if backend.get("artifact_target") == "packaged_android_reference":
        return "published Android deployment artifact"
    if profile.get("inference_library") == "onnxruntime":
        return "ONNX artifact (role not recorded)"
    if profile.get("artifact"):
        return "selected artifact (role not recorded)"
    return "artifact not recorded"


def input_processing_label(references: dict[str, Any]) -> str:
    audio = references.get("audio_preparation")
    audio = audio if isinstance(audio, str) else ""
    frontend = _inference_reference(references, "input_processing")
    chunking = _inference_reference(references, "chunking")

    audio_labels = {
        "shared_soundfile_librosa_16khz": "shared SoundFile/librosa 16 kHz audio",
        "android_pcm16_linear_16khz": "Android PCM16/linear 16 kHz audio",
        "openrouter_original_wav_upload": "original WAV uploaded as base64 JSON",
    }
    frontend_labels = {
        "canonical_transformers_auto_processor": "canonical Transformers frontend",
        "canonical_nemo_model_preprocessor": "canonical NeMo frontend",
        "compatible_onnx_asr_nemo_frontend": "compatible onnx-asr NeMo frontend",
        "deployment_parity_android_frontend": "Android deployment-parity frontend",
        "canonical_sherpa_online_frontend": "canonical Sherpa-ONNX frontend",
        "openrouter_provider_managed_audio_frontend": (
            "OpenRouter provider-managed audio frontend"
        ),
        "omnilingual_reference_inference_pipeline": (
            "Omnilingual reference audio frontend"
        ),
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


def inference_engine_version(run_metadata: dict[str, Any]) -> tuple[str, str]:
    execution_stack = run_metadata.get("execution_stack")
    execution_stack = execution_stack if isinstance(execution_stack, dict) else {}
    packages = execution_stack.get("packages")
    packages = packages if isinstance(packages, dict) else {}
    backend = run_metadata.get("inference_adapter")
    backend = backend if isinstance(backend, dict) else {}
    observed = (
        packages.get("onnxruntime")
        or packages.get("onnxruntime-gpu")
        or backend.get("onnxruntime_version")
    )
    required = backend.get("onnxruntime_version_required")
    return str(observed or ""), str(required or "")


def _openrouter_region(
    profile: dict[str, Any],
    run_metadata: dict[str, Any],
    references: dict[str, Any],
) -> str | None:
    backend = run_metadata.get("inference_adapter")
    backend = backend if isinstance(backend, dict) else {}
    enforcement = backend.get("region_enforcement")
    enforcement = enforcement if isinstance(enforcement, dict) else {}
    api = profile.get("api")
    api = api if isinstance(api, dict) else {}
    for region in (enforcement.get("gateway"), api.get("routing_region")):
        if region in ("eu", "global"):
            return region
    return {
        "openrouter_eu_api": "eu",
        "openrouter_global_zdr_api": "global",
    }.get(_inference_reference(references, "execution_stack"))


def execution_stack_context(
    profile: dict[str, Any],
    run_metadata: dict[str, Any],
    references: dict[str, Any],
) -> tuple[str, bool]:
    """Describe the software and environment used for one inference run."""
    execution_stack = run_metadata.get("execution_stack")
    execution_stack = execution_stack if isinstance(execution_stack, dict) else {}
    launch = execution_stack.get("launch_context")
    launch = launch if isinstance(launch, dict) else {}
    observed_version, required_version = inference_engine_version(run_metadata)
    inference_library = profile.get("inference_library")
    if inference_library not in {"onnxruntime", "sherpa_onnx"}:
        observed_version = ""
        required_version = ""

    adapter = profile.get("adapter")
    openrouter_region = _openrouter_region(profile, run_metadata, references)
    route_labels = {
        "transformers": "Transformers → PyTorch",
        "nemo": "NeMo → PyTorch",
        "sherpa_onnx": "Sherpa-ONNX → ONNX Runtime",
        "onnxruntime": (
            "Android-parity adapter → ONNX Runtime"
            if adapter == "android_ctc"
            else "onnx-asr → ONNX Runtime"
        ),
        "multimodal": "multimodal Transformers adapter → PyTorch",
        "openrouter": (
            "OpenRouter API → eligible EU/ZDR provider"
            if openrouter_region == "eu"
            else "OpenRouter API → global ZDR routing"
            if openrouter_region == "global"
            else "OpenRouter API → eligible ZDR provider"
        ),
        "omnilingual": "Omnilingual ASR → Fairseq2 → PyTorch",
    }
    route = route_labels.get(
        str(inference_library),
        str(inference_library or "inference library not recorded").replace("_", " "),
    )

    if launch.get("status") == "observed":
        service = launch.get("compose_service")
        service = service if isinstance(service, str) else None
        image = launch.get("image")
        image = image if isinstance(image, dict) else {}
        image_reference = image.get("reference")
        service_labels = {
            "nemo-asr": "shared ASR container",
            "transformers-asr": "shared ASR container",
            "sherpa-onnx-asr": "shared ASR container",
            "onnxruntime-asr": "shared ASR container on PC",
            "onnxruntime-android-asr": (
                "dedicated Android-parity container on PC"
            ),
            "multimodal-asr": "shared ASR container",
            "phi4-multimodal-asr": "dedicated Phi-4 container",
            "qwen-omni-asr": "dedicated Qwen Omni container",
            "openrouter-asr": "lightweight OpenRouter client container",
            "omnilingual-asr": "dedicated Omnilingual container",
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
        environment = f"{label} ({'; '.join(details)})" if details else label
        return f"{route} → {environment}", True

    execution_stack_reference = _inference_reference(references, "execution_stack")
    openrouter_environment = (
        "OpenRouter fail-closed EU API"
        if openrouter_region == "eu"
        else "OpenRouter fail-closed global ZDR API"
    )
    labels = {
        "shared_transformers_image": "shared ASR container",
        "multimodal_image": "model-family multimodal container",
        "nemo_image": "shared ASR container",
        "onnxruntime_shared_image": "shared ASR container on PC",
        "android_pinned_execution_environment_proxy": (
            "dedicated Android-parity container on PC"
        ),
        "sherpa_onnx_image": "shared ASR container",
        "openrouter_eu_api": openrouter_environment,
        "openrouter_global_zdr_api": openrouter_environment,
        "omnilingual_fairseq2_image": "dedicated Omnilingual container",
    }
    if execution_stack_reference in labels:
        label = labels[execution_stack_reference]
        if observed_version:
            label += f" (ONNX Runtime {observed_version} observed)"
        elif required_version:
            label += f" (ONNX Runtime {required_version} required)"
        return f"{route} → {label}", False

    if inference_library == "onnxruntime":
        backend = run_metadata.get("inference_adapter")
        backend = backend if isinstance(backend, dict) else {}
        quantization = backend.get("quantization")
        precision = "INT8" if quantization == "int8" else "FP32"
        return f"{route} ({precision}; environment not recorded)", False
    if adapter == "phi4_audio":
        return f"{route} (Phi-4 environment not recorded)", False
    if adapter == "qwen_omni_audio":
        return f"{route} (Qwen environment not recorded)", False
    if adapter == "gemma4_audio":
        return f"{route} (Gemma environment not recorded)", False

    return f"{route} → environment not recorded", False


def execution_target_label(
    profile: dict[str, Any],
    run_metadata: dict[str, Any],
    references: dict[str, Any],
) -> str:
    """Return a compact execution-target label for presentation names."""
    execution_stack = run_metadata.get("execution_stack")
    execution_stack = execution_stack if isinstance(execution_stack, dict) else {}
    launch = execution_stack.get("launch_context")
    launch = launch if isinstance(launch, dict) else {}
    service = launch.get("compose_service")
    service = service if isinstance(service, str) else None
    openrouter_region = _openrouter_region(profile, run_metadata, references)
    openrouter_target = (
        f"openrouter-{openrouter_region}-zdr"
        if openrouter_region is not None
        else "openrouter-zdr"
    )
    service_labels = {
        "nemo-asr": "nemo",
        "transformers-asr": "transformers",
        "sherpa-onnx-asr": "sherpa-onnx",
        "onnxruntime-asr": "onnxruntime-desktop",
        "onnxruntime-android-asr": "onnxruntime-android-proxy",
        "multimodal-asr": "multimodal-shared",
        "phi4-multimodal-asr": "multimodal-phi4",
        "qwen-omni-asr": "multimodal-qwen-omni",
        "openrouter-asr": openrouter_target,
        "omnilingual-asr": "omnilingual-fairseq2",
    }
    if service in service_labels:
        return service_labels[service]

    execution_stack_reference = _inference_reference(references, "execution_stack")
    execution_stack_labels = {
        "shared_transformers_image": "transformers",
        "nemo_image": "nemo",
        "onnxruntime_shared_image": "onnxruntime-desktop",
        "android_pinned_execution_environment_proxy": (
            "onnxruntime-android-proxy"
        ),
        "sherpa_onnx_image": "sherpa-onnx",
        "openrouter_eu_api": openrouter_target,
        "openrouter_global_zdr_api": openrouter_target,
        "omnilingual_fairseq2_image": "omnilingual-fairseq2",
    }
    if execution_stack_reference in execution_stack_labels:
        return execution_stack_labels[execution_stack_reference]

    adapter = profile.get("adapter")
    multimodal_labels = {
        "gemma4_audio": "multimodal-gemma",
        "phi4_audio": "multimodal-phi4",
        "qwen_omni_audio": "multimodal-qwen-omni",
    }
    if adapter in multimodal_labels:
        return multimodal_labels[adapter]

    inference_library = profile.get("inference_library")
    if inference_library == "openrouter":
        return openrouter_target
    if isinstance(inference_library, str) and inference_library:
        return inference_library.strip().lower().replace("_", "-")
    return "execution-target-not-recorded"


def decoder_slug(label: str) -> str:
    """Normalize a readable decoder description for stable display labels."""
    normalized = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return normalized or "decoder-not-recorded"
