"""Paza Phi-4 audio transcription with bounded CPU and disk offloading."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from inference.common import load_audio_and_resample
from inference.contracts import TranscriptionResult
from inference.multimodal.adapters.common import (
    cuda_memory_gib,
    move_inputs,
    serializable_device_map,
    split_audio,
    torch_dtype_from_profile,
)
from inference.profile import ModelProfile, ProfileError
from inference.provenance import generation_provenance


def _local_auto_map(value: Any) -> Any:
    """Remove a Hub repository prefix while retaining the bundled module path."""
    if isinstance(value, str) and "--" in value:
        return value.split("--", 1)[1]
    return value


def _prepare_processor_snapshot(model_path: Path, destination: Path) -> Path:
    """Create a lightweight local-only processor view without changing weights."""
    processor_path = destination / "processor"
    processor_path.mkdir(parents=True, exist_ok=False)
    for source in model_path.iterdir():
        if not source.is_file():
            continue
        if (
            source.suffix == ".safetensors"
            or source.name == "model.safetensors.index.json"
        ):
            continue
        if source.name in {".gitattributes", "README.md"}:
            continue
        shutil.copy2(source, processor_path / source.name)

    for name in ("config.json", "preprocessor_config.json"):
        path = processor_path / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        auto_map = payload.get("auto_map")
        if isinstance(auto_map, dict):
            payload["auto_map"] = {
                key: _local_auto_map(value) for key, value in auto_map.items()
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return processor_path


class Phi4AudioBackend:
    """Text-only Phi-4 ASR using the local custom code and Accelerate offload."""

    def __init__(
        self,
        profile: ModelProfile,
        model_path: str | Path,
        *,
        device: torch.device | None = None,
        offload_root: str | Path | None = None,
    ) -> None:
        if profile.framework != "multimodal" or profile.adapter != "phi4_audio":
            raise ProfileError(
                "Phi4AudioBackend requires a multimodal/phi4_audio profile"
            )
        if profile.prompt is None or profile.audio is None or profile.hardware is None:
            raise ProfileError(
                "The Phi-4 profile requires prompt, audio, and hardware settings"
            )
        if profile.hardware.memory_strategy != "cpu_disk_offload":
            raise ProfileError("Phi-4 requires the cpu_disk_offload memory strategy")

        self.profile = profile
        self.model_path = Path(model_path)
        self.model = None
        self.processor = None
        self._runtime_dir: Path | None = None
        if not self.model_path.is_dir():
            raise ProfileError(
                f"Multimodal model directory not found: {self.model_path}"
            )

        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        if self.device.type != "cuda":
            raise RuntimeError("Paza Phi-4 offload inference still requires a CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Paza Phi-4 requires CUDA bfloat16 support")
        total_gpu_gib = cuda_memory_gib(self.device)
        if total_gpu_gib < profile.hardware.minimum_gpu_memory_gib:
            raise RuntimeError(
                "Paza Phi-4 hardware preflight failed: "
                f"{total_gpu_gib:.1f} GiB GPU memory detected, "
                f"{profile.hardware.minimum_gpu_memory_gib:.1f} GiB required"
            )

        root = Path(
            offload_root
            or os.environ.get("ASR_OFFLOAD_ROOT", "")
            or tempfile.gettempdir()
        )
        root.mkdir(parents=True, exist_ok=True)
        self._runtime_dir = Path(tempfile.mkdtemp(prefix="phi4_", dir=root))
        processor_path = _prepare_processor_snapshot(
            self.model_path,
            self._runtime_dir,
        )
        weight_offload_path = self._runtime_dir / "weights"
        weight_offload_path.mkdir()

        max_memory = {
            0: f"{profile.hardware.gpu_max_memory_gib:g}GiB",
            "cpu": f"{profile.hardware.cpu_max_memory_gib:g}GiB",
        }
        common_kwargs = {
            "local_files_only": profile.loader.local_files_only,
            "trust_remote_code": profile.loader.trust_remote_code,
        }
        model_dtype = torch_dtype_from_profile(profile.loader.torch_dtype)
        attention_implementation = profile.loader.attention_implementation or "sdpa"
        try:
            self.processor = AutoProcessor.from_pretrained(
                str(processor_path),
                **common_kwargs,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **common_kwargs,
                torch_dtype=model_dtype,
                device_map="auto",
                max_memory=max_memory,
                offload_folder=str(weight_offload_path),
                offload_state_dict=True,
                low_cpu_mem_usage=True,
                _attn_implementation=attention_implementation,
            )
            self.model.eval()
            self.generation_config = GenerationConfig.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
        except Exception as exc:
            self.close()
            raise RuntimeError(
                f"Failed to load local Paza Phi-4 model from {self.model_path}: {exc}"
            ) from exc

        audio_processor = getattr(self.processor, "audio_processor", None)
        self.sampling_rate = int(getattr(audio_processor, "sampling_rate", 16000))
        if self.sampling_rate <= 0:
            self.close()
            raise RuntimeError(
                f"Phi-4 processor reported an invalid sampling rate: {self.sampling_rate}"
            )

        self.input_device = self.device
        self._rendered_prompt = (
            f"<|user|><|audio_1|>{profile.prompt}<|end|><|assistant|>"
        )
        self._generation_kwargs = dict(profile.decoding.generation_kwargs)
        self._generation_provenance = generation_provenance(
            self.generation_config,
            requested_kwargs=profile.decoding.generation_kwargs,
            call_kwargs=self._generation_kwargs,
        )
        self._metadata: dict[str, Any] = {
            "framework": "multimodal",
            "adapter": "phi4_audio",
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "device": str(self.device),
            "device_map": serializable_device_map(self.model),
            "sampling_rate": self.sampling_rate,
            "requested_torch_dtype": profile.loader.torch_dtype,
            "effective_torch_dtype": str(model_dtype).removeprefix("torch."),
            "attention_implementation": attention_implementation,
            "trusted_code": {
                "enabled": True,
                "source": "bundled_local_snapshot",
                "network_access": False,
            },
            "hardware": {
                "detected_gpu_memory_gib": round(total_gpu_gib, 2),
                "memory_strategy": profile.hardware.memory_strategy,
                "gpu_max_memory_gib": profile.hardware.gpu_max_memory_gib,
                "cpu_max_memory_gib": profile.hardware.cpu_max_memory_gib,
                "output_mode": profile.hardware.output_mode,
            },
            "generation_kwargs": dict(self._generation_kwargs),
            "generation": self._generation_provenance,
            "rendered_prompt": self._rendered_prompt,
            "prompt": profile.prompt,
            "long_audio": {
                "strategy": profile.audio.long_audio_strategy,
                "maximum_seconds": profile.audio.maximum_seconds,
                "chunk_seconds": profile.audio.chunk_seconds,
                "overlap_seconds": profile.audio.overlap_seconds,
            },
            "files_seen": 0,
            "long_audio_files": 0,
            "chunks_generated": 0,
        }

    def _decode_chunk(self, audio: np.ndarray) -> str:
        encoded = self.processor(
            text=self._rendered_prompt,
            audios=[(np.asarray(audio, dtype=np.float32), self.sampling_rate)],
            return_tensors="pt",
        )
        model_inputs = move_inputs(encoded, device=self.input_device)
        input_length = int(model_inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config,
                **self._generation_kwargs,
            )
        response_ids = generated_ids[:, input_length:]
        decoded = self.processor.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return str(decoded[0]).strip()

    def transcribe_batch(self, audio_paths: Sequence[str]) -> list[TranscriptionResult]:
        rows: list[TranscriptionResult] = []
        for raw_path in audio_paths:
            audio_path = str(raw_path)
            self._metadata["files_seen"] += 1
            try:
                audio, _, duration, _ = load_audio_and_resample(
                    audio_path,
                    self.sampling_rate,
                )
            except Exception as exc:
                rows.append(
                    TranscriptionResult(
                        audio_filepath=audio_path,
                        duration=0.0,
                        pred_text="",
                        error=f"failed_to_read_audio: {exc}",
                    )
                )
                continue

            chunks = split_audio(
                audio,
                sampling_rate=self.sampling_rate,
                profile=self.profile,
            )
            self._metadata["chunks_generated"] += len(chunks)
            self._metadata["long_audio_files"] += int(len(chunks) > 1)
            try:
                predictions = [self._decode_chunk(chunk) for chunk in chunks]
                prediction = " ".join(text for text in predictions if text).strip()
            except Exception as exc:
                rows.append(
                    TranscriptionResult(
                        audio_filepath=audio_path,
                        duration=duration,
                        pred_text="",
                        error=f"inference_failed: {exc}",
                    )
                )
                continue
            rows.append(
                TranscriptionResult(
                    audio_filepath=audio_path,
                    duration=duration,
                    pred_text=prediction,
                )
            )
        return rows

    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def close(self) -> None:
        self.model = None
        self.processor = None
        if getattr(self, "device", None) is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
        runtime_dir = getattr(self, "_runtime_dir", None)
        if runtime_dir is not None:
            shutil.rmtree(runtime_dir, ignore_errors=True)
            self._runtime_dir = None
