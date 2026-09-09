"""Prepare and verify ONNX artifacts derived from the internal Exp41 model."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from inference.provenance import sha256_file

ARTIFACT_SCHEMA_VERSION = 1
FP32_MODEL_NAME = "model.onnx"
INT8_MODEL_NAME = "model.int8.onnx"
VOCAB_NAME = "vocab.txt"
CONFIG_NAME = "config.json"
METADATA_NAME = "artifact_metadata.json"

EXP41_PROFILE_ID = "swahili-exp41-ctc"
EXP41_SOURCE_NAME = "model_exp41_avg.nemo"
EXP41_SOURCE_SHA256 = "6450926bc1338827ab201b2d9f8f94bcb7a5bd06b6f72690f60ead14a067a7b0"
ANDROID_SOURCE_URL = "https://github.com/AI-for-Education/nemoasr-android"
ANDROID_SOURCE_REVISION = "6cd031267aed45e47347bc05a0eb63eee285f0d8"
ONNX_ASR_SOURCE_URL = "https://github.com/istupakov/onnx-asr"
ONNX_ASR_SOURCE_REVISION = "b9e0ce0ae3223b3d24ce5a22a5e701a726ca35fc"


class ArtifactError(RuntimeError):
    """Raised when an exported artifact is incomplete or has wrong lineage."""


def _software_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _publish_permissions(bundle_dir: Path) -> None:
    """Make a completed bind-mounted bundle readable by non-root runners."""
    bundle_dir.chmod(0o755)
    for path in bundle_dir.iterdir():
        if path.is_file():
            path.chmod(0o644)


def _load_metadata(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / METADATA_NAME
    if not path.is_file():
        raise ArtifactError(f"Artifact metadata is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Artifact metadata is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"Artifact metadata must contain an object: {path}")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError(
            f"Unsupported artifact metadata schema in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    return payload


def _validate_vocab(path: Path) -> int:
    if not path.is_file():
        raise ArtifactError(f"Vocabulary is missing: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ArtifactError(f"Vocabulary is empty: {path}")
    parsed: list[tuple[str, int]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            token, raw_index = line.rsplit(" ", 1)
            index = int(raw_index)
        except (ValueError, TypeError) as exc:
            raise ArtifactError(
                f"Invalid vocabulary row {line_number} in {path}: {line!r}"
            ) from exc
        parsed.append((token, index))
    indices = [index for _token, index in parsed]
    if indices != list(range(len(parsed))):
        raise ArtifactError(f"Vocabulary IDs are not contiguous from zero: {path}")
    blank_rows = [(token, index) for token, index in parsed if token == "<blk>"]
    if blank_rows != [("<blk>", len(parsed) - 1)]:
        raise ArtifactError(
            f"Vocabulary must contain exactly one final <blk> CTC token: {path}"
        )
    return len(parsed)


def _validate_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactError(f"ONNX ASR config is missing: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"ONNX ASR config is unreadable: {path}: {exc}") from exc
    expected = {
        "model_type": "nemo-conformer-ctc",
        "features_size": 80,
        "subsampling_factor": 8,
    }
    if config != expected:
        raise ArtifactError(
            f"ONNX ASR config does not match Exp41: expected {expected}, got {config}"
        )
    return config


def verify_artifact(
    model_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Verify one selected ONNX file and all shared bundle dependencies."""
    selected = Path(model_path)
    if selected.name not in {FP32_MODEL_NAME, INT8_MODEL_NAME}:
        raise ArtifactError(
            f"Expected {FP32_MODEL_NAME} or {INT8_MODEL_NAME}, got: {selected}"
        )
    if not selected.is_file():
        raise ArtifactError(f"Selected ONNX model is missing: {selected}")

    bundle_dir = selected.parent
    vocab_size = _validate_vocab(bundle_dir / VOCAB_NAME)
    config = _validate_config(bundle_dir / CONFIG_NAME)
    metadata = _load_metadata(bundle_dir)
    source = metadata.get("source_checkpoint")
    if not isinstance(source, dict) or source.get("sha256") != EXP41_SOURCE_SHA256:
        raise ArtifactError(
            "ONNX bundle does not record the expected Exp41 source checkpoint SHA-256"
        )
    if metadata.get("same_trained_checkpoint") is not True:
        raise ArtifactError(
            "ONNX bundle must declare that it is derived from the same trained checkpoint"
        )
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(
        artifacts.get(selected.name), dict
    ):
        raise ArtifactError(
            f"Metadata has no integrity record for selected artifact: {selected.name}"
        )
    selected_record = artifacts[selected.name]
    if selected_record.get("size_bytes") != selected.stat().st_size:
        raise ArtifactError(f"Artifact size does not match metadata: {selected}")
    if verify_hashes and selected_record.get("sha256") != sha256_file(selected):
        raise ArtifactError(f"Artifact SHA-256 does not match metadata: {selected}")

    return {
        "metadata": metadata,
        "config": config,
        "vocab_size": vocab_size,
        "selected_record": selected_record,
    }


def _model_cfg_value(model: Any, section: str, key: str) -> Any:
    cfg = getattr(model, "cfg", None)
    block = getattr(cfg, section, None)
    if block is None and isinstance(cfg, dict):
        block = cfg.get(section)
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _write_vocab(model: Any, path: Path) -> int:
    tokenizer = getattr(model, "tokenizer", None)
    raw_vocab = getattr(tokenizer, "vocab", None)
    if raw_vocab is None:
        raise ArtifactError("The restored NeMo model has no tokenizer.vocab")
    tokens = [str(token) for token in raw_vocab]
    if not tokens:
        raise ArtifactError("The restored NeMo tokenizer vocabulary is empty")
    if "<blk>" in tokens:
        raise ArtifactError("The tokenizer vocabulary already contains <blk>")
    tokens.append("<blk>")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for index, token in enumerate(tokens):
            if not token or any(char in token for char in (" ", "\r", "\n")):
                raise ArtifactError(
                    "A tokenizer token cannot be represented in onnx-asr vocab.txt: "
                    f"{token!r}"
                )
            stream.write(f"{token} {index}\n")
    return len(tokens)


def export_nemo_to_fp32_onnx(
    source_nemo: str | Path,
    output_dir: str | Path,
    *,
    expected_source_sha256: str = EXP41_SOURCE_SHA256,
    model_loader: Callable[[Path], Any] | None = None,
) -> Path:
    """Export Exp41 atomically; an existing destination is never overwritten."""
    source = Path(source_nemo).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_file():
        raise ArtifactError(f"Source NeMo checkpoint is missing: {source}")
    if destination.exists():
        raise ArtifactError(
            f"Refusing to overwrite existing ONNX bundle directory: {destination}"
        )

    source_sha256 = sha256_file(source)
    if source_sha256 != expected_source_sha256.lower():
        raise ArtifactError(
            "Source checkpoint SHA-256 mismatch; refusing to export a different model: "
            f"expected {expected_source_sha256.lower()}, got {source_sha256}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    model = None
    try:
        if model_loader is None:
            try:
                from nemo.collections.asr.models import ASRModel
            except ModuleNotFoundError as exc:
                raise ArtifactError(
                    "NeMo export requires nemo_toolkit[asr]; run the preparation "
                    "command through the shared ASR image"
                ) from exc

            model_loader = lambda path: ASRModel.restore_from(
                str(path), map_location="cpu"
            )

        model = model_loader(source)
        if hasattr(model, "eval"):
            model.eval()

        features = _model_cfg_value(model, "preprocessor", "features")
        sample_rate = _model_cfg_value(model, "preprocessor", "sample_rate")
        subsampling_factor = _model_cfg_value(model, "encoder", "subsampling_factor")
        observed = {
            "features_size": int(features) if features is not None else None,
            "sample_rate": int(sample_rate) if sample_rate is not None else None,
            "subsampling_factor": (
                int(subsampling_factor) if subsampling_factor is not None else None
            ),
        }
        expected = {
            "features_size": 80,
            "sample_rate": 16000,
            "subsampling_factor": 8,
        }
        if observed != expected:
            raise ArtifactError(
                f"Restored model configuration is not Exp41-compatible: {observed}"
            )

        fp32_path = temp_dir / FP32_MODEL_NAME
        model.export(str(fp32_path))
        if not fp32_path.is_file() or fp32_path.stat().st_size == 0:
            raise ArtifactError(f"NeMo did not create a valid ONNX file: {fp32_path}")

        vocab_size = _write_vocab(model, temp_dir / VOCAB_NAME)
        config = {
            "model_type": "nemo-conformer-ctc",
            "features_size": 80,
            "subsampling_factor": 8,
        }
        _atomic_write_json(temp_dir / CONFIG_NAME, config)

        metadata = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "same_trained_checkpoint": True,
            "source_checkpoint": {
                "profile_id": EXP41_PROFILE_ID,
                "name": source.name,
                "sha256": source_sha256,
                "size_bytes": source.stat().st_size,
                "remote_revision_verified": False,
            },
            "model_contract": {
                "language": "sw",
                "task": "transcribe",
                "output_units": "orthographic",
                "sample_rate": 16000,
                "features_size": 80,
                "subsampling_factor": 8,
                "vocab_size_including_ctc_blank": vocab_size,
                "decoding": "greedy_ctc",
            },
            "artifacts": {
                FP32_MODEL_NAME: {
                    **_file_record(fp32_path),
                    "precision": "fp32",
                    "deployment_role": "portable_reference",
                },
                VOCAB_NAME: _file_record(temp_dir / VOCAB_NAME),
                CONFIG_NAME: _file_record(temp_dir / CONFIG_NAME),
            },
            "provenance": {
                "android_reference": {
                    "url": ANDROID_SOURCE_URL,
                    "revision": ANDROID_SOURCE_REVISION,
                    "license_file_present_at_revision": False,
                },
                "desktop_inference_library_reference": {
                    "url": ONNX_ASR_SOURCE_URL,
                    "revision": ONNX_ASR_SOURCE_REVISION,
                    "package": "onnx-asr==0.12.0",
                },
            },
            "software": {
                "nemo_toolkit": _software_version("nemo_toolkit"),
                "onnx": _software_version("onnx"),
                "torch": _software_version("torch"),
            },
        }
        _atomic_write_json(temp_dir / METADATA_NAME, metadata)
        _publish_permissions(temp_dir)
        verify_artifact(fp32_path)
        temp_dir.replace(destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        if model is not None:
            del model

    return destination / FP32_MODEL_NAME


def quantize_fp32_to_int8(
    bundle_dir: str | Path,
    *,
    quantizer: Callable[..., Any] | None = None,
) -> Path:
    """Create the mobile-target INT8 model without overwriting any model file."""
    directory = Path(bundle_dir).resolve()
    fp32_path = directory / FP32_MODEL_NAME
    verify_artifact(fp32_path)
    int8_path = directory / INT8_MODEL_NAME
    if int8_path.exists():
        raise ArtifactError(f"Refusing to overwrite existing INT8 model: {int8_path}")

    if quantizer is None:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ModuleNotFoundError as exc:
            raise ArtifactError(
                "INT8 preparation requires ONNX Runtime quantization support"
            ) from exc

        def quantizer(model_input: str, model_output: str) -> None:
            quantize_dynamic(
                model_input=model_input,
                model_output=model_output,
                per_channel=True,
                weight_type=QuantType.QUInt8,
            )

    temp_path = directory / f".{INT8_MODEL_NAME}.{uuid4().hex}.tmp.onnx"
    try:
        quantizer(str(fp32_path), str(temp_path))
        if not temp_path.is_file() or temp_path.stat().st_size == 0:
            raise ArtifactError("Quantization did not create a valid INT8 ONNX file")
        if int8_path.exists():
            raise ArtifactError(
                f"INT8 model appeared during quantization; refusing overwrite: {int8_path}"
            )
        temp_path.replace(int8_path)

        metadata = _load_metadata(directory)
        artifacts = metadata["artifacts"]
        artifacts[INT8_MODEL_NAME] = {
            **_file_record(int8_path),
            "precision": "int8_dynamic_weights",
            "deployment_role": "mobile_target_candidate",
            "derived_from": FP32_MODEL_NAME,
            "quantization": {
                "method": "onnxruntime.quantize_dynamic",
                "per_channel": True,
                "weight_type": "QUInt8",
            },
        }
        metadata["quantized_at"] = datetime.now(timezone.utc).isoformat()
        metadata.setdefault("software", {})["onnxruntime"] = _software_version(
            "onnxruntime-gpu"
        ) or _software_version("onnxruntime")
        _atomic_write_json(directory / METADATA_NAME, metadata)
        _publish_permissions(directory)
        verify_artifact(int8_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        if int8_path.exists():
            # Only remove a file created by this invocation before metadata commit.
            directory / METADATA_NAME
            try:
                current = _load_metadata(directory)
            except ArtifactError:
                current = {}
            if INT8_MODEL_NAME not in current.get("artifacts", {}):
                int8_path.unlink(missing_ok=True)
        raise

    return int8_path
