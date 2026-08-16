from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference.onnxruntime import artifacts
from inference.onnxruntime.artifacts import (
    FP32_MODEL_NAME,
    INT8_MODEL_NAME,
    ArtifactError,
    export_nemo_to_fp32_onnx,
    quantize_fp32_to_int8,
    verify_artifact,
)
from inference.onnxruntime.prepare import prepare


class _FakeNemoModel:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(
            preprocessor=SimpleNamespace(features=80, sample_rate=16000),
            encoder=SimpleNamespace(subsampling_factor=8),
        )
        self.tokenizer = SimpleNamespace(vocab=["<unk>", "▁", "s", "wa"])
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def export(self, path: str) -> None:
        Path(path).write_bytes(b"fake-fp32-onnx")


def _export_fake_bundle(monkeypatch, tmp_path: Path) -> Path:
    source = tmp_path / "model_exp41_avg.nemo"
    source.write_bytes(b"fake-exp41")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(artifacts, "EXP41_SOURCE_SHA256", source_sha)
    destination = tmp_path / "swahili-exp41-ctc"
    export_nemo_to_fp32_onnx(
        source,
        destination,
        expected_source_sha256=source_sha,
        model_loader=lambda _path: _FakeNemoModel(),
    )
    return destination


def test_export_is_atomic_verified_and_never_overwrites(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = _export_fake_bundle(monkeypatch, tmp_path)

    verified = verify_artifact(destination / FP32_MODEL_NAME)
    assert verified["config"]["subsampling_factor"] == 8
    assert verified["vocab_size"] == 5
    assert verified["metadata"]["same_trained_checkpoint"] is True
    assert not list(tmp_path.glob(".swahili-exp41-ctc.tmp-*"))

    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        export_nemo_to_fp32_onnx(
            tmp_path / "model_exp41_avg.nemo",
            destination,
            expected_source_sha256=artifacts.EXP41_SOURCE_SHA256,
            model_loader=lambda _path: _FakeNemoModel(),
        )


def test_export_rejects_wrong_source_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "wrong.nemo"
    source.write_bytes(b"wrong checkpoint")

    with pytest.raises(ArtifactError, match="SHA-256 mismatch"):
        export_nemo_to_fp32_onnx(
            source,
            tmp_path / "bundle",
            expected_source_sha256="0" * 64,
            model_loader=lambda _path: _FakeNemoModel(),
        )
    assert not (tmp_path / "bundle").exists()


def test_quantization_is_resume_safe_and_does_not_overwrite(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = _export_fake_bundle(monkeypatch, tmp_path)

    def fake_quantizer(model_input: str, model_output: str) -> None:
        assert Path(model_input).name == FP32_MODEL_NAME
        Path(model_output).write_bytes(b"fake-int8-onnx")

    int8_path = quantize_fp32_to_int8(destination, quantizer=fake_quantizer)
    verified = verify_artifact(int8_path)
    record = verified["selected_record"]
    assert record["precision"] == "int8_dynamic_weights"
    assert record["derived_from"] == FP32_MODEL_NAME

    prepared = prepare("unused.nemo", destination)
    assert prepared == {
        "fp32": destination / FP32_MODEL_NAME,
        "int8": destination / INT8_MODEL_NAME,
    }

    with pytest.raises(ArtifactError, match="Refusing to overwrite"):
        quantize_fp32_to_int8(destination, quantizer=fake_quantizer)


def test_integrity_check_rejects_modified_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    destination = _export_fake_bundle(monkeypatch, tmp_path)
    (destination / FP32_MODEL_NAME).write_bytes(b"changed")

    with pytest.raises(ArtifactError, match="size does not match|SHA-256"):
        verify_artifact(destination / FP32_MODEL_NAME)
