from __future__ import annotations

import hashlib
from pathlib import Path

import inference.provenance as provenance_module
from inference.provenance import (
    canonical_json_sha256,
    generation_provenance,
    git_identity,
    model_artifact_identity,
)


class FakeGenerationConfig:
    def to_dict(self) -> dict[str, object]:
        return {
            "do_sample": True,
            "num_beams": 1,
            "temperature": 0.7,
        }


def test_generation_provenance_distinguishes_defaults_requested_and_actual() -> None:
    provenance = generation_provenance(
        FakeGenerationConfig(),
        requested_kwargs={"do_sample": False, "num_beams": 5},
        call_kwargs={"do_sample": False, "num_beams": 5, "task": "transcribe"},
        conditional_overrides={"long_form": {"return_timestamps": True}},
    )

    assert provenance["model_generation_config"]["do_sample"] is True
    assert provenance["requested_profile_kwargs"] == {
        "do_sample": False,
        "num_beams": 5,
    }
    assert provenance["actual_call_kwargs"]["task"] == "transcribe"
    assert provenance["effective_generation_config"]["do_sample"] is False
    assert provenance["effective_generation_config"]["temperature"] == 0.7
    assert provenance["conditional_call_overrides"] == {
        "long_form": {"return_timestamps": True}
    }


def test_hub_model_artifact_identity_uses_pinned_source_manifests_and_etags(
    tmp_path: Path,
) -> None:
    artifact = "whisper-large"
    model_path = tmp_path / artifact
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"model_type":"whisper"}', encoding="utf-8"
    )
    (model_path / "model.safetensors").write_bytes(b"weights")

    metadata_path = (
        model_path
        / ".cache"
        / "huggingface"
        / "download"
        / "model.safetensors.metadata"
    )
    metadata_path.parent.mkdir(parents=True)
    revision = "4ef9b41f0d4fe232daafdb5f76bb1dd8b23e01d7"
    metadata_path.write_text(f"{revision}\nweight-etag\n", encoding="utf-8")
    (tmp_path / f".{artifact}.download-complete").touch()
    (tmp_path / f".{artifact}.download-source.tsv").write_text(
        f"repository_id\topenai/whisper-large\nrevision\t{revision}\nscope\tfull\n",
        encoding="utf-8",
    )
    (tmp_path / f".{artifact}.download-sizes.tsv").write_text(
        "7\tmodel.safetensors\n24\tconfig.json\n",
        encoding="utf-8",
    )

    identity = model_artifact_identity(model_path, artifact=artifact)

    assert identity["intended_source"] == {
        "repository": "openai/whisper-large",
        "revision": revision,
    }
    assert identity["download_manifests"]["completion_marker"] is True
    assert identity["download_manifests"]["source"]["values"]["revision"] == revision
    assert identity["download_manifests"]["sizes"]["file_count"] == 2
    hub = identity["local_hugging_face_metadata"]
    assert hub["consistent_revision"] == revision
    assert hub["files"]["model.safetensors"]["etag"] == "weight-etag"
    assert hub["files"]["model.safetensors"]["size_bytes"] == 7
    assert (
        identity["configuration_files"]["config.json"]["sha256"]
        == hashlib.sha256(b'{"model_type":"whisper"}').hexdigest()
    )
    assert "artifact_files" not in identity


def test_pinned_partial_hub_snapshot_hashes_unrepresented_executable_files(
    tmp_path: Path,
) -> None:
    artifact = "sherpa-onnx-zipformer-streaming-robust-sw-v4"
    revision = "0e52da6c03294fd983f3a8621b32ac6a71b4787d"
    model_path = tmp_path / artifact
    model_path.mkdir()
    encoder = model_path / "encoder-epoch-40.int8.onnx"
    encoder.write_bytes(b"bookbot-encoder")
    unselected_encoder = model_path / "encoder-epoch-40.onnx"
    unselected_encoder.write_bytes(b"unused-fp32-encoder")
    tokens = model_path / "tokens.txt"
    tokens.write_text("<eps> 0\na 1\n", encoding="utf-8")
    metadata_path = (
        model_path / ".cache" / "huggingface" / "download" / "tokens.txt.metadata"
    )
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(f"{revision}\ntokens-etag\n", encoding="utf-8")

    identity = model_artifact_identity(
        model_path,
        artifact=artifact,
        selected_files=[encoder.name, tokens.name],
    )

    assert identity["local_hugging_face_metadata"]["files"] == {
        "tokens.txt": {
            "revision": revision,
            "etag": "tokens-etag",
            "size_bytes": tokens.stat().st_size,
        }
    }
    assert identity["artifact_files"][encoder.name] == {
        "size_bytes": encoder.stat().st_size,
        "sha256": hashlib.sha256(b"bookbot-encoder").hexdigest(),
    }
    assert unselected_encoder.name not in identity["artifact_files"]
    assert identity["selected_files"] == [encoder.name, tokens.name]
    assert identity["configuration_files"]["tokens.txt"]["sha256"] == (
        hashlib.sha256(tokens.read_bytes()).hexdigest()
    )


def test_local_model_artifact_identity_hashes_exported_files(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "zipformer-export"
    model_path.mkdir()
    (model_path / "encoder.int8.onnx").write_bytes(b"onnx-data")
    (model_path / "tokens.txt").write_text("<blk> 0\na 1\n", encoding="utf-8")

    identity = model_artifact_identity(
        model_path,
        artifact="zipformer-export",
    )

    expected = hashlib.sha256(b"onnx-data").hexdigest()
    assert identity["artifact_files"]["encoder.int8.onnx"]["sha256"] == expected
    assert identity["artifact_manifest_sha256"] == canonical_json_sha256(
        identity["artifact_files"]
    )


def test_git_identity_falls_back_to_head_files_and_source_hash(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    git_directory = root / ".git"
    reference = git_directory / "refs" / "heads" / "main"
    reference.parent.mkdir(parents=True)
    revision = "0123456789abcdef0123456789abcdef01234567"
    (git_directory / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    reference.write_text(f"{revision}\n", encoding="utf-8")
    inference_root = root / "inference"
    inference_root.mkdir()
    (inference_root / "adapter.py").write_text("VALUE = 1\n", encoding="utf-8")

    def git_is_unavailable(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance_module.subprocess, "run", git_is_unavailable)
    identity = git_identity(inference_root)

    assert identity["available"] is True
    assert identity["commit"] == revision
