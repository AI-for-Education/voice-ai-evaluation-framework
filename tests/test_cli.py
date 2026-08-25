from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from inference.nemo import infer as nemo_infer
from inference.onnxruntime import infer as onnxruntime_infer
from inference.multimodal import infer as multimodal_infer
from inference.profile import ProfileError
from inference.sherpa_onnx import infer as sherpa_infer
from inference.transformers import infer as transformers_infer


def _load_eval_manifest(path: Path, **overrides: object) -> pd.DataFrame:
    from eval_pipeline2 import load_eval_manifest

    return load_eval_manifest(
        str(path),
        audio_key="audio_filepath",
        ref_key="ref_text",
        can_key="can_text",
        hyp_key="pred_text",
        logger=logging.getLogger("test"),
        **overrides,
    )


@pytest.mark.parametrize(
    "parse_args",
    [
        multimodal_infer.parse_args,
        nemo_infer.parse_profile_args,
        onnxruntime_infer.parse_args,
        sherpa_infer.parse_args,
        transformers_infer.parse_args,
    ],
)
def test_framework_launchers_require_model_config(parse_args) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_args(
            [
                "--root_audio_dir",
                "audio",
                "--output_root",
                "output",
            ]
        )
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "parse_args",
    [
        multimodal_infer.parse_args,
        nemo_infer.parse_profile_args,
        onnxruntime_infer.parse_args,
        sherpa_infer.parse_args,
        transformers_infer.parse_args,
    ],
)
def test_framework_launchers_default_to_standard_output_base(parse_args) -> None:
    args = parse_args(
        [
            "--model_config",
            "profile.yaml",
            "--root_audio_dir",
            "audio",
            "--smoke_test",
        ]
    )

    assert args.output_root == "input_output_data/output"
    assert args.smoke_test is True


@pytest.mark.parametrize(
    "module",
    [
        multimodal_infer,
        nemo_infer,
        onnxruntime_infer,
        sherpa_infer,
        transformers_infer,
    ],
)
def test_invalid_batch_size_fails_before_profile_or_model_loading(
    monkeypatch,
    module,
) -> None:
    monkeypatch.setattr(
        module,
        "load_profile",
        lambda *args, **kwargs: pytest.fail("profile/model loading must not start"),
    )
    with pytest.raises(SystemExit, match="--batch_size must be at least 1"):
        module.main(
            [
                "--model_config",
                "profile.yaml",
                "--root_audio_dir",
                "audio",
                "--output_root",
                "output",
                "--batch_size",
                "0",
            ]
        )


def test_transformers_launcher_rejects_wrong_framework_before_backend_creation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "nemo.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "id": "wrong-framework",
                "framework": "nemo",
                "adapter": "nemo",
                "artifact": "model.nemo",
                "decoding": {"strategy": "ctc"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        transformers_infer,
        "create_backend",
        lambda *args, **kwargs: pytest.fail("backend creation must not start"),
    )

    with pytest.raises(SystemExit, match="expected 'transformers'"):
        transformers_infer.main(
            [
                "--model_config",
                str(profile_path),
                "--root_audio_dir",
                str(tmp_path),
                "--output_root",
                str(tmp_path / "output"),
            ]
        )


def test_nemo_launcher_rejects_missing_artifact_before_backend_creation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "nemo.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "id": "missing-model",
                "framework": "nemo",
                "adapter": "nemo",
                "artifact": "missing.nemo",
                "decoding": {"strategy": "ctc"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        nemo_infer,
        "NemoBackend",
        lambda *args, **kwargs: pytest.fail("backend creation must not start"),
    )

    with pytest.raises(SystemExit, match="Model artifact not found"):
        nemo_infer.main(
            [
                "--model_config",
                str(profile_path),
                "--root_audio_dir",
                str(tmp_path),
                "--output_root",
                str(tmp_path / "output"),
            ]
        )


def test_evaluation_output_defaults_follow_transcript_run() -> None:
    from eval_pipeline2 import _default_evaluation_root as evaluation_root
    from manifest_pipeline import _default_evaluation_root as manifest_root

    run_name = "bookbot-orthographic-ctc_2026_08_12_13_14_15"
    normal_transcript = (
        Path("input_output_data")
        / "output"
        / "transcripts"
        / run_name
        / "transcriptions.jsonl"
    )
    smoke_transcript = (
        Path("input_output_data")
        / "output"
        / "smoke_tests"
        / "transcripts"
        / run_name
        / "transcriptions.jsonl"
    )

    normal_evaluation = (
        Path("input_output_data") / "output" / "evaluations" / run_name
    )
    smoke_evaluation = (
        Path("input_output_data")
        / "output"
        / "smoke_tests"
        / "evaluations"
        / run_name
    )

    assert manifest_root([str(normal_transcript)]) == normal_evaluation
    assert manifest_root([str(smoke_transcript)]) == smoke_evaluation
    assert evaluation_root(
        str(normal_evaluation / "manifests" / "ref_manifest.clean.jsonl")
    ) == normal_evaluation
    assert evaluation_root(
        str(smoke_evaluation / "manifests" / "ref_manifest.clean.jsonl")
    ) == smoke_evaluation


@pytest.mark.parametrize(
    ("audio_flag", "prediction_flag"),
    [
        ("--audio_manifest", "--prediction_manifest"),
        ("--manifest_base_in", "--asr_manifest"),
        ("--manifest_base_in", "--nemo_manifest"),
    ],
)
def test_manifest_pipeline_accepts_preferred_and_legacy_manifest_names(
    monkeypatch: pytest.MonkeyPatch,
    audio_flag: str,
    prediction_flag: str,
) -> None:
    from manifest_pipeline import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifest_pipeline.py",
            "--dataset_root",
            "dataset",
            audio_flag,
            "segments.jsonl",
            prediction_flag,
            "predictions.jsonl",
        ],
    )

    args = parse_args()

    assert args.manifest_base_in == "segments.jsonl"
    assert args.asr_manifest == ["predictions.jsonl"]



def test_supplied_prediction_manifest_replaces_stale_base_hypotheses() -> None:
    from manifest_pipeline import _attach_authoritative_hypotheses

    raw = pd.DataFrame(
        [
            {"audio_filepath": "audio/a.wav", "pred_text": "stale a"},
            {"audio_filepath": "audio/b.wav", "pred_text": "stale b"},
            {"audio_filepath": "audio/c.wav", "pred_text": "stale c"},
        ]
    )
    supplied = pd.DataFrame(
        [
            {
                "audio_path": "audio/a.wav",
                "audio_name": "a.wav",
                "audio_stem": "a",
                "hyp_text": "fresh a",
            },
            {
                "audio_path": "audio/c.wav",
                "audio_name": "c.wav",
                "audio_stem": "c",
                "hyp_text": "nan",
            },
        ]
    )

    result = _attach_authoritative_hypotheses(
        raw,
        supplied,
        match_on="stem",
        logger=logging.getLogger("test"),
    )

    assert result["pred_text"].tolist() == ["fresh a", "", "nan"]


def test_eval_manifest_rejects_corrupted_reference_text(tmp_path: Path) -> None:
    manifest = tmp_path / "corrupt.jsonl"
    manifest.write_text(
        '{"audio_filepath":"a.wav","ref_text":"safi","can_text":"ng\ufffdambo","pred_text":"hyp"}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Reference integrity check failed"):
        _load_eval_manifest(manifest)


def test_eval_manifest_legacy_recovery_preserves_corrupted_reference(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "legacy_corrupt.jsonl"
    manifest.write_text(
        '{"audio_filepath":"a.wav","ref_text":"safi","can_text":"ng\ufffdambo","pred_text":"hyp"}\n',
        encoding="utf-8",
    )

    result = _load_eval_manifest(manifest, validate_references=False)

    assert result.loc[0, "manifest_can_text"] == "ng\ufffdambo"


def test_eval_manifest_allows_ipa_and_corrupted_model_output(tmp_path: Path) -> None:
    ipa_ref = "\u014b\u0261ombe"
    model_output = "model\ufffdoutput"
    manifest = tmp_path / "valid.jsonl"
    manifest.write_text(
        '{"audio_filepath":"a.wav","ref_text":"'
        + ipa_ref
        + '","can_text":"ngombe","pred_text":"'
        + model_output
        + '"}\n',
        encoding="utf-8",
    )

    result = _load_eval_manifest(manifest)

    assert result.loc[0, "manifest_ref_text"] == ipa_ref
    assert result.loc[0, "manifest_hyp_text"] == model_output
