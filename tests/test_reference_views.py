from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from egra_eval2.reference.views import (
    ReferenceViewError,
    build_reference_views,
    phonemize_with_africa_g2p,
    prepare_ipa_reference_view,
)
from tools.build_reference_views import parse_args


def _write_manifest(path: Path) -> None:
    rows = [
        {
            "audio_filepath": "segments/first.wav",
            "duration": 1.25,
            "can_text": "Kijiko!",
            "ref_text": "kijiko",
            "pred_text": "must not be copied",
        },
        {
            "audio_filepath": "segments/second.wav",
            "duration": 2.5,
            "can_text": "Dhahabu",
            "ref_text": "dhahabu",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fake_ipa(texts):
    return [f"ɪ p a {text}" for text in texts]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_phonemize_with_africa_g2p_calls_the_package_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakePipeline:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def run(self, texts, **kwargs):
            calls.append(("run", texts, kwargs))
            return [f"ipa:{text}" for text in texts]

    monkeypatch.setitem(
        sys.modules,
        "africa_g2p",
        SimpleNamespace(AfricaPipeline=FakePipeline),
    )

    result = phonemize_with_africa_g2p(["kijiko", "dhahabu"], language="swh")

    assert result == ["ipa:kijiko", "ipa:dhahabu"]
    assert calls == [
        ("init", {"lang": "swh", "output": "ipa", "unknown": "passthrough"}),
        ("run", ["kijiko", "dhahabu"], {"sep": " "}),
    ]


def test_builds_dataset_adjacent_views_with_provenance(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = tmp_path / "references.jsonl"
    _write_manifest(manifest)

    paths = build_reference_views(
        dataset_root=dataset,
        manifest_in=manifest,
        phonemize=_fake_ipa,
        phonemizer_version="0.1.0-test",
    )

    assert paths.output_dir == dataset / "_derived" / "reference_views"
    assert paths.reused is False
    assert paths.orthographic.name == "orthographic.sw.v1.jsonl"
    assert paths.ipa.name == "phonemic.ipa.africa_g2p.swh.0.1.0-test.jsonl"

    orthographic = _read_jsonl(paths.orthographic)
    ipa = _read_jsonl(paths.ipa)
    assert [row["audio_filepath"] for row in orthographic] == [
        row["audio_filepath"] for row in ipa
    ]
    assert orthographic[0]["can_text"] == "kijiko"
    assert "pred_text" not in orthographic[0]
    assert ipa[0]["can_text"] == "ɪ p a kijiko"

    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert set(metadata) == {"schema_version", "source", "orthographic", "ipa"}
    assert metadata["source"]["rows"] == 2
    assert metadata["orthographic"] == {"path": "orthographic.sw.v1.jsonl"}
    assert metadata["ipa"]["producer"] == "africa-g2p"
    assert metadata["ipa"]["producer_version"] == "0.1.0-test"
    assert metadata["ipa"]["inventory"] == "africa_g2p_swh_ipa_v1"


def test_reuses_only_an_exact_matching_cache(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = tmp_path / "references.jsonl"
    _write_manifest(manifest)

    first = build_reference_views(
        dataset_root=dataset,
        manifest_in=manifest,
        phonemize=_fake_ipa,
        phonemizer_version="test-version",
    )
    second = build_reference_views(
        dataset_root=dataset,
        manifest_in=manifest,
        phonemize=lambda _: pytest.fail("an exact cache must not be regenerated"),
        phonemizer_version="test-version",
    )

    assert first.reused is False
    assert second.reused is True

    manifest.write_text(
        manifest.read_text(encoding="utf-8") + json.dumps(
            {
                "audio_filepath": "segments/third.wav",
                "can_text": "tatu",
                "ref_text": "tatu",
            }
        ) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ReferenceViewError, match="--force"):
        build_reference_views(
            dataset_root=dataset,
            manifest_in=manifest,
            phonemize=_fake_ipa,
            phonemizer_version="test-version",
        )


def test_rejects_duplicate_item_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = tmp_path / "references.jsonl"
    duplicate = {"audio_filepath": "same.wav", "can_text": "a", "ref_text": "a"}
    manifest.write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ReferenceViewError, match="Duplicate audio_filepath"):
        build_reference_views(
            dataset_root=dataset,
            manifest_in=manifest,
            phonemize=_fake_ipa,
            phonemizer_version="test-version",
        )


def test_cli_defaults_to_swahili_and_dataset_adjacent_output() -> None:
    args = parse_args(["--dataset_root", "dataset", "--manifest_in", "manifest.jsonl"])

    assert args.language == "swh"
    assert args.output_dir is None
    assert args.force is False


def test_rejects_unsafe_language_component(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = tmp_path / "references.jsonl"
    _write_manifest(manifest)

    with pytest.raises(ReferenceViewError, match="Language must contain"):
        build_reference_views(
            dataset_root=dataset,
            manifest_in=manifest,
            language="../swh",
            phonemize=_fake_ipa,
            phonemizer_version="test-version",
        )


def test_manifest_pipeline_prepares_cache_for_all_declared_model_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript_dir = tmp_path / "transcripts" / "phoneme-run"
    transcript_dir.mkdir(parents=True)
    transcript = transcript_dir / "transcriptions.jsonl"
    transcript.touch()
    (transcript_dir / "run_metadata.json").write_text(
        json.dumps(
            {"inference_profile": {"output_units": "phoneme", "language": "sw"}}
        ),
        encoding="utf-8",
    )
    calls = []

    class Views:
        reused = False
        ipa = tmp_path / "view.jsonl"

    monkeypatch.setattr(
        "egra_eval2.reference.views.build_reference_views",
        lambda **kwargs: calls.append(kwargs) or Views(),
    )

    prepare_ipa_reference_view(
        dataset_root=tmp_path,
        audio_manifest="base.jsonl",
        prediction_manifests=[str(transcript)],
        logger=logging.getLogger("test_reference_views"),
    )

    assert calls == [
        {
            "dataset_root": tmp_path,
            "manifest_in": "base.jsonl",
            "language": "swh",
        }
    ]

    (transcript_dir / "run_metadata.json").write_text(
        json.dumps({"inference_profile": {"output_units": "orthographic"}}),
        encoding="utf-8",
    )
    calls.clear()
    prepare_ipa_reference_view(
        dataset_root=tmp_path,
        audio_manifest="base.jsonl",
        prediction_manifests=[str(transcript)],
        logger=logging.getLogger("test_reference_views"),
    )
    assert calls == [
        {
            "dataset_root": tmp_path,
            "manifest_in": "base.jsonl",
            "language": "swh",
        }
    ]
