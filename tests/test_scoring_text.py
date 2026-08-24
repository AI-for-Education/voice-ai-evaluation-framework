from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from egra_eval2.eval_utils import text_normalize
from egra_eval2.reference.ipa_inventory_maps import (
    BOOKBOT_GRUUT_TO_AFRICA_G2P,
    IPAInventoryAdapter,
    IPA_INVENTORY_ADAPTERS,
)
from egra_eval2.scoring_text import (
    ScoringRepresentationError,
    align_ipa_inventory,
    prepare_scoring_texts,
)


def _paths(tmp_path: Path, profile: dict) -> tuple[Path, Path]:
    dataset = tmp_path / "input" / "dataset"
    dataset.mkdir(parents=True)
    run_name = "model_run"
    manifest = tmp_path / "output" / "evaluations" / run_name / "manifests" / "clean.jsonl"
    manifest.parent.mkdir(parents=True)
    transcript_dir = tmp_path / "output" / "transcripts" / run_name
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "run_metadata.json").write_text(
        json.dumps({"profile": profile}),
        encoding="utf-8",
    )
    return dataset, manifest


def _manifest_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audio_path": "segments/one.wav",
                "manifest_can_text": "orthographic can",
                "manifest_ref_text": "orthographic ref",
                "manifest_hyp_text": "ɑ  m",
            }
        ]
    )


def _write_ipa_view(dataset: Path, inventory: str) -> None:
    view_dir = dataset / "_derived" / "reference_views"
    view_dir.mkdir(parents=True)
    view_name = "ipa.jsonl"
    (view_dir / view_name).write_text(
        json.dumps(
            {
                "audio_filepath": "segments/one.wav",
                "can_text": "k a n",
                "ref_text": "r e f",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (view_dir / "reference_views.metadata.json").write_text(
        json.dumps(
            {
                "ipa": {
                    "path": view_name,
                    "inventory": inventory,
                    "language": "swh",
                    "producer": "africa-g2p",
                    "producer_version": "0.1.0-test",
                }
            }
        ),
        encoding="utf-8",
    )


def _load_resolve_outputs(monkeypatch: pytest.MonkeyPatch):
    """Import the output resolver without loading the full evaluation stack."""
    evaluate_stub = types.ModuleType("egra_eval2.evaluate")
    evaluate_stub.aggregate_row_scores = lambda *args, **kwargs: {}
    evaluate_stub.evaluate_rows = lambda value: value
    monkeypatch.setitem(sys.modules, "egra_eval2.evaluate", evaluate_stub)
    sys.modules.pop("eval_pipeline2", None)

    from eval_pipeline2 import resolve_outputs

    return resolve_outputs


def _evaluation_args(
    run_root: Path,
    *,
    scoring_representation: str,
    derive_output_root: bool = False,
    out_csv: Path | None = None,
) -> SimpleNamespace:
    """Build the common CLI-shaped namespace used by output-routing tests."""
    manifest_name = "ref_manifest.clean.jsonl" if derive_output_root else "clean.jsonl"
    return SimpleNamespace(
        output_root=None if derive_output_root else str(run_root),
        manifest_in=str(run_root / "manifests" / manifest_name),
        scoring_representation=scoring_representation,
        out_csv=str(out_csv) if out_csv is not None else None,
        summary_can_ref_dir=None,
        summary_can_hyp_dir=None,
        summary_ref_hyp_dir=None,
    )


# Reference and hypothesis representation routing


def test_orthographic_profile_keeps_existing_manifest_text(tmp_path: Path) -> None:
    dataset, manifest = _paths(tmp_path, {"output_units": "orthographic"})
    source = _manifest_df()

    result, units = prepare_scoring_texts(
        source,
        dataset_root=dataset,
        manifest_in=manifest,
        logger=logging.getLogger("test_scoring_text"),
    )

    assert units == "orthographic"
    assert result is source


def test_orthographic_profile_can_be_scored_through_shared_g2p_ipa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = "africa_g2p_swh_ipa_v1"
    dataset, manifest = _paths(
        tmp_path,
        {"output_units": "orthographic", "language": "en"},
    )
    _write_ipa_view(dataset, inventory)
    source = _manifest_df()
    source.loc[0, "manifest_hyp_text"] = "kijiko"
    calls: list[tuple[list[str], str]] = []

    def fake_phonemize(texts, *, language):
        calls.append((list(texts), language))
        return ["k i j i k o" for _ in texts]

    monkeypatch.setattr(
        "egra_eval2.scoring_text.phonemize_with_africa_g2p",
        fake_phonemize,
    )
    result, units = prepare_scoring_texts(
        source,
        dataset_root=dataset,
        manifest_in=manifest,
        logger=logging.getLogger("test_scoring_text"),
        scoring_representation="ipa",
    )

    assert units == "phoneme"
    assert calls == [(["kijiko"], "swh")]
    assert result.loc[0, "manifest_can_text"] == "k a n"
    assert result.loc[0, "manifest_ref_text"] == "r e f"
    assert result.loc[0, "manifest_hyp_text"] == "k i j i k o"
    aligned = manifest.with_name("clean.ipa_aligned.jsonl")
    assert json.loads(aligned.read_text(encoding="utf-8")) == {
        "audio_filepath": "segments/one.wav",
        "can_text": "k a n",
        "ref_text": "r e f",
        "pred_text": "k i j i k o",
    }

def test_phoneme_profile_uses_matching_ipa_reference(tmp_path: Path) -> None:
    inventory = "africa_g2p_swh_ipa_v1"
    dataset, manifest = _paths(
        tmp_path,
        {
            "output_units": "phoneme",
            "output_notation": "ipa",
            "output_inventory": inventory,
        },
    )
    _write_ipa_view(dataset, inventory)

    result, units = prepare_scoring_texts(
        _manifest_df(),
        dataset_root=dataset,
        manifest_in=manifest,
        logger=logging.getLogger("test_scoring_text"),
    )

    assert units == "phoneme"
    assert result.loc[0, "manifest_can_text"] == "k a n"
    assert result.loc[0, "manifest_ref_text"] == "r e f"
    assert result.loc[0, "manifest_hyp_text"] == "ɑ m"
    aligned = manifest.with_name("clean.ipa_aligned.jsonl")
    assert [json.loads(line) for line in aligned.read_text(encoding="utf-8").splitlines()] == [
        {
            "audio_filepath": "segments/one.wav",
            "can_text": "k a n",
            "ref_text": "r e f",
            "pred_text": "ɑ m",
        }
    ]


# Native phoneme inventory alignment


def test_reviewed_bookbot_ipa_inventory_mapping() -> None:
    assert BOOKBOT_GRUUT_TO_AFRICA_G2P.status == "approved"
    assert not BOOKBOT_GRUUT_TO_AFRICA_G2P.unresolved
    assert align_ipa_inventory(
        "t͡ʃ ɾ ᵐɓ ᵑg ⁿɗ ⁿɗ͡ʒ nj n j",
        source="bookbot_gruut_sw_v1",
        target="africa_g2p_swh_ipa_v1",
    ) == "ʧ r ᵐb ᵑɡ ⁿd ⁿdʒ ɲ n j"


def test_bookbot_ipa_inventory_mapping_rejects_unknown_symbols() -> None:
    with pytest.raises(ScoringRepresentationError, match="Unknown"):
        align_ipa_inventory(
            "ɲ",
            source="bookbot_gruut_sw_v1",
            target="africa_g2p_swh_ipa_v1",
        )


def test_inventory_mapping_is_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        IPA_INVENTORY_ADAPTERS,
        ("source", "target"),
        IPAInventoryAdapter(
            source="source",
            target="target",
            status="approved",
            source_tokens=("a", "b"),
            replacements={"a": "b", "b": "c"},
            unresolved={},
        ),
    )

    assert align_ipa_inventory("a b", source="source", target="target") == "b c"


def test_matching_inventory_normalizes_missing_hypothesis() -> None:
    assert align_ipa_inventory(None, source="same", target="same") == ""


def test_shared_normalizer_preserves_ipa_combining_marks() -> None:
    assert text_normalize("t\u0361ʃ ə\u0303!") == "t\u0361ʃ ə\u0303"


# Manifest and completed-run metadata requirements


def test_standard_evaluation_manifest_requires_run_metadata(tmp_path: Path) -> None:
    dataset, manifest = _paths(tmp_path, {"output_units": "orthographic"})
    metadata_path = (
        tmp_path / "output" / "transcripts" / "model_run" / "run_metadata.json"
    )
    metadata_path.unlink()

    with pytest.raises(ScoringRepresentationError, match="metadata not found"):
        prepare_scoring_texts(
            _manifest_df(),
            dataset_root=dataset,
            manifest_in=manifest,
            logger=logging.getLogger("test_scoring_text"),
        )


def test_manual_manifest_keeps_legacy_orthographic_default(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = _manifest_df()

    result, units = prepare_scoring_texts(
        source,
        dataset_root=dataset,
        manifest_in=tmp_path / "manual.jsonl",
        logger=logging.getLogger("test_scoring_text"),
    )

    assert units == "orthographic"
    assert result is source


def test_rejects_non_object_ipa_reference_row(tmp_path: Path) -> None:
    inventory = "africa_g2p_swh_ipa_v1"
    dataset, manifest = _paths(
        tmp_path,
        {
            "output_units": "phoneme",
            "output_notation": "ipa",
            "output_inventory": inventory,
        },
    )
    _write_ipa_view(dataset, inventory)
    view = dataset / "_derived" / "reference_views" / "ipa.jsonl"
    view.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    with pytest.raises(ScoringRepresentationError, match="is not an object"):
        prepare_scoring_texts(
            _manifest_df(),
            dataset_root=dataset,
            manifest_in=manifest,
            logger=logging.getLogger("test_scoring_text"),
        )


def test_phoneme_manifest_requires_audio_and_hypothesis_columns(
    tmp_path: Path,
) -> None:
    inventory = "africa_g2p_swh_ipa_v1"
    dataset, manifest = _paths(
        tmp_path,
        {
            "output_units": "phoneme",
            "output_notation": "ipa",
            "output_inventory": inventory,
        },
    )
    _write_ipa_view(dataset, inventory)

    with pytest.raises(ScoringRepresentationError, match="manifest_hyp_text"):
        prepare_scoring_texts(
            pd.DataFrame({"audio_path": ["segments/one.wav"]}),
            dataset_root=dataset,
            manifest_in=manifest,
            logger=logging.getLogger("test_scoring_text"),
        )

# Representation-specific output routing


def test_explicit_ipa_evaluation_uses_separate_output_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_outputs = _load_resolve_outputs(monkeypatch)

    run_root = tmp_path / "output" / "evaluations" / "model_run"
    args = _evaluation_args(
        run_root,
        scoring_representation="ipa",
        derive_output_root=True,
    )

    outputs = resolve_outputs(args, logging.getLogger("test_scoring_text"))

    assert outputs["base"] == run_root / "ipa"
    assert Path(args.out_csv) == run_root / "ipa" / "egra_eval_detailed.csv"


def test_legacy_orthographic_preserves_phoneme_manifest_for_restoration(
    tmp_path: Path,
) -> None:
    dataset, manifest = _paths(
        tmp_path,
        {
            "output_units": "phoneme",
            "output_notation": "ipa",
            "output_inventory": "bookbot_gruut_sw_v1",
        },
    )
    source = _manifest_df()

    result, units = prepare_scoring_texts(
        source,
        dataset_root=dataset,
        manifest_in=manifest,
        logger=logging.getLogger("test_scoring_text"),
        scoring_representation="legacy_orthographic",
    )

    assert units == "orthographic"
    assert result is source


def test_legacy_orthographic_rejects_orthographic_model_profile(
    tmp_path: Path,
) -> None:
    dataset, manifest = _paths(tmp_path, {"output_units": "orthographic"})

    with pytest.raises(
        ScoringRepresentationError,
        match="only valid for native phoneme output",
    ):
        prepare_scoring_texts(
            _manifest_df(),
            dataset_root=dataset,
            manifest_in=manifest,
            logger=logging.getLogger("test_scoring_text"),
            scoring_representation="legacy_orthographic",
        )


@pytest.mark.parametrize(
    ("requested", "units", "folder"),
    [
        ("auto", "orthographic", "orthographic"),
        ("orthographic", "orthographic", "orthographic"),
        ("auto", "phoneme", "ipa"),
        ("ipa", "phoneme", "ipa"),
        ("legacy_orthographic", "orthographic", "orthographic_legacy"),
    ],
)
def test_evaluation_outputs_are_namespaced_by_representation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    units: str,
    folder: str,
) -> None:
    resolve_outputs = _load_resolve_outputs(monkeypatch)

    run_root = tmp_path / "output" / "evaluations" / "model_run"
    args = _evaluation_args(
        run_root,
        scoring_representation=requested,
    )

    outputs = resolve_outputs(
        args,
        logging.getLogger("test_scoring_text"),
        scoring_units=units,
    )
    assert outputs["base"] == run_root / folder
    assert Path(args.out_csv) == run_root / folder / "egra_eval_detailed.csv"


def test_evaluation_rejects_custom_output_outside_representation_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_outputs = _load_resolve_outputs(monkeypatch)

    run_root = tmp_path / "output" / "evaluations" / "model_run"
    args = _evaluation_args(
        run_root,
        scoring_representation="orthographic",
        out_csv=run_root / "egra_eval_detailed.csv",
    )

    with pytest.raises(SystemExit, match="representation output directory"):
        resolve_outputs(
            args,
            logging.getLogger("test_scoring_text"),
            scoring_units="orthographic",
        )
