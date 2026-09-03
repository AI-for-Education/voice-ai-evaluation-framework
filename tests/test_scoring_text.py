from __future__ import annotations

import json
import hashlib
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
    BOOKBOT_GRUUT_TO_BABYGRUUT,
    IPAInventoryAdapter,
    IPA_INVENTORY_ADAPTERS,
    describe_ipa_inventory_route,
)
from egra_eval2.reference.g2p import G2PSystem, make_test_system
from egra_eval2.scoring_text import (
    ScoringContext,
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
        json.dumps({"inference_profile": profile}),
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


def _test_system(
    *,
    tool_id: str = "africa_g2p",
    inventory: str = "africa_g2p_swh_ipa_v1",
    phonemize=lambda values: [str(value) for value in values],
) -> G2PSystem:
    return make_test_system(
        tool_id=tool_id,
        display_name="babygruut" if tool_id == "babygruut" else "Africa G2P",
        language="sw" if tool_id == "babygruut" else "swh",
        inventory=inventory,
        version_value="0.1.0-test",
        phonemize=phonemize,
    )


def _write_ipa_view(dataset: Path, system: G2PSystem) -> Path:
    view_dir = dataset / "_derived" / "reference_views"
    view_dir.mkdir(parents=True)
    orthographic_path = view_dir / "orthographic.sw.v1.jsonl"
    orthographic_path.write_text("{}\n", encoding="utf-8")
    view_name = f"phonemic.ipa.{system.system_id}.jsonl"
    view_path = view_dir / view_name
    view_path.write_text(
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
                "schema_version": 2,
                "orthographic": {
                    "path": orthographic_path.name,
                    "sha256": hashlib.sha256(
                        orthographic_path.read_bytes()
                    ).hexdigest(),
                },
                "ipa_views": {
                    system.system_id: {
                    **system.metadata(),
                    "path": view_name,
                    "sha256": hashlib.sha256(view_path.read_bytes()).hexdigest(),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return view_path


def _ipa_context(system: G2PSystem | None = None) -> ScoringContext:
    selected = system or _test_system()
    return ScoringContext(
        scoring_units="phoneme", namespace="ipa", g2p_system=selected
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
    dataset, manifest = _paths(
        tmp_path,
        {"output_units": "orthographic", "language": "en"},
    )
    source = _manifest_df()
    source.loc[0, "manifest_hyp_text"] = "kijiko"
    calls: list[list[str]] = []

    def fake_phonemize(texts):
        calls.append(list(texts))
        return ["k i j i k o" for _ in texts]

    system = _test_system(phonemize=fake_phonemize)
    _write_ipa_view(dataset, system)
    result, units = prepare_scoring_texts(
        source,
        dataset_root=dataset,
        manifest_in=manifest,
        logger=logging.getLogger("test_scoring_text"),
        scoring_representation="ipa",
        g2p_system=system,
    )

    assert units == "phoneme"
    assert calls == [["kijiko"]]
    assert result.loc[0, "manifest_can_text"] == "k a n"
    assert result.loc[0, "manifest_ref_text"] == "r e f"
    assert result.loc[0, "manifest_hyp_text"] == "k i j i k o"
    aligned = manifest.with_name(
        f"clean.ipa_aligned.{system.system_id}.jsonl"
    )
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
    system = _test_system(inventory=inventory)
    _write_ipa_view(dataset, system)

    prepared = prepare_scoring_texts(
        _manifest_df(),
        dataset_root=dataset,
        manifest_in=manifest,
        logger=logging.getLogger("test_scoring_text"),
        g2p_system=system,
    )
    result, units = prepared

    assert units == "phoneme"
    assert result.loc[0, "manifest_can_text"] == "k a n"
    assert result.loc[0, "manifest_ref_text"] == "r e f"
    assert result.loc[0, "manifest_hyp_text"] == "ɑ m"
    assert prepared.context.hypothesis_route == (
        "native IPA africa_g2p_swh_ipa_v1 — no phoneme conversion"
    )
    aligned = manifest.with_name(
        f"clean.ipa_aligned.{system.system_id}.jsonl"
    )
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


def test_reviewed_bookbot_to_babygruut_identity_mapping() -> None:
    assert BOOKBOT_GRUUT_TO_BABYGRUUT.status == "approved"
    assert align_ipa_inventory(
        "t͡ʃ ɾ ᵐɓ",
        source="bookbot_gruut_sw_v1",
        target="babygruut_sw_ipa_v1",
    ) == "t͡ʃ ɾ ᵐɓ"
    assert describe_ipa_inventory_route(
        "bookbot_gruut_sw_v1", "babygruut_sw_ipa_v1"
    ) == (
        "native IPA bookbot_gruut_sw_v1 — already compatible with "
        "babygruut_sw_ipa_v1 (no phoneme conversion)"
    )


def test_changed_ipa_inventory_route_keeps_directional_arrow() -> None:
    assert describe_ipa_inventory_route(
        "bookbot_gruut_sw_v1", "africa_g2p_swh_ipa_v1"
    ) == "native IPA bookbot_gruut_sw_v1 -> africa_g2p_swh_ipa_v1"


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


def test_manual_manifest_defaults_to_orthographic_scoring(tmp_path: Path) -> None:
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
    system = _test_system(inventory=inventory)
    view = _write_ipa_view(dataset, system)
    view.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    metadata_path = view.parent / "reference_views.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["ipa_views"][system.system_id]["sha256"] = hashlib.sha256(
        view.read_bytes()
    ).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ScoringRepresentationError, match="is not an object"):
        prepare_scoring_texts(
            _manifest_df(),
            dataset_root=dataset,
            manifest_in=manifest,
            logger=logging.getLogger("test_scoring_text"),
            g2p_system=system,
        )


def test_scoring_rejects_a_broken_unselected_reference_view(tmp_path: Path) -> None:
    inventory = "africa_g2p_swh_ipa_v1"
    dataset, manifest = _paths(
        tmp_path,
        {
            "output_units": "phoneme",
            "output_notation": "ipa",
            "output_inventory": inventory,
        },
    )
    system = _test_system(inventory=inventory)
    view = _write_ipa_view(dataset, system)
    metadata_path = view.parent / "reference_views.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["ipa_views"]["unselected-system"] = {
        "path": "missing.jsonl",
        "sha256": "0" * 64,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ScoringRepresentationError, match="unselected-system"):
        prepare_scoring_texts(
            _manifest_df(),
            dataset_root=dataset,
            manifest_in=manifest,
            logger=logging.getLogger("test_scoring_text"),
            g2p_system=system,
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
    system = _test_system(inventory=inventory)
    _write_ipa_view(dataset, system)

    with pytest.raises(ScoringRepresentationError, match="manifest_hyp_text"):
        prepare_scoring_texts(
            pd.DataFrame({"audio_path": ["segments/one.wav"]}),
            dataset_root=dataset,
            manifest_in=manifest,
            logger=logging.getLogger("test_scoring_text"),
            g2p_system=system,
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

    system = _test_system()
    outputs = resolve_outputs(
        args,
        logging.getLogger("test_scoring_text"),
        scoring_context=_ipa_context(system),
    )

    assert outputs["base"] == run_root / "ipa" / system.system_id
    assert Path(args.out_csv) == (
        run_root / "ipa" / system.system_id / "egra_eval_detailed.csv"
    )


@pytest.mark.parametrize(
    ("requested", "units", "folder"),
    [
        ("auto", "orthographic", "orthographic"),
        ("orthographic", "orthographic", "orthographic"),
        ("auto", "phoneme", "ipa"),
        ("ipa", "phoneme", "ipa"),
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

    system = _test_system()
    context = _ipa_context(system) if units == "phoneme" else None
    outputs = resolve_outputs(
        args,
        logging.getLogger("test_scoring_text"),
        scoring_units=units,
        scoring_context=context,
    )
    expected = (
        run_root / folder / system.system_id
        if folder == "ipa"
        else run_root / folder
    )
    assert outputs["base"] == expected
    assert Path(args.out_csv) == expected / "egra_eval_detailed.csv"


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
