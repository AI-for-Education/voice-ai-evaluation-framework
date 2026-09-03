from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.build_leaderboard_html import _public_scoring_route, build_html


SYSTEM_ID = "babygruut-sw-fixture123456"


def _write_fixture(root: Path, *, include_policy: bool = True) -> None:
    ipa_dir = root / "ipa"
    ipa_dir.mkdir(parents=True)
    row = {
        "rank": "1",
        "model_label": "Bookbot · fixture model (greedy)",
        "model_group": "Bookbot",
        "g2p_system_id": SYSTEM_ID,
        "g2p_display_name": "babygruut",
        "target_inventory": "babygruut_sw_ipa_v1",
        "global_per": "12.34",
        "global_mer": "8.90",
        "native_output_units": "orthographic",
        "completed_at": "2026-08-19T10:37:58+00:00",
        # Legacy exports may contain this column. It must never be rendered.
        "evaluation_status": "scored (full run, 5,143 items)",
    }
    csv_path = ipa_dir / f"leaderboard_{SYSTEM_ID}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    metadata: dict[str, object] = {
        "schema_version": 6,
        "generated_at": "2026-08-19T10:37:58+00:00",
        "leaderboards": {
            "ipa_by_system": {
                SYSTEM_ID: {
                    "rows": 1,
                    "ranking_metric": "per",
                }
            }
        },
    }
    if include_policy:
        metadata["eligibility_policy"] = {
            "completed_evaluations_only": True,
            "representation_compatible_only": True,
            "smoke_tests_excluded": True,
        }
    (root / "leaderboard_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_html_never_renders_legacy_run_status_or_counts(tmp_path: Path) -> None:
    root = tmp_path / "leaderboards"
    _write_fixture(root)

    output = build_html(root, SYSTEM_ID)
    rendered = output.read_text(encoding="utf-8")

    assert "babygruut" in rendered
    assert "Bookbot" in rendered
    assert "full run" not in rendered.lower()
    assert "5,143 items" not in rendered
    assert "evaluation_status" not in rendered
    assert 'class="status"' not in rendered


def test_html_requires_explicit_eligibility_policy(tmp_path: Path) -> None:
    root = tmp_path / "leaderboards"
    _write_fixture(root, include_policy=False)

    with pytest.raises(ValueError, match="eligibility policy"):
        build_html(root, SYSTEM_ID)


def test_public_route_has_no_arrow_when_native_ipa_is_compatible() -> None:
    route = _public_scoring_route(
        {
            "native_output_units": "phoneme",
            "g2p_display_name": "babygruut",
            "hypothesis_route": (
                "native IPA bookbot_gruut_sw_v1 — already compatible with "
                "babygruut_sw_ipa_v1 (no phoneme conversion)"
            ),
        }
    )

    assert route == (
        "native IPA — already babygruut-compatible (no phoneme conversion)"
    )
    assert "→" not in route


def test_public_route_keeps_arrow_for_real_inventory_conversion() -> None:
    route = _public_scoring_route(
        {
            "native_output_units": "phoneme",
            "g2p_display_name": "Africa G2P",
            "hypothesis_route": (
                "native IPA bookbot_gruut_sw_v1 -> africa_g2p_swh_ipa_v1"
            ),
        }
    )

    assert route == "native IPA → Africa G2P IPA"
