from __future__ import annotations

import pandas as pd

from egra_eval2.leaderboard_view import RESULT_SECTIONS, display_frame


def test_display_orders_analysis_before_configuration_details() -> None:
    row: dict[str, object] = {
        "rank": 1,
        "model_label": "Bookbot · XLS-R 300M · orthographic (greedy)",
        "evaluation_status": "scored",
        "architecture": "XLS-R",
        "run_id": "example_run",
        "inference_setup_id": "bookbot-orthographic-ctc",
        "model_group": "Bookbot",
        "model_name": "XLS-R 300M",
        "model_variant": "orthographic",
        "official_model_url": "https://example.test/model",
        "official_model_url_note": "model-owner listing",
        "architecture_evidence_status": "artifact_verified",
        "summary_path": "output/evaluations/example_run/summary.txt",
        "global_wer": 10.0,
        "global_mer": 8.0,
    }
    expected_results = ["Overall · WER", "Overall · MER"]
    for section in RESULT_SECTIONS[1:]:
        label = {
            "passage_passage": "Passage",
            "syllables_grid": "Syllables · grid",
            "syllables_isolated": "Syllables · isolated",
            "nonwords_grid": "Non-words · grid",
            "nonwords_isolated": "Non-words · isolated",
            "letters_grid": "Letters · grid",
            "letters_isolated": "Letters · isolated",
        }[section]
        row[f"{section}_wer"] = 10.0
        row[f"{section}_mer"] = 8.0
        outcome = "accuracy" if section.endswith("_isolated") else "corr"
        row[f"{section}_{outcome}"] = 0.9
        expected_results.extend(
            [
                f"{label} · WER",
                f"{label} · MER",
                f"{label} · {'accuracy' if outcome == 'accuracy' else 'correlation (r)'}",
            ]
        )

    displayed = display_frame(pd.DataFrame([row]), "wer")

    assert displayed.columns.tolist() == [
        "rank",
        "group · model · variant (decoder)",
        *expected_results,
        "status",
        "architecture",
    ]
