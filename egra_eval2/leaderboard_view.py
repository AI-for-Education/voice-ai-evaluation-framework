"""Pure presentation transforms for the Streamlit leaderboard."""

from __future__ import annotations

import pandas as pd


RESULT_SECTIONS = (
    "global",
    "passage_passage",
    "syllables_grid",
    "syllables_isolated",
    "nonwords_grid",
    "nonwords_isolated",
    "letters_grid",
    "letters_isolated",
)

RESULT_SECTION_LABELS = {
    "global": "Overall",
    "passage_passage": "Passage",
    "syllables_grid": "Syllables · grid",
    "syllables_isolated": "Syllables · isolated",
    "nonwords_grid": "Non-words · grid",
    "nonwords_isolated": "Non-words · isolated",
    "letters_grid": "Letters · grid",
    "letters_isolated": "Letters · isolated",
}

AUDIT_ONLY_COLUMNS = (
    "summary_path",
    "inference_setup_id",
    # Deprecated v1 aliases remain in CSV output but are hidden in the UI.
    "model_id",
    "run_name",
    "platform",
    "artifact_context",
    "preprocessing_context",
    "runtime_context",
    "model_group",
    "model_name",
    "model_variant",
    "official_model_url",
    "official_model_url_note",
    "architecture_evidence_status",
)


def _result_columns(metric: str) -> list[str]:
    columns = [f"global_{metric}", "global_mer"]
    for section in RESULT_SECTIONS[1:]:
        columns.extend([f"{section}_{metric}", f"{section}_mer"])
        columns.append(
            f"{section}_accuracy"
            if section.endswith("_isolated")
            else f"{section}_corr"
        )
    return columns


def display_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Put analytical results before configuration and provenance details."""
    visible = frame.drop(columns=list(AUDIT_ONLY_COLUMNS), errors="ignore").copy()
    primary_columns = ["rank", "model_label", *_result_columns(metric)]
    detail_columns = [
        column for column in visible.columns if column not in primary_columns
    ]
    visible = visible[
        [column for column in primary_columns if column in visible.columns]
        + detail_columns
    ]

    labels = {
        "model_label": "group · model · variant (decoder)",
        "evaluation_status": "status",
        "run_id": "run",
        "inference_setup_id": "inference setup ID",
        "architecture": "architecture",
        "execution_target": "execution target",
        "decoding": "decoder",
        "model_artifact": "model artifact",
        "inference_setup": "inference setup",
        "execution_stack": "execution stack",
        "context_evidence": "factor evidence",
        "native_output_units": "native output",
        "hypothesis_route": "scoring route",
        "scored_hypothesis": "scored hypothesis",
        "postprocessing_method": "post-processing",
        "postprocessed_rows": "adjusted rows",
        "postprocessed_rows_pct": "adjusted rows (%)",
        "postprocessing_words_removed": "words removed",
        "postprocessing_audit_source": "audit source",
        "completed_at": "completed",
    }
    for section in RESULT_SECTIONS:
        task_label = RESULT_SECTION_LABELS[section]
        labels[f"{section}_{metric}"] = f"{task_label} · {metric.upper()}"
        labels[f"{section}_mer"] = f"{task_label} · MER"
        if section.endswith("_isolated"):
            labels[f"{section}_accuracy"] = f"{task_label} · accuracy"
        elif section != "global":
            labels[f"{section}_corr"] = f"{task_label} · correlation (r)"
    return visible.rename(columns=labels)
