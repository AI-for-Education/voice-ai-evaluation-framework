#!/usr/bin/env python3
"""Streamlit leaderboards for representation-compatible EGRA evaluations."""

from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd
import streamlit as st

from egra_eval2.leaderboard import LeaderboardError, build_leaderboards
from egra_eval2.leaderboard_view import display_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--evaluations-root",
        "--evaluations_root",
        default="input_output_data/output/evaluations",
    )
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def load_leaderboards(
    evaluations_root: str,
    latest_only: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return build_leaderboards(evaluations_root, latest_only=latest_only)


def render_leaderboard(
    frame: pd.DataFrame,
    skipped: list[str],
    *,
    namespace: str,
    metric: str,
    description: str,
    download_name: str | None = None,
) -> None:
    st.caption(
        description
        + " Lower error rates rank higher. Results are presented first in this "
        "order: overall, passage, syllables, non-words, and letters; configuration "
        "and provenance details follow. Passage/grid tasks report correct-count "
        "Pearson correlation; isolated tasks report accuracy. Post-processing "
        "columns identify any adjusted hypotheses used for scoring. Model "
        "artifact, inference setup, and execution stack appear in that "
        "comparison order; Android proxy results are not physical-phone "
        "performance measurements."
    )
    if frame.empty:
        st.info(f"No eligible {namespace} evaluations found.")
    else:
        displayed = display_frame(frame, metric)
        st.dataframe(displayed, hide_index=True, width="stretch")
        st.download_button(
            f"Download {namespace} leaderboard (CSV)",
            data=frame.to_csv(index=False, float_format="%.4f").encode("utf-8"),
            file_name=download_name or f"leaderboard_{namespace}.csv",
            mime="text/csv",
            key=f"download_{namespace}_{download_name or 'leaderboard'}",
        )

    if skipped:
        with st.expander(f"Skipped {len(skipped)} ineligible run(s)"):
            for reason in skipped:
                st.code(reason)


st.set_page_config(page_title="EGRA ASR Leaderboards", layout="wide")
st.title("EGRA ASR Leaderboards")
st.write(
    "Orthographic word scoring is separate from IPA phoneme scoring. Every IPA "
    "leaderboard is also isolated by the exact G2P tool, version, resources, and "
    "configuration used to create it. Archived, incompatible, and smoke-test "
    "evaluations are never included."
)

args = parse_args()
default_root = os.environ.get("EGRA_EVALUATIONS_ROOT", args.evaluations_root)
evaluations_root = st.text_input("Evaluations root", default_root)
latest_only = st.toggle(
    "Show only the newest completed run per inference setup",
    value=True,
)

try:
    frames, skipped_runs = load_leaderboards(evaluations_root, latest_only)
except LeaderboardError as exc:
    st.error(str(exc))
    st.stop()

ipa_system_ids = list(frames["ipa_by_system"])
tab_labels = ["Orthographic (WER)"] + [
    f"IPA · {system_id} (PER)" for system_id in ipa_system_ids
]
if not ipa_system_ids:
    tab_labels.append("IPA (PER)")
tabs = st.tabs(tab_labels)
orthographic_tab = tabs[0]
with orthographic_tab:
    render_leaderboard(
        frames["orthographic"],
        skipped_runs["orthographic"],
        namespace="orthographic",
        metric="wer",
        description=(
            "Native written-text hypotheses scored by word error rate. This is "
            "orthographic scoring, not a character/grapheme error rate."
        ),
    )

if not ipa_system_ids:
    with tabs[1]:
        st.info("No exact-system IPA evaluations found.")
else:
    for tab, system_id in zip(tabs[1:], ipa_system_ids):
        frame = frames["ipa_by_system"][system_id]
        system_name = (
            str(frame.iloc[0]["g2p_display_name"])
            if not frame.empty
            else system_id
        )
        with tab:
            render_leaderboard(
                frame,
                skipped_runs["ipa_by_system"][system_id],
                namespace=f"IPA / {system_id}",
                metric="per",
                description=(
                    f"Phoneme error rate using only {system_name}. Native IPA "
                    "hypotheses are admitted only through an approved inventory "
                    "adapter for this exact target system."
                ),
                download_name=f"leaderboard_{system_id}.csv",
            )
