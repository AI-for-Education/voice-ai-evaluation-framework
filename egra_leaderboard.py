#!/usr/bin/env python3
"""Streamlit leaderboards for representation-compatible EGRA evaluations."""

from __future__ import annotations

import argparse
import os

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
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    return build_leaderboards(evaluations_root, latest_only=latest_only)


def render_leaderboard(
    frame: pd.DataFrame,
    skipped: list[str],
    *,
    namespace: str,
    metric: str,
    description: str,
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
            file_name=f"leaderboard_{namespace}.csv",
            mime="text/csv",
            key=f"download_{namespace}_leaderboard",
        )

    if skipped:
        with st.expander(f"Skipped {len(skipped)} ineligible run(s)"):
            for reason in skipped:
                st.code(reason)


st.set_page_config(page_title="EGRA ASR Leaderboards", layout="wide")
st.title("EGRA ASR Leaderboards")
st.write(
    "Orthographic word scoring and IPA phoneme scoring are ranked separately. "
    "Archived and representation-incompatible evaluations are never included."
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

orthographic_tab, ipa_tab = st.tabs(["Orthographic (WER)", "IPA (PER)"])
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

with ipa_tab:
    render_leaderboard(
        frames["ipa"],
        skipped_runs["ipa"],
        namespace="ipa",
        metric="per",
        description=(
            "Native IPA hypotheses and orthographic hypotheses converted through "
            "the approved Africa G2P route, scored by phoneme error rate."
        ),
    )
