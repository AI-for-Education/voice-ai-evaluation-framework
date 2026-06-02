#!/usr/bin/env python3
"""EGRA Evaluation Dashboard — summary view with filters."""

import argparse
import math
import os
import urllib.request
from pathlib import Path

import io

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EGRA Evaluation Dashboard", layout="wide")
st.title("EGRA Evaluation Dashboard")

SUMMARY_KEYS = [
    "global",
    "passage_passage",
    "syllables_grid",
    "syllables_isolated",
    "nonwords_grid",
    "nonwords_isolated",
    "letters_grid",
    "letters_isolated",
]
GRID_PASSAGE_TYPES = [
    "passage_passage",
    "syllables_grid",
    "nonwords_grid",
    "letters_grid",
]
ISOLATED_TYPES = [
    "syllables_isolated",
    "nonwords_isolated",
    "letters_isolated",
]
CSV_DEFAULT = "./egra_eval_detailed.csv"


# --- Aggregation helpers (mirrors egra_eval2/evaluate.py and eval_utils.py) ---

def _safe_div(num: float, den: float) -> float:
    return num / den if den else math.nan


def _precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _accuracy(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def _error_rate(group: pd.DataFrame, prefix: str) -> float:
    S = group[f"S_{prefix}"].sum()
    D = group[f"D_{prefix}"].sum()
    I = group[f"I_{prefix}"].sum()
    N = group[f"N_{prefix}"].sum()
    return _safe_div((S + D + I) * 100.0, N)


def aggregate_scores(df: pd.DataFrame, detailed: bool) -> dict:
    scores: dict = {}

    # Global
    scores["global"] = {
        "wer": _error_rate(df, "ref_hyp"),
        "mer": _error_rate(df, "mer"),
    }

    # Per audio_type WER + MER
    for audio_type, subset in df.groupby("audio_type"):
        scores.setdefault(audio_type, {})
        scores[audio_type]["wer"] = _error_rate(subset, "ref_hyp")
        scores[audio_type]["mer"] = _error_rate(subset, "mer")

    # Grid / passage: correlation + del/ins/sub F1 (and precision/recall when detailed)
    for audio_type in GRID_PASSAGE_TYPES:
        sub = df[df["audio_type"] == audio_type].copy()
        scores.setdefault(audio_type, {})

        pair = sub[["C_can_hyp", "C_can_ref"]].dropna()
        scores[audio_type]["corr"] = (
            pair["C_can_hyp"].corr(pair["C_can_ref"]) if len(pair) > 1 else math.nan
        )

        totals = sub[
            ["del_tp", "del_fp", "del_fn", "ins_tp", "ins_fp", "ins_fn",
             "sub_tp", "sub_fp", "sub_fn"]
        ].sum()
        for key in ["del", "ins", "sub"]:
            tp = totals[f"{key}_tp"]
            fp = totals[f"{key}_fp"]
            fn = totals[f"{key}_fn"]
            p = _precision(tp, fp)
            r = _recall(tp, fn)
            f = _f1(p, r)
            if detailed:
                scores[audio_type][f"{key}_precision"] = p * 100
                scores[audio_type][f"{key}_recall"] = r * 100
            scores[audio_type][f"{key}_f1"] = f * 100

    # Isolated: accuracy (and precision/recall/F1 when detailed)
    for audio_type in ISOLATED_TYPES:
        sub = df[df["audio_type"] == audio_type].copy()
        scores.setdefault(audio_type, {})
        sub["true_label"] = (sub["C_can_ref"] == 0).astype(int)
        sub["pred_label"] = (sub["C_can_hyp"] == 0).astype(int)
        tp = int(((sub["true_label"] == 1) & (sub["pred_label"] == 1)).sum())
        tn = int(((sub["true_label"] == 0) & (sub["pred_label"] == 0)).sum())
        fp = int(((sub["true_label"] == 0) & (sub["pred_label"] == 1)).sum())
        fn = int(((sub["true_label"] == 1) & (sub["pred_label"] == 0)).sum())
        p = _precision(tp, fp)
        r = _recall(tp, fn)
        f = _f1(p, r)
        if detailed:
            scores[audio_type]["precision"] = p * 100
            scores[audio_type]["recall"] = r * 100
            scores[audio_type]["f1"] = f * 100
        scores[audio_type]["accuracy"] = _accuracy(tp, tn, fp, fn) * 100

    return scores


# --- CSV loading ---

def parse_args() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", default=CSV_DEFAULT)
    args, _ = parser.parse_known_args()
    return args.csv


def fetch_csv_if_needed(url: str, dest: str) -> None:
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        st.error(f"Failed to download CSV from {url}: {e}")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


csv_cli_default = parse_args()
env_path = os.environ.get("EGRA_CSV_PATH")
env_url = os.environ.get("EGRA_CSV_URL")
if env_path:
    suggested_csv = env_path
elif env_url:
    suggested_csv = os.environ.get("EGRA_CSV_LOCAL", "egra_eval_detailed.csv")
    if not os.path.exists(suggested_csv):
        fetch_csv_if_needed(env_url, suggested_csv)
else:
    suggested_csv = csv_cli_default

csv_path = st.text_input("CSV path", suggested_csv)
st.session_state["csv_path"] = csv_path
df = load_data(csv_path)

# --- Sidebar filters ---
st.sidebar.header("Filters")

region_filter = st.sidebar.multiselect(
    "Region", sorted(df["region"].dropna().unique()), default=None
)
gender_filter = st.sidebar.multiselect(
    "Gender", sorted(df["gender"].dropna().unique()), default=None
)
grade_filter = st.sidebar.multiselect(
    "Grade", sorted(df["child_grade"].dropna().unique()), default=None
)
age_options = ["All"] + [
    str(v) for v in sorted(df["child_age"].dropna().unique().tolist())
]
age_filter = st.sidebar.selectbox("Child age", age_options, index=0)

st.sidebar.divider()
detailed = st.sidebar.toggle("Detailed metrics (precision / recall)", value=False)
show_scatter = st.sidebar.toggle("Show scatter plots", value=False)

# --- Apply filters ---
filtered = df.copy()
if region_filter:
    filtered = filtered[filtered["region"].isin(region_filter)]
if gender_filter:
    filtered = filtered[filtered["gender"].isin(gender_filter)]
if grade_filter:
    filtered = filtered[filtered["child_grade"].isin(grade_filter)]
if age_filter != "All":
    filtered = filtered[filtered["child_age"] == int(age_filter)]

_m1, _m2 = st.columns(2)
_m1.metric("Rows after filters", f"{len(filtered):,}")
_m2.metric("Unique children", f"{filtered['learner_id'].nunique():,}")

if filtered.empty:
    st.warning("No rows match the current filters.")
    st.stop()

# --- Compute summary ---
scores_dict = aggregate_scores(filtered, detailed=detailed)

# --- Display summary ---
st.subheader("Evaluation Summary")

pairs = list(zip(SUMMARY_KEYS[::2], SUMMARY_KEYS[1::2]))
for left_key, right_key in pairs:
    col_left, col_right = st.columns(2)
    for col, key in ((col_left, left_key), (col_right, right_key)):
        with col:
            subdict = scores_dict.get(key, {})
            n_items = len(filtered) if key == "global" else int((filtered["audio_type"] == key).sum())
            st.markdown(f"**{key.upper()}** (No. items: {n_items:,})")
            if subdict:
                rows = []
                for m, v in subdict.items():
                    if m == "corr":
                        rows.append({"metric": m, "value": f"{v:.4f}"})
                    else:
                        rows.append({"metric": m, "value": f"{v:.2f}%"})
                st.dataframe(
                    pd.DataFrame(rows).set_index("metric"),
                    use_container_width=True,
                )
            else:
                st.info("No data for this task type in the current selection.")

# --- Scatter plots ---
if show_scatter:
    st.subheader("Scatter plots: C_can_hyp vs C_can_ref")

    def _build_scatter_fig(audio_type, corr, x, y):
        fig, ax = plt.subplots()
        ax.scatter(x, y, alpha=0.6)
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            ax.plot(np.sort(x), m * np.sort(x) + b, color="tab:green")
        ax.set_xlabel("C_can_ref")
        ax.set_ylabel("C_can_hyp")
        corr_str = f"{corr:.3f}" if not np.isnan(corr) else "n/a"
        ax.set_title(f"{audio_type}  (r = {corr_str})")
        return fig

    scatter_data = []
    for audio_type in GRID_PASSAGE_TYPES:
        subset = (
            filtered[filtered["audio_type"] == audio_type][["C_can_ref", "C_can_hyp"]]
            .dropna()
        )
        if subset.empty:
            continue
        corr = scores_dict.get(audio_type, {}).get("corr", float("nan"))
        scatter_data.append((audio_type, corr, subset["C_can_ref"].values, subset["C_can_hyp"].values))

    scatter_cols = st.columns(2)
    for col_idx, (audio_type, corr, x, y) in enumerate(scatter_data):
        fig = _build_scatter_fig(audio_type, corr, x, y)
        with scatter_cols[col_idx % 2]:
            st.pyplot(fig)
        plt.close(fig)

    if scatter_data:
        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            for audio_type, corr, x, y in scatter_data:
                fig = _build_scatter_fig(audio_type, corr, x, y)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        st.download_button(
            label="Download scatter plots (PDF)",
            data=pdf_buf.getvalue(),
            file_name="scatter_plots.pdf",
            mime="application/pdf",
        )
