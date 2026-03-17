import argparse
import os
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EGRA Evaluation Framework", layout="wide")
st.title("EGRA Evaluation Framework")

# Default CSV used locally; when deployed we prefer egra_eval_detailed.csv in /app
CSV_DEFAULT = "./egra_eval_detailed.csv"

GROUPABLE_COLUMNS = [
    "learner_id",
    "audio_category",
    "audio_subcategory",
    "audio_type",
    "gender",
    "child_grade",
    "child_age",
    "region",
    "council",
    "ward",
    "Kiswahili_lang1",
    "Kigogo_lang2",
    "Kirangi_lang3",
    "Kihaya_lang4",
    "Runyambo_lang5",
    "Kihangaza_lang6",
    "English_lang7",
]

COUNT_COLUMNS = [
    "S_can_ref",
    "D_can_ref",
    "I_can_ref",
    "C_can_ref (EGRA_COR)",
    "N_can_ref",
    "S_can_hyp",
    "D_can_hyp",
    "I_can_hyp",
    "C_can_hyp (ASR_EGRA_COR)",
    "N_can_hyp",
    "S_ref_hyp",
    "D_ref_hyp",
    "I_ref_hyp",
    "C_ref_hyp",
    "N_ref_hyp",
]

ROW_META = [
    "learner_id",
    "audio_type",
    "gender",
    "child_grade",
    "child_age",
    "region",
    "council",
    "ward",
    "Kiswahili_lang1",
    "Kigogo_lang2",
    "Kirangi_lang3",
    "Kihaya_lang4",
    "Runyambo_lang5",
    "Kihangaza_lang6",
    "English_lang7",
]



def audio_categories(audio_type: str) -> tuple[str, str]:
    if not isinstance(audio_type, str):
        return ("other", "other")
    at = audio_type.lower()
    if at == "full_letter_grid":
        return ("letters", "grid")
    if at.startswith("iso_letter_"):
        return ("letters", "isolated")
    if at.startswith("iso_random_letters"):
        return ("letters", "random")
    if at.startswith("full_syllable"):
        return ("syllables", "grid")
    if at.startswith("iso_syllable"):
        return ("syllables", "isolated")
    if at.startswith("rand_syllable") or at.startswith("random_syl"):
        return ("syllables", "random")
    if at.startswith("full_nonword"):
        return ("nonwords", "grid")
    if at.startswith("iso_non_word"):
        return ("nonwords", "isolated")
    if at.startswith("iso_random_nw"):
        return ("nonwords", "random")
    if at.startswith("passage_num"):
        return ("passage", "passage")
    return ("other", "other")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    mainCat, subCat = zip(*df["audio_type"].apply(audio_categories))
    df["audio_category"] = mainCat
    df["audio_subcategory"] = subCat
    return df


def division(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where(den != 0, num / den * 100.0, np.nan)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EGRA-COR"] = df["C_can_ref (EGRA_COR)"]
    df["EGRA-ACC"] = division(df["C_can_ref (EGRA_COR)"], df["N_can_ref"])
    df["ASR-EGRA-COR"] = df["C_can_hyp (ASR_EGRA_COR)"]
    df["ASR-EGRA-ACC"] = division(df["C_can_hyp (ASR_EGRA_COR)"], df["N_can_hyp"])
    df["MAE_EGRA_COR"] = (df["C_can_ref (EGRA_COR)"] - df["C_can_hyp (ASR_EGRA_COR)"]).abs()
    df["ASR_WER"] = division(df["S_ref_hyp"] + df["D_ref_hyp"] + df["I_ref_hyp"], df["N_ref_hyp"])
    df["WER_can_ref"] = division(df["S_can_ref"] + df["D_can_ref"] + df["I_can_ref"], df["N_can_ref"])
    df["ACC_can_ref (EGRA_ACC)"] = division(df["C_can_ref (EGRA_COR)"], df["N_can_ref"])
    df["WER_can_hyp"] = division(df["S_can_hyp"] + df["D_can_hyp"] + df["I_can_hyp"], df["N_can_hyp"])
    df["ACC_can_hyp (ASR_EGRA_ACC)"] = division(df["C_can_hyp (ASR_EGRA_COR)"], df["N_can_hyp"])
    df["WER_ref_hyp"] = division(df["S_ref_hyp"] + df["D_ref_hyp"] + df["I_ref_hyp"], df["N_ref_hyp"])
    return df


def aggregate_counts(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if not by:
        summed = df[COUNT_COLUMNS].sum()
        result = pd.DataFrame([summed])
    else:
        result = df.groupby(by, dropna=False)[COUNT_COLUMNS].sum().reset_index()
    result = compute_metrics(result)
    return result


def parse_args() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", default=CSV_DEFAULT, help="Path to egra_eval_detailed.csv")
    args, _ = parser.parse_known_args()
    return args.csv


def fetch_csv_if_needed(url: str, dest: str) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        st.error(f"Failed to download CSV from {url}: {e}")
        return False


csv_cli_default = parse_args()
# Allow environment overrides: EGRA_CSV_PATH or EGRA_CSV_URL
env_path = os.environ.get("EGRA_CSV_PATH")
env_url = os.environ.get("EGRA_CSV_URL")
if env_path:
    suggested_csv = env_path
elif env_url:
    # download to local filename unless overridden
    suggested_csv = os.environ.get("EGRA_CSV_LOCAL", "egra_eval_detailed.csv")
    if not os.path.exists(suggested_csv):
        fetch_csv_if_needed(env_url, suggested_csv)
else:
    suggested_csv = csv_cli_default

# Allow the user to edit the path in the UI (or supply via CLI/env)
csv_path = st.text_input("CSV path", suggested_csv)
df = load_data(csv_path)
df = compute_metrics(df)

st.sidebar.header("Filters")
category_filter = st.sidebar.multiselect("Audio category", sorted(df["audio_category"].unique()), default=None)
gender_filter = st.sidebar.multiselect("Gender", sorted(df["gender"].dropna().unique()), default=None)
region_filter = st.sidebar.multiselect("Region", sorted(df["region"].dropna().unique()), default=None)
grade_filter = st.sidebar.multiselect("Grade", sorted(df["child_grade"].dropna().unique()), default=None)

filtered = df.copy()
if category_filter:
    filtered = filtered[filtered["audio_category"].isin(category_filter)]
if gender_filter:
    filtered = filtered[filtered["gender"].isin(gender_filter)]
if region_filter:
    filtered = filtered[filtered["region"].isin(region_filter)]
if grade_filter:
    filtered = filtered[filtered["child_grade"].isin(grade_filter)]

st.metric("Rows after filters", f"{len(filtered):,}")

group_cols = st.multiselect("Group / sort columns", GROUPABLE_COLUMNS, default=["audio_category", "gender", "region"])
sort_cols = st.multiselect("Sort output by ", group_cols
    + [
        "WER_can_ref",
        "WER_can_hyp",
        "WER_ref_hyp",
        "EGRA-COR",
        "EGRA-ACC",
        "ASR-EGRA-COR",
        "ASR-EGRA-ACC",
        "MAE_EGRA_COR",
        "ASR_WER",
    ],
    default=group_cols)

ascending = st.radio("Sort direction", options=["Ascending", "Descending"], index=0, horizontal=True)

agg = aggregate_counts(filtered, group_cols)
if sort_cols:
    agg = agg.sort_values(by=sort_cols, ascending=(ascending == "Ascending"))

st.subheader("Aggregated metrics")
st.dataframe(agg, width='stretch')

## Summary table for CAN counts (S/D/I/C/N) and MAE between S_can_ref and S_can_hyp
st.subheader("CAN vs ASR counts summary (filtered & grouped)")
if len(filtered) == 0:
    st.info("No rows in current selection to summarize.")
else:
    # We want the same grouping as the aggregated metrics table (group_cols).
    # Use the already-computed `agg` which contains summed COUNT_COLUMNS per group.
    # Build a summary DataFrame containing S/D/I/C/N for CAN and HYP plus MAE(S_can_ref, S_can_hyp)
    # Select the count columns for CAN and HYP from `agg`.
    can_cols = [
        "S_can_ref",
        "D_can_ref",
        "I_can_ref",
        "C_can_ref (EGRA_COR)",
        "N_can_ref",
    ]
    hyp_cols = [
        "S_can_hyp",
        "D_can_hyp",
        "I_can_hyp",
        "C_can_hyp (ASR_EGRA_COR)",
        "N_can_hyp",
    ]

    # If grouping is empty, `agg` is a single-row DataFrame; keep group_cols in the result for consistency
    mae_names = [
        "MAE_S_can_ref_vs_S_can_hyp",
        "MAE_D_can_ref_vs_D_can_hyp",
        "MAE_I_can_ref_vs_I_can_hyp",
        "MAE_C_can_ref_vs_C_can_hyp",
        "MAE_N_can_ref_vs_N_can_hyp",
    ]

    if not group_cols:
        summary = agg[can_cols + hyp_cols].copy()
        # compute MAE over the filtered rows (mean absolute per-row difference) for all metrics
        mae_S = float(np.mean(np.abs(filtered["S_can_ref"].fillna(0) - filtered["S_can_hyp"].fillna(0))))
        mae_D = float(np.mean(np.abs(filtered["D_can_ref"].fillna(0) - filtered["D_can_hyp"].fillna(0))))
        mae_I = float(np.mean(np.abs(filtered["I_can_ref"].fillna(0) - filtered["I_can_hyp"].fillna(0))))
        mae_C = float(np.mean(np.abs(filtered["C_can_ref (EGRA_COR)"].fillna(0) - filtered["C_can_hyp (ASR_EGRA_COR)"].fillna(0))))
        mae_N = float(np.mean(np.abs(filtered["N_can_ref"].fillna(0) - filtered["N_can_hyp"].fillna(0))))
        summary["MAE_S_can_ref_vs_S_can_hyp"] = mae_S
        summary["MAE_D_can_ref_vs_D_can_hyp"] = mae_D
        summary["MAE_I_can_ref_vs_I_can_hyp"] = mae_I
        summary["MAE_C_can_ref_vs_C_can_hyp"] = mae_C
        summary["MAE_N_can_ref_vs_N_can_hyp"] = mae_N
    else:
        # Keep grouping columns + counts
        display_cols = group_cols + can_cols + hyp_cols
        # agg already has group columns at front when group_cols provided to aggregate_counts
        summary = agg[display_cols].copy()
        # compute MAE per group based on the original filtered rows for all metrics
        def group_mae_fn(g):
            return pd.Series({
                "MAE_S_can_ref_vs_S_can_hyp": float(np.mean(np.abs(g["S_can_ref"].fillna(0) - g["S_can_hyp"].fillna(0)))) ,
                "MAE_D_can_ref_vs_D_can_hyp": float(np.mean(np.abs(g["D_can_ref"].fillna(0) - g["D_can_hyp"].fillna(0)))) ,
                "MAE_I_can_ref_vs_I_can_hyp": float(np.mean(np.abs(g["I_can_ref"].fillna(0) - g["I_can_hyp"].fillna(0)))) ,
                "MAE_C_can_ref_vs_C_can_hyp": float(np.mean(np.abs(g["C_can_ref (EGRA_COR)"].fillna(0) - g["C_can_hyp (ASR_EGRA_COR)"].fillna(0)))) ,
                "MAE_N_can_ref_vs_N_can_hyp": float(np.mean(np.abs(g["N_can_ref"].fillna(0) - g["N_can_hyp"].fillna(0)))) ,
            })

        group_mae = filtered.groupby(group_cols).apply(group_mae_fn).reset_index()
        # Merge MAE into summary on grouping columns
        summary = summary.merge(group_mae, on=group_cols, how="left")

    # Show the summary table; allow sorting by the group columns order
    st.dataframe(summary, width='stretch')

st.subheader("Filtered rows (raw)")
st.dataframe(filtered[ROW_META + COUNT_COLUMNS], width='stretch')
