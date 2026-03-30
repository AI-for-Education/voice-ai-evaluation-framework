from __future__ import annotations

import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import dp_align
from egra_eval.pipeline.eval_utils import compute_fine_grained_metrics
from egra_eval.report.summarize import (
    _annotate_audio_categories,
    summary_for_pair,
    summary_per_speaker,
    summary_per_speaker_macro,
    summary_per_speaker_subcategory,
    summary_phonological_by_category,
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def write_detailed_csv(df: pd.DataFrame, path: str, logger: logging.Logger) -> None:
    rename_map = {
        "learner_id": "learner_id",
        "audio_type": "audio_type",
        "audio_file": "audio_file",
        "CAN": "CAN",
        "REF": "REF",
        "HYP": "HYP",
        "WER_can_ref": "WER_can_ref",
        "ACC_can_ref": "ACC_can_ref (EGRA_ACC)",
        "S_can_ref": "S_can_ref",
        "D_can_ref": "D_can_ref",
        "I_can_ref": "I_can_ref",
        "C_can_ref": "C_can_ref (EGRA_COR)",
        "N_can_ref": "N_can_ref",
        "WER_can_hyp": "WER_can_hyp",
        "ACC_can_hyp": "ACC_can_hyp (ASR_EGRA_ACC)",
        "S_can_hyp": "S_can_hyp",
        "D_can_hyp": "D_can_hyp",
        "I_can_hyp": "I_can_hyp",
        "C_can_hyp": "C_can_hyp (ASR_EGRA_COR)",
        "N_can_hyp": "N_can_hyp",
        "WER_ref_hyp": "WER_ref_hyp",
        "ACC_ref_hyp": "ACC_ref_hyp",
        "S_ref_hyp": "S_ref_hyp",
        "D_ref_hyp": "D_ref_hyp",
        "I_ref_hyp": "I_ref_hyp",
        "C_ref_hyp": "C_ref_hyp",
        "N_ref_hyp": "N_ref_hyp",
        "EGRA_COR": "EGRA-COR",
        "EGRA_ACC": "EGRA-ACC",
        "ASR_EGRA_COR": "ASR-EGRA-COR",
        "ASR_EGRA_ACC": "ASR-EGRA-ACC",
        "MAE_EGRA_COR": "MAE_EGRA_COR",
        "ASR_WER": "ASR_WER",
        "has_hyp": "has_hyp",
        "gender": "gender",
        "child_grade": "child_grade",
        "child_age": "child_age",
        "region": "region",
        "council": "council",
        "ward": "ward",
        "Kiswahili_lang1": "Kiswahili_lang1",
        "Kigogo_lang2": "Kigogo_lang2",
        "Kirangi_lang3": "Kirangi_lang3",
        "Kihaya_lang4": "Kihaya_lang4",
        "Runyambo_lang5": "Runyambo_lang5",
        "Kihangaza_lang6": "Kihangaza_lang6",
        "English_lang7": "English_lang7",
        "ACC_can_ref_norm_for_mae": "ACC_can_ref_norm_for_mae",
        "ACC_can_hyp_norm_for_mae": "ACC_can_hyp_norm_for_mae",
        "MAE_EGRA_ACC": "MAE_EGRA_ACC",
        "Bias_Baseline_MAE": "Bias_Baseline_MAE",
        "S_Precision": "S_Precision",
        "S_Recall": "S_Recall",
        "S_F1": "S_F1",
        "I_Precision": "I_Precision",
        "I_Recall": "I_Recall",
        "I_F1": "I_F1",
        "D_Precision": "D_Precision",
        "D_Recall": "D_Recall",
        "D_F1": "D_F1",
        "Mistakes_Precision": "Mistakes_Precision",
        "Mistakes_Recall": "Mistakes_Recall",
        "Mistakes_F1": "Mistakes_F1",
        "Mistake_Error_Rate (MER)": "Mistake_Error_Rate (MER)",
        "S_TP": "S_TP",
        "S_FP": "S_FP",
        "S_FN": "S_FN",
        "D_TP": "D_TP",
        "D_FP": "D_FP",
        "D_FN": "D_FN",
        "I_TP": "I_TP",
        "I_FP": "I_FP",
        "I_FN": "I_FN",
    }

    initial_rows = len(df)
    base_keys = ["learner_id", "audio_type", "audio_file"]
    text_keys = ["CAN", "REF", "HYP", "canonical_text", "reference_text", "hypothesis_text"]
    subset_keys = [c for c in base_keys + text_keys if c in df.columns]

    if subset_keys:
        df = df.drop_duplicates(subset=subset_keys, keep="first")
        removed = initial_rows - len(df)
        if removed > 0:
            logger.info("Removed %d duplicate row(s) based on keys %s before writing detailed CSV.", removed, subset_keys)
    else:
        df = df.drop_duplicates(keep="first")
        removed = initial_rows - len(df)
        if removed > 0:
            logger.info("Removed %d exact duplicate row(s) before writing detailed CSV.", removed)

    df.rename(columns=rename_map).to_csv(path, index=False)
    logger.info("Wrote detailed results -> %s (rows: %d)", path, len(df))


def write_summary_csvs(df: pd.DataFrame, summary_dirs: Dict[str, Path], logger: logging.Logger) -> List[Path]:
    written: List[Path] = []

    def _write_one(pair: str, directory: Path) -> None:
        per_speaker = summary_per_speaker(df, pair)
        global_row = summary_for_pair(df, pair, by=None)
        global_row.insert(0, "learner_id", "__GLOBAL__")

        combined = (
            global_row
            if per_speaker.empty
            else pd.concat([global_row[per_speaker.columns], per_speaker], ignore_index=True)
        )
        per_speaker_path = directory / "egra_eval_summary_per_speaker_global.csv"
        combined.to_csv(per_speaker_path, index=False)
        logger.info("[%s] Wrote per-speaker summary -> %s", pair, per_speaker_path)
        written.append(per_speaker_path)

        macro_path = directory / "egra_eval_summary_per_speaker_macro.csv"
        summary_per_speaker_macro(df, pair).to_csv(macro_path, index=False)
        logger.info("[%s] Wrote per-speaker x macro-category summary -> %s", pair, macro_path)
        written.append(macro_path)

        subcat_path = directory / "egra_eval_summary_per_speaker_subcat.csv"
        summary_per_speaker_subcategory(df, pair).to_csv(subcat_path, index=False)
        logger.info("[%s] Wrote per-speaker x subcategory summary -> %s", pair, subcat_path)
        written.append(subcat_path)

    for pair, directory in summary_dirs.items():
        _write_one(pair, directory)

    return written


def write_text_summary(df: pd.DataFrame, out_csv: str, logger: logging.Logger) -> Path:
    summary_path = Path(out_csv).parent / "egra_eval_summary.txt"

    metrics = {}
    can_ref = summary_for_pair(df, "can_ref")
    if not can_ref.empty:
        metrics["EGRA-COR"] = can_ref["C_can_ref"].iloc[0]
        metrics["EGRA-ACC"] = can_ref["ACC_can_ref"].iloc[0]

    can_hyp = summary_for_pair(df, "can_hyp")
    if not can_hyp.empty:
        metrics["ASR-EGRA-COR"] = can_hyp["C_can_hyp"].iloc[0]
        metrics["ASR-EGRA-ACC"] = can_hyp["ACC_can_hyp"].iloc[0]

    ref_hyp = summary_for_pair(df, "ref_hyp")
    if not ref_hyp.empty:
        metrics["ASR_WER"] = ref_hyp["WER_ref_hyp"].iloc[0]

    r_correlation_global = "Not Available"
    if "C_can_ref" in df.columns and "C_can_hyp" in df.columns:
        common = df[["C_can_ref", "C_can_hyp"]].dropna()
        if len(common) >= 2:
            arr_x = common["C_can_ref"].to_numpy()
            arr_y = common["C_can_hyp"].to_numpy()
            if arr_x.std() > 0 and arr_y.std() > 0:
                r_correlation_global = float(np.corrcoef(arr_x, arr_y)[0, 1])
                try:
                    if plt is None:
                        raise RuntimeError("matplotlib is unavailable")
                    scatter_dir = summary_path.parent / "scatter_plots"
                    scatter_dir.mkdir(parents=True, exist_ok=True)
                    plt.figure(figsize=(6, 4))
                    plt.scatter(arr_x, arr_y, alpha=0.6, s=30)
                    mn = min(arr_x.min(), arr_y.min())
                    mx = max(arr_x.max(), arr_y.max())
                    plt.plot([mn, mx], [mn, mx], color="gray", linestyle="--", linewidth=0.8)
                    plt.xlabel("EGRA-COR (C_can_ref)")
                    plt.ylabel("ASR-EGRA-COR (C_can_hyp)")
                    plt.title(f"Global: EGRA-COR vs ASR-EGRA-COR (r={r_correlation_global:.3f})")
                    out_name = f"scatter_GLOBAL_C_can_ref_vs_C_can_hyp_r{r_correlation_global:.3f}.png".replace(" ", "_")
                    plt.tight_layout()
                    plt.savefig(scatter_dir / out_name)
                    plt.close()
                except Exception as exc:
                    logger.warning("Could not write global scatter plot: %s", exc)
    metrics["r_correlation"] = r_correlation_global

    if "MAE_EGRA_ACC" in df.columns:
        metrics["MAE_EGRA_ACC"] = df["MAE_EGRA_ACC"].dropna().mean() * 100.0
    if "Bias_Baseline_MAE" in df.columns:
        metrics["Bias_Baseline_MAE"] = df["Bias_Baseline_MAE"].dropna().mean()
    if "MER" in df.columns:
        metrics["MER"] = df["MER"].dropna().mean()
    if "MAE_EGRA_COR" in df.columns:
        metrics["MAE_EGRA_COR"] = df["MAE_EGRA_COR"].dropna().mean()
    else:
        metrics["MAE_EGRA_COR"] = float("nan")
    if "Mistakes_F1" in df.columns:
        metrics["Mistakes_F1"] = df["Mistakes_F1"].dropna().mean()
    else:
        metrics["Mistakes_F1"] = float("nan")

    phono_by_cat = summary_phonological_by_category(df)
    task_map = {
        "T1": "passage_passage",
        "T2": "syllables_grid",
        "T3": "syllables_isolated",
        "T4": "nonwords_grid",
        "T5": "nonwords_isolated",
        "T6": "letters_grid",
        "T7": "letters_isolated",
    }
    group_a = ["T1", "T2", "T4", "T6"]
    group_b = ["T3", "T5", "T7"]

    def calc_prf1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        def _fmt4(val) -> str:
            if val is None:
                return "NaN"
            if isinstance(val, str) and val in ("N/A", "Not Available"):
                return val
            try:
                if pd.isna(val):
                    return "NaN"
            except Exception:
                pass
            try:
                return f"{float(val):.4f}"
            except Exception:
                return str(val)

        f.write("Metric\tValue\n")
        for key in [
            "EGRA-COR", "EGRA-ACC", "ASR-EGRA-COR", "ASR-EGRA-ACC",
            "MAE_EGRA_COR", "MAE_EGRA_ACC", "Bias_Baseline_MAE", "MER",
            "Mistakes_F1", "ASR_WER", "r_correlation",
        ]:
            val = metrics.get(key, float("nan"))
            if pd.isna(val):
                out_val = "NaN"
            elif isinstance(val, (int, np.integer)):
                out_val = f"{int(val)}"
            elif isinstance(val, (float, np.floating)):
                out_val = f"{val:.2f}"
            else:
                out_val = str(val)
            f.write(f"{key}\t{out_val}\n")
        f.write("\n=== MAPPED METRICS ===\n\n")

        for task_id, cat_name in task_map.items():
            f.write(f"--- Task {task_id} ({cat_name}) ---\n")
            cat_data = phono_by_cat.get(cat_name, {})
            df_cat = _annotate_audio_categories(df)
            try:
                macro, sub = cat_name.split("_", 1)
            except Exception:
                macro, sub = (cat_name, None)
            sel = (
                df_cat[(df_cat["macro_category"] == macro) & (df_cat["sub_category"] == sub)]
                if sub is not None else df_cat[df_cat["macro_category"] == macro]
            )

            if not sel.empty and "MER" in sel.columns:
                cat_mer = float(sel["MER"].dropna().mean())
            else:
                cat_mer = metrics.get("MER", float("nan"))

            m_tp = sum(cat_data.get(f"{k}_TP", 0) for k in ["S", "D", "I"])
            m_fp = sum(cat_data.get(f"{k}_FP", 0) for k in ["S", "D", "I"])
            m_fn = sum(cat_data.get(f"{k}_FN", 0) for k in ["S", "D", "I"])
            m_p, m_r, m_f1 = calc_prf1(m_tp, m_fp, m_fn)

            if task_id in group_a:
                wer_task = sel["WER_ref_hyp"].dropna().mean() if ("WER_ref_hyp" in sel.columns and not sel.empty) else metrics.get("ASR_WER", "N/A")
                f.write(f"wer_ref_hyp: {_fmt4(wer_task)}\n")

                r_val = "Not Available"
                if not sel.empty and "C_ref_hyp" in sel.columns and "C_can_ref" in sel.columns:
                    common = sel[["C_ref_hyp", "C_can_ref"]].dropna()
                    if len(common) >= 2:
                        arr_x = common["C_ref_hyp"].to_numpy()
                        arr_y = common["C_can_ref"].to_numpy()
                        if arr_x.std() > 0 and arr_y.std() > 0:
                            r_val = float(np.corrcoef(arr_x, arr_y)[0, 1])
                f.write(f"r_correlation: {_fmt4(r_val)}\n")

                try:
                    if plt is None:
                        raise RuntimeError("matplotlib is unavailable")
                    if isinstance(r_val, float):
                        common = sel[["C_ref_hyp", "C_can_ref"]].dropna()
                        if not common.empty:
                            xs = common["C_can_ref"].to_numpy()
                            ys = common["C_ref_hyp"].to_numpy()
                            scatter_dir = summary_path.parent / "scatter_plots"
                            scatter_dir.mkdir(parents=True, exist_ok=True)
                            plt.figure(figsize=(6, 4))
                            plt.scatter(xs, ys, alpha=0.6, s=20)
                            mn = min(xs.min(), ys.min())
                            mx = max(xs.max(), ys.max())
                            plt.plot([mn, mx], [mn, mx], color="gray", linestyle="--", linewidth=0.8)
                            plt.xlabel("Actual correct count (C_can_ref)")
                            plt.ylabel("Predicted correct count (C_ref_hyp)")
                            plt.title(f"{task_id} - {cat_name} - scatter C_can_ref vs C_ref_hyp (r={r_val:.3f})")
                            out_name = f"scatter_{task_id}_{cat_name}_C_can_ref_vs_C_ref_hyp_r{r_val:.3f}.png".replace(" ", "_")
                            plt.tight_layout()
                            plt.savefig(scatter_dir / out_name)
                            plt.close()
                except Exception as exc:
                    logger.warning("Could not write scatter plot for %s (%s): %s", task_id, cat_name, exc)

                mae_counts = (
                    float(sel["MAE_EGRA_COR"].dropna().mean())
                    if ("MAE_EGRA_COR" in sel.columns and not sel["MAE_EGRA_COR"].dropna().empty)
                    else metrics.get("MAE_EGRA_ACC", float("nan"))
                )
                f.write(f"MAE_correct_counts: {mae_counts:.2f}\n")
                f.write(f"MER: {cat_mer:.2f} (fraction)\n")

                for proc_code, proc_name in [("S", "subs"), ("D", "del"), ("I", "insert")]:
                    p = cat_data.get(f"{proc_code}_Precision", 0.0)
                    r = cat_data.get(f"{proc_code}_Recall", 0.0)
                    f1 = cat_data.get(f"{proc_code}_F1", 0.0)
                    f.write(f"{proc_name}_prec: {p:.4f}\n")
                    f.write(f"{proc_name}_r: {r:.4f}\n")
                    f.write(f"{proc_name}_f1: {f1:.4f}\n")

                f.write(f"mistakes_prec: {m_p:.4f}\n")
                f.write(f"mistakes_r: {m_r:.4f}\n")
                f.write(f"mistakes_f1: {m_f1:.4f}\n")

            if task_id in group_b:
                wer_task = (
                    sel["WER_ref_hyp"].dropna().mean()
                    if ("WER_ref_hyp" in sel.columns and not sel["WER_ref_hyp"].dropna().empty)
                    else metrics.get("ASR_WER", "N/A")
                )
                egra_acc_task = (
                    sel["ACC_can_ref"].dropna().mean()
                    if ("ACC_can_ref" in sel.columns and not sel["ACC_can_ref"].dropna().empty)
                    else metrics.get("EGRA-ACC", "N/A")
                )
                f.write(f"wer_ref_hyp: {_fmt4(wer_task)}\n")
                f.write(f"egra_acc: {_fmt4(egra_acc_task)}\n")
                f.write(f"corr_mistake_pred_prec: {m_p:.4f}\n")
                f.write(f"corr_mistake_pred_r: {m_r:.4f}\n")
                f.write(f"corr_mistake_pred_f1: {m_f1:.4f}\n")

                if not sel.empty and all(c in sel.columns for c in ["S_can_ref", "D_can_ref", "I_can_ref"]):
                    gt = ((sel.get("S_can_ref", 0).fillna(0) + sel.get("D_can_ref", 0).fillna(0) + sel.get("I_can_ref", 0).fillna(0)) > 0).astype(int)
                    y_true = gt.values
                    from sklearn.metrics import precision_recall_fscore_support as prfs

                    y_pred_no = np.zeros_like(y_true)
                    p_no, r_no, f1_no, _ = prfs(y_true, y_pred_no, labels=[1], zero_division=0)
                    y_pred_all = np.ones_like(y_true)
                    p_all, r_all, f1_all, _ = prfs(y_true, y_pred_all, labels=[1], zero_division=0)

                    f.write(f"baseline_no_mistakes_prec: {float(p_no[0]):.4f}\n")
                    f.write(f"baseline_no_mistakes_r: {float(r_no[0]):.4f}\n")
                    f.write(f"baseline_no_mistakes_f1: {float(f1_no[0]):.4f}\n")
                    f.write(f"baseline_all_mistakes_prec: {float(p_all[0]):.4f}\n")
                    f.write(f"baseline_all_mistakes_r: {float(r_all[0]):.4f}\n")
                    f.write(f"baseline_all_mistakes_f1: {float(f1_all[0]):.4f}\n")
                else:
                    f.write("baseline_no_mistakes_prec: Not Available\n")
                    f.write("baseline_no_mistakes_r: Not Available\n")
                    f.write("baseline_no_mistakes_f1: Not Available\n")
                    f.write("baseline_all_mistakes_prec: Not Available\n")
                    f.write("baseline_all_mistakes_r: Not Available\n")
                    f.write("baseline_all_mistakes_f1: Not Available\n")

            f.write("\n")

    logger.info("Wrote text summary -> %s", summary_path)
    return summary_path


def write_t1_example_walkthrough(df: pd.DataFrame, out_csv: str, logger: logging.Logger) -> Path:
    out_path = Path(out_csv).parent / "t1_metric_example.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_cat = _annotate_audio_categories(df)
    sel = df_cat[(df_cat["macro_category"] == "passage") & (df_cat["sub_category"] == "passage")]

    if sel.empty:
        out_path.write_text("No passage_passage rows were found; cannot build the T1 example.\n", encoding="utf-8")
        logger.info("No T1 rows found; wrote placeholder at %s", out_path)
        return out_path

    candidates = sel
    if "has_hyp" in candidates.columns:
        candidates = candidates[candidates["has_hyp"] == True]
    if "HYP" in candidates.columns:
        candidates = candidates[candidates["HYP"].notna() & (candidates["HYP"].astype(str) != "")]
    if candidates.empty:
        candidates = sel

    row = candidates.iloc[0]
    can = row.get("canonical_text", row.get("CAN", ""))
    ref = row.get("reference_text", row.get("REF", ""))
    hyp = row.get("hypothesis_text", row.get("HYP", ""))

    def normalize_transcript(text: str) -> str:
        if pd.isna(text) or text is None:
            return ""
        s = str(text).lower()
        s = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    can_norm = normalize_transcript(can)
    ref_norm = normalize_transcript(ref)
    hyp_norm = normalize_transcript(hyp)

    can_tokens = can_norm.split()
    ref_tokens = ref_norm.split()
    hyp_tokens = hyp_norm.split()

    _, align_can_ref = dp_align.dp_align(can_tokens, ref_tokens, output_align=True)
    _, align_can_hyp = dp_align.dp_align(can_tokens, hyp_tokens, output_align=True)

    seq_ref = [a[2] for a in align_can_ref]
    seq_hyp = [a[2] for a in align_can_hyp]
    mer_errors, mer_alignment = dp_align.dp_align(seq_ref, seq_hyp, output_align=True)

    fine_metrics = compute_fine_grained_metrics(row)
    phono_by_cat = summary_phonological_by_category(df)
    cat_data = phono_by_cat.get("passage_passage", {})

    r_val = "Not Available"
    sel_counts = sel[["C_ref_hyp", "C_can_ref"]].dropna() if all(c in sel.columns for c in ["C_ref_hyp", "C_can_ref"]) else pd.DataFrame()
    if not sel_counts.empty and sel_counts["C_ref_hyp"].std() > 0 and sel_counts["C_can_ref"].std() > 0:
        arr_x = sel_counts["C_ref_hyp"].to_numpy()
        arr_y = sel_counts["C_can_ref"].to_numpy()
        r_val = float(np.corrcoef(arr_x, arr_y)[0, 1])

    mae_counts = float(sel["MAE_EGRA_COR"].dropna().mean()) if ("MAE_EGRA_COR" in sel.columns and not sel["MAE_EGRA_COR"].dropna().empty) else float("nan")
    cat_mer = float(sel["MER"].dropna().mean()) if ("MER" in sel.columns and not sel["MER"].dropna().empty) else float("nan")

    m_tp = sum(cat_data.get(f"{k}_TP", 0) for k in ["S", "D", "I"])
    m_fp = sum(cat_data.get(f"{k}_FP", 0) for k in ["S", "D", "I"])
    m_fn = sum(cat_data.get(f"{k}_FN", 0) for k in ["S", "D", "I"])
    m_p = m_tp / (m_tp + m_fp) if (m_tp + m_fp) > 0 else 0.0
    m_r = m_tp / (m_tp + m_fn) if (m_tp + m_fn) > 0 else 0.0
    m_f1 = 2 * m_p * m_r / (m_p + m_r) if (m_p + m_r) > 0 else 0.0

    def _summarize_alignment(alignment, max_rows: int = 30) -> list[str]:
        lines = [f"Alignment (showing up to {max_rows} rows):"]
        for idx, (test_tok, ref_tok, tag) in enumerate(alignment):
            if idx >= max_rows:
                lines.append(f"... ({len(alignment) - max_rows} more rows omitted)")
                break
            lines.append(f"{idx + 1:03d}: CAN='{ref_tok}' | OTHER='{test_tok}' | tag={tag}")
        return lines

    lines: list[str] = []
    lines.append("T1 Metric Walkthrough (passage_passage)\n")
    lines.append("Selected example row (first available with a hypothesis):")
    lines.append(f"- learner_id: {row.get('learner_id', 'N/A')}")
    lines.append(f"- audio_file: {row.get('audio_file', 'N/A')}")
    lines.append(f"- audio_type: {row.get('audio_type', 'N/A')}")
    lines.append(f"- C_can_ref (actual correct count): {row.get('C_can_ref', 'N/A')}")
    lines.append(f"- C_ref_hyp (predicted correct count): {row.get('C_ref_hyp', 'N/A')}\n")
    lines.append("Step 1: Canonical vs Reference alignment (dp_align)")
    lines.append(f"- Canonical token count: {len(can_tokens)} | Reference token count: {len(ref_tokens)}")
    lines.append(f"- csid sequence length: {len(seq_ref)} | counts: {dict(Counter(seq_ref))}")
    lines.extend(_summarize_alignment(align_can_ref))
    lines.append("")
    lines.append("Step 2: Canonical vs Hypothesis alignment (dp_align)")
    lines.append(f"- Hypothesis token count: {len(hyp_tokens)}")
    lines.append(f"- csid sequence length: {len(seq_hyp)} | counts: {dict(Counter(seq_hyp))}")
    lines.extend(_summarize_alignment(align_can_hyp))
    lines.append("")
    lines.append("Step 3: Meta-alignment of mistake sequences (MER)")
    lines.append(f"- MER WER (sequence-level): {mer_errors.get_wer():.4f}")
    lines.append(f"- MER alignment length: {len(mer_alignment)}")
    lines.append("- MER alignment sample (seq_b_token | seq_a_token | align_type):")
    for idx, (pred_tok, true_tok, align_type) in enumerate(mer_alignment[:30]):
        lines.append(f"  {idx + 1:03d}: pred='{pred_tok}' | true='{true_tok}' | align={align_type}")
    if len(mer_alignment) > 30:
        lines.append(f"  ... ({len(mer_alignment) - 30} more rows omitted)")
    lines.append("")
    lines.append("Step 4: Row-level fine-grained metrics (from compute_fine_grained_metrics)")
    lines.append(f"- MER (row): {fine_metrics.get('MER', float('nan')):.4f}")
    lines.append(f"- Substitution P/R/F1: {fine_metrics.get('S_Precision', float('nan')):.4f} / {fine_metrics.get('S_Recall', float('nan')):.4f} / {fine_metrics.get('S_F1', float('nan')):.4f}")
    lines.append(f"- Insertion P/R/F1: {fine_metrics.get('I_Precision', float('nan')):.4f} / {fine_metrics.get('I_Recall', float('nan')):.4f} / {fine_metrics.get('I_F1', float('nan')):.4f}")
    lines.append(f"- Deletion P/R/F1: {fine_metrics.get('D_Precision', float('nan')):.4f} / {fine_metrics.get('D_Recall', float('nan')):.4f} / {fine_metrics.get('D_F1', float('nan')):.4f}")
    lines.append(f"- Mistakes (binary) P/R/F1: {fine_metrics.get('Mistakes_Precision', float('nan')):.4f} / {fine_metrics.get('Mistakes_Recall', float('nan')):.4f} / {fine_metrics.get('Mistakes_F1', float('nan')):.4f}")
    lines.append("")
    lines.append("Step 5: How the T1 summary values are formed")
    lines.append(f"- WER_ref_hyp (row): {row.get('WER_ref_hyp', float('nan'))}")
    lines.append(f"- Correlation r_correlation uses all T1 rows' (C_can_ref, C_ref_hyp) pairs; this row contributes ({row.get('C_can_ref', 'N/A')}, {row.get('C_ref_hyp', 'N/A')}) -> r_correlation = {r_val}")
    lines.append(f"- MAE_correct_counts for this row: {abs(row.get('C_can_ref', 0) - row.get('C_ref_hyp', 0)) if pd.notna(row.get('C_can_ref', np.nan)) and pd.notna(row.get('C_ref_hyp', np.nan)) else 'N/A'}; category mean = {mae_counts:.4f}")
    lines.append(f"- Category MER (mean of row MER): {cat_mer:.4f}")
    lines.append(f"- Category substitution P/R/F1: {cat_data.get('S_Precision', 0.0):.4f} / {cat_data.get('S_Recall', 0.0):.4f} / {cat_data.get('S_F1', 0.0):.4f}")
    lines.append(f"- Category deletion P/R/F1: {cat_data.get('D_Precision', 0.0):.4f} / {cat_data.get('D_Recall', 0.0):.4f} / {cat_data.get('D_F1', 0.0):.4f}")
    lines.append(f"- Category insertion P/R/F1: {cat_data.get('I_Precision', 0.0):.4f} / {cat_data.get('I_Recall', 0.0):.4f} / {cat_data.get('I_F1', 0.0):.4f}")
    lines.append(f"- Category mistakes P/R/F1 (aggregated S+D+I): {m_p:.4f} / {m_r:.4f} / {m_f1:.4f}")
    lines.append("")
    lines.append("Step 6: Text snippets (original and normalized)")
    lines.append(f"CAN (original): {can}")
    lines.append(f"CAN (normalized): {can_norm}")
    lines.append(f"REF (original): {ref}")
    lines.append(f"REF (normalized): {ref_norm}")
    lines.append(f"HYP (original): {hyp}")
    lines.append(f"HYP (normalized): {hyp_norm}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote T1 walkthrough -> %s", out_path)
    return out_path
