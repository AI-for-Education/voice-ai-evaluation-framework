import logging
import math
import re
import pandas as pd
from pathlib import Path

from egra_eval2.eval_utils import (
    aggregate_error_rate,
    recall,
    precision,
    f1,
    accuracy,
)
from egra_eval2.metrics import (
    score_error_rate,
    score_mistake_error_rate,
    score_fine_sub_del_ins,
)


def evaluate_rows(df_egra: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-row scores and intermediate variables."""

    logger = logging.getLogger("egra_eval")
    logger.info("Calculating per-row scores")

    rows = []

    for _, r in df_egra.iterrows():
        can = r.get("canonical_text", "")
        ref = r.get("ref_text", "")
        hyp = r.get("hyp_text", "")

        audio_type = r.get("audio_type")
        if "iso_letter" in audio_type or "iso_random_letters" in audio_type:
            audio_type = "letters_isolated"
        elif "letters_grid" in audio_type or "letter_grid" in audio_type:
            audio_type = "letters_grid"
        elif "iso_non_word" in audio_type or "iso_random_nw" in audio_type:
            audio_type = "nonwords_isolated"
        elif (
            "non_words_grid" in audio_type
            or "non_word_grid" in audio_type
            or "nonword_grid" in audio_type
        ):
            audio_type = "nonwords_grid"
        elif "iso_syllable" in audio_type or "random_syl" in audio_type:
            audio_type = "syllables_isolated"
        elif "syllables_grid" in audio_type or "full_syllable" in audio_type:
            audio_type = "syllables_grid"
        elif "passage" in audio_type:
            audio_type = "passage_passage"
        else:
            logger.error(f"Invalid name: {audio_type}")
            assert False

        row = {
            "learner_id": r.get("learner_id"),
            "audio_file": r.get("audio_file"),
            "audio_type": audio_type,
            "CAN": can,
            "REF": ref,
            "HYP": hyp,
        }

        # # Temp
        # can = "ga u la e ka ha"
        # ref = "ga ga u e ka hi hi"
        # hyp = "ga u la i ka hi ho"
        # print("CAN", can)
        # print("REF", ref)
        # print("HYP", hyp)

        # Reference vs hypothesis (required for WER)
        score_ref_hyp = score_error_rate(ref, hyp)
        row.update(
            {
                "S_ref_hyp": score_ref_hyp.S,
                "D_ref_hyp": score_ref_hyp.D,
                "I_ref_hyp": score_ref_hyp.I,
                "C_ref_hyp": score_ref_hyp.C,
                "N_ref_hyp": score_ref_hyp.N,
            }
        )

        # Canonical vs hypothesis and canonical vs reference (required for correlation)
        score_can_ref = score_error_rate(can, ref)
        score_can_hyp = score_error_rate(can, hyp)
        row.update({"C_can_ref": score_can_ref.C, "C_can_hyp": score_can_hyp.C})

        # Scores for MER
        score_mer = score_mistake_error_rate(can, ref, hyp)
        row.update(
            {
                "S_mer": score_mer.S,
                "D_mer": score_mer.D,
                "I_mer": score_mer.I,
                "C_mer": score_mer.C,
                "N_mer": score_mer.N,
            }
        )

        # Score fine-grained errors
        score_sub_del_ins = score_fine_sub_del_ins(can, ref, hyp)
        row.update(score_sub_del_ins)

        rows.append(row)

    df_scores = pd.DataFrame(rows)

    return df_scores


def aggregate_row_scores(
    df_egra: pd.DataFrame,
    output_dir: Path,
    detailed: bool = False,
    scoring_units: str = "orthographic",
) -> dict:

    logger = logging.getLogger("egra_eval")
    logger.info("Aggregating per-row scores")
    scores_dict = {}

    if scoring_units not in {"orthographic", "phoneme"}:
        raise ValueError(f"Unsupported scoring units: {scoring_units}")
    error_metric = "per" if scoring_units == "phoneme" else "wer"

    # Global reference-to-hypothesis error rate and MER
    scores_dict["global"] = {}
    error_rate = aggregate_error_rate(df_egra, "ref_hyp")
    scores_dict["global"][error_metric] = error_rate
    mer = aggregate_error_rate(df_egra, "mer")
    scores_dict["global"]["mer"] = mer

    # Per-task reference-to-hypothesis error rate and MER
    for audio_type, df_subset in df_egra.groupby("audio_type"):
        logger.info(f"Calculating scores for {audio_type}")
        scores_dict[audio_type] = {}

        error_rate = aggregate_error_rate(df_subset, "ref_hyp")
        scores_dict[audio_type][error_metric] = error_rate

        # MER
        mer = aggregate_error_rate(df_subset, "mer")
        scores_dict[audio_type]["mer"] = mer

    # Passage and grid metrics
    for audio_type in [
        "passage_passage",
        "syllables_grid",
        "nonwords_grid",
        "letters_grid",
    ]:
        df_subset = df_egra.loc[df_egra["audio_type"] == audio_type].copy()

        # Correlation
        subset = df_subset[["C_can_hyp", "C_can_ref"]].dropna()
        corr = subset["C_can_hyp"].corr(subset["C_can_ref"])
        scores_dict[audio_type]["corr"] = corr

        # Scatter plot
        if detailed:
            import matplotlib.pyplot as plt
            import numpy as np

            x = subset["C_can_ref"]
            y = subset["C_can_hyp"]
            plt.figure()
            plt.scatter(x, y)
            m, b = np.polyfit(x, y, 1)
            plt.plot(x, m * x + b, color="tab:green")  # regression line
            plt.xlabel("C_can_ref")
            plt.ylabel("C_can_hyp")
            plt.title(f"C_can_hyp vs C_can_ref (r = {corr:.3f}) ({audio_type})")
            output_fn = output_dir / f"scatter_{audio_type}.png"
            logger.info(f"Writing: {output_fn}")
            plt.savefig(output_fn, dpi=300, bbox_inches="tight")
            plt.close()

        # Fine-grained
        cols = [
            "del_tp",
            "del_fp",
            "del_fn",
            "ins_tp",
            "ins_fp",
            "ins_fn",
            "sub_tp",
            "sub_fp",
            "sub_fn",
        ]
        totals = df_subset[cols].sum()
        results = {}
        for key in ["del", "ins", "sub"]:
            tp = totals[f"{key}_tp"]
            fp = totals[f"{key}_fp"]
            fn = totals[f"{key}_fn"]

            p = precision(tp, fp)
            r = recall(tp, fn)
            f = f1(p, r)

            if detailed:
                results[f"{key}_precision"] = p * 100
                results[f"{key}_recall"] = r * 100
            results[f"{key}_f1"] = f * 100
        scores_dict[audio_type].update(results)

    # Isolated metrics
    for audio_type in ["syllables_isolated", "nonwords_isolated", "letters_isolated"]:
        df_subset = df_egra.loc[df_egra["audio_type"] == audio_type].copy()

        # Define binary labels based on your rules:
        # Positive (1) = mistake = C_can_ref == 0
        # Predicted positive (1) = C_can_hyp == 0

        df_subset["true_label"] = (df_subset["C_can_ref"] == 0).astype(int)
        df_subset["pred_label"] = (df_subset["C_can_hyp"] == 0).astype(int)

        # Confusion matrix components
        tp = (
            (df_subset["true_label"] == 1) & (df_subset["pred_label"] == 1)
        ).sum()  # Both are mistakes
        tn = (
            (df_subset["true_label"] == 0) & (df_subset["pred_label"] == 0)
        ).sum()  # Both are correct
        fp = (
            (df_subset["true_label"] == 0) & (df_subset["pred_label"] == 1)
        ).sum()  # Predicted mistake, actually correct
        fn = (
            (df_subset["true_label"] == 1) & (df_subset["pred_label"] == 0)
        ).sum()  # Predicted correct, actually mistake

        p = precision(tp, fp)
        r = recall(tp, fn)
        f = f1(p, r)
        acc = accuracy(tp, tn, fp, fn)

        results = {}
        if detailed:
            results["precision"] = p * 100
            results["recall"] = r * 100
            results["f1"] = f * 100
        results["accuracy"] = acc * 100
        scores_dict[audio_type].update(results)

    return scores_dict
