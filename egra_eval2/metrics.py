from dataclasses import dataclass

from jiwer import compute_measures

from egra_eval2.dp_align import dp_align
from egra_eval2.eval_utils import text_normalize


@dataclass
class Counts:
    S: int
    D: int
    I: int
    C: int
    N: int  # tokens in TRUTH after normalization


def score_error_rate(truth: str, hyp: str) -> Counts:
    t = text_normalize(truth or "")
    h = text_normalize(hyp or "")

    if not t.strip():
        # Match NeMo-style behavior for empty reference:
        # - empty hyp => undefined WER (NaN)
        # - non-empty hyp => insertion-only path (WER = inf)
        i_only = len(h.split()) if h.strip() else 0
        return Counts(S=0, D=0, I=i_only, C=0, N=0)

    res = compute_measures(t, h)
    # jiwer >=3.1 -> 'truth_len'; older -> 'truth_words'
    n_ref = res.get("truth_len", res.get("truth_words"))
    if n_ref is None:
        n_ref = res["hits"] + res["substitutions"] + res["deletions"]

    return Counts(
        S=res["substitutions"],
        D=res["deletions"],
        I=res["insertions"],
        C=res["hits"],
        N=int(n_ref),
    )


def get_csid_sequence(truth: str, hyp: str) -> list[str]:
    can_norm = text_normalize(truth or "")
    other_norm = text_normalize(hyp or "")

    can_list = can_norm.split()
    other_list = other_norm.split()

    _, alignment = dp_align(can_list, other_list, output_align=True)
    return [item[2] for item in alignment]


def score_mistake_error_rate(can: str, ref: str, hyp: str) -> Counts:

    ref_to_can = get_csid_sequence(can, ref)
    hyp_to_can = get_csid_sequence(can, hyp)

    mer_errors, _ = dp_align(ref_to_can, hyp_to_can, output_align=True)

    return Counts(
        S=mer_errors.n_sub,
        D=mer_errors.n_del,
        I=mer_errors.n_ins,
        C=mer_errors.n_match,
        N=mer_errors.n_total,
    )


def score_fine_sub_del_ins(can: str, ref: str, hyp: str) -> dict:

    can = text_normalize(can or "")
    ref = text_normalize(ref or "")
    hyp = text_normalize(hyp or "")

    can = can.split()
    ref = ref.split()
    hyp = hyp.split()

    _, ref_to_can_alignment = dp_align(can, ref, output_align=True)
    _, hyp_to_can_alignment = dp_align(can, hyp, output_align=True)

    ref_to_can_alignment_sequence = [i[-1] for i in ref_to_can_alignment]
    hyp_to_can_alignment_sequence = [i[-1] for i in hyp_to_can_alignment]

    mer_errors, mer_alignment = dp_align(
        ref_to_can_alignment_sequence, hyp_to_can_alignment_sequence, output_align=True
    )
    true_csid = [i[1] for i in mer_alignment]
    pred_csid = [i[0] for i in mer_alignment]

    classes = {"d", "i", "s"}
    counts = {cls: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for cls in classes}
    for true, pred in zip(true_csid, pred_csid):
        for cls in classes:
            true_pos = true == cls
            pred_pos = pred == cls

            if true_pos and pred_pos:
                counts[cls]["tp"] += 1
            elif not true_pos and not pred_pos:
                counts[cls]["tn"] += 1
            elif not true_pos and pred_pos:
                counts[cls]["fp"] += 1
            elif true_pos and not pred_pos:
                counts[cls]["fn"] += 1

    label_map = {"s": "sub", "i": "ins", "d": "del", "c": "cor"}
    flat_counts = {}
    for cls in sorted(classes):
        label = label_map.get(cls, cls)
        for metric in ["tp", "fp", "fn"]:
            flat_counts[f"{label}_{metric}"] = counts[cls][metric]

    return flat_counts
