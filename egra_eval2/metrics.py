from dataclasses import dataclass
from jiwer import compute_measures
from typing import List, Tuple

from egra_eval2.dp_align import dp_align
from egra_eval2.eval_utils import text_normalize


@dataclass
class Counts:
    S: int
    D: int
    I: int
    C: int
    N: int  # tokens in TRUTH after normalization

    @property
    def WER(self) -> float:
        if self.N:
            return ((self.S + self.D + self.I) / self.N) * 100.0
        return math.inf if self.I > 0 else math.nan

    @property
    def ACC(self) -> float:
        return self.C / self.N if self.N else math.nan

    @property
    def COR(self) -> int:
        """
        Correctness count derived from N-S-D (equivalent to hits on valid alignments).
        """
        return int(self.N - self.S - self.D) if self.N else 0

    @property
    def ACC_COR(self) -> float:
        """
        Accuracy derived from COR/N.
        """
        return (self.COR / self.N) if self.N else math.nan

    # Macro precision/recall/F1 at token level (REF = truth, HYP = system)
    @property
    def precision(self) -> float:
        denom = self.C + self.I
        return self.C / denom if denom > 0 else math.nan

    @property
    def recall(self) -> float:
        denom = self.C + self.D
        return self.C / denom if denom > 0 else math.nan

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (
            (2 * p * r) / (p + r)
            if (not math.isnan(p) and not math.isnan(r) and (p + r) > 0)
            else math.nan
        )


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


def get_csid_sequence(truth: str, hyp: str) -> List[str]:

    # This normalisation below is faulty because it removes apostrophes
    # def normalize_transcript(text: str) -> str:
    #     import re, string
    #     if text is None:
    #         return ""
    #     s = str(text).lower()
    #     s = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", s)
    #     s = re.sub(r"\s+", " ", s).strip()
    #     return s
    # can_norm2 = normalize_transcript(truth or "")
    # other_norm2 = normalize_transcript(hyp or "")

    can_norm = text_normalize(truth or "")
    other_norm = text_normalize(hyp or "")

    can_list = can_norm.split()
    other_list = other_norm.split()

    _, alignment = dp_align(can_list, other_list, output_align=True)
    return [item[2] for item in alignment]


def get_csid_sequence_with_pos(truth: str, hyp: str) -> List[Tuple[int, str, str]]:

    # This normalisation below is faulty because it removes apostrophes
    # def normalize_transcript(text: str) -> str:
    #     import re, string
    #     if text is None:
    #         return ""
    #     s = str(text).lower()
    #     s = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", s)
    #     s = re.sub(r"\s+", " ", s).strip()
    #     return s
    # can_norm2 = normalize_transcript(truth or "")
    # other_norm2 = normalize_transcript(hyp or "")

    can_norm = text_normalize(truth or "")
    other_norm = text_normalize(hyp or "")

    can_list = can_norm.split()
    other_list = other_norm.split()

    _, alignment = dp_align(can_list, other_list, output_align=True)
    # return [item[2] for item in alignment]


def score_mistake_error_rate(can: str, ref: str, hyp: str) -> Counts:

    ref_to_can = get_csid_sequence(can, ref)
    hyp_to_can = get_csid_sequence(can, hyp)

    mer_errors, mer_alignment = dp_align(ref_to_can, hyp_to_can, output_align=True)

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

    ref_to_can_errors, ref_to_can_alignment = dp_align(can, ref, output_align=True)
    hyp_to_can_errors, hyp_to_can_alignment = dp_align(can, hyp, output_align=True)

    ref_to_can_alignment_sequence = [i[-1] for i in ref_to_can_alignment]
    hyp_to_can_alignment_sequence = [i[-1] for i in hyp_to_can_alignment]

    mer_errors, mer_alignment = dp_align(
        ref_to_can_alignment_sequence, hyp_to_can_alignment_sequence, output_align=True
    )
    true_csid = [i[1] for i in mer_alignment]
    pred_csid = [i[0] for i in mer_alignment]

    classes = set(["d", "i", "c", "s", "-"])
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
        if cls == "-" or cls == "c":
            continue
        label = label_map.get(cls, cls)
        # for metric in ["tp", "tn", "fp", "fn"]:
        for metric in ["tp", "fp", "fn"]:
            flat_counts[f"{label}_{metric}"] = counts[cls][metric]

    return flat_counts
