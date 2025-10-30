from __future__ import annotations

import math
from dataclasses import dataclass
from jiwer import compute_measures
from egra_eval.normalize.textnorm import normalize


@dataclass
class Counts:
    S: int
    D: int
    I: int
    C: int
    N: int  # tokens in TRUTH after normalization

    @property
    def WER(self) -> float:
        return ((self.S + self.D + self.I) / self.N) * 100.0 if self.N else math.nan

    @property
    def ACC(self) -> float:
        return self.C / self.N if self.N else math.nan

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
        return (2 * p * r) / (p + r) if (not math.isnan(p) and not math.isnan(r) and (p + r) > 0) else math.nan


def score(truth: str, hyp: str) -> Counts:
    t = normalize(truth or "")
    h = normalize(hyp or "")

    if not t.strip():
        return Counts(S=0, D=0, I=0, C=0, N=0)

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
