"""
Phonological metrics computation for EGRA evaluation.

Computes True Positives (TP), False Positives (FP), False Negatives (FN),
Precision, Recall, and F1-score for phonological processes:
- Substitutions
- Deletions  
- Insertions

These metrics compare REF (what child said) vs CAN (canonical) as ground truth
against HYP (ASR output) vs CAN as predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import dp_align


@dataclass
class PhonologicalMetrics:
    """Phonological metrics for a single error type (substitutions, deletions, or insertions)."""
    tp: int  # True Positives: errors present in both REF and HYP
    fp: int  # False Positives: errors in HYP but not in REF
    fn: int  # False Negatives: errors in REF but not in HYP
    
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else math.nan
    
    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else math.nan
    
    @property
    def f1(self) -> float:
        """F1-score = 2 * (Precision * Recall) / (Precision + Recall)"""
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return math.nan
        return 2 * (p * r) / (p + r)


def _get_alignment_errors(canonical_text: str, other_text: str) -> List[Tuple[int, str, str]]:
    """
    Get alignment between canonical and other text, returning list of errors.
    
    Args:
        canonical_text: The canonical/reference text
        other_text: The text to compare (REF or HYP)
    
    Returns:
        List of (canonical_position, error_type, token_info) tuples
        - canonical_position: position in canonical sequence (0-indexed)
        - error_type: 's' (substitution), 'd' (deletion), or 'i' (insertion)
        - token_info: the token(s) involved (for matching)
        For substitutions: (can_pos, 's', (can_token, other_token))
        For deletions: (can_pos, 'd', can_token)
        For insertions: (insertion_idx, 'i', other_token) where insertion_idx is position in alignment
    """
    if not canonical_text:
        return []
    if not other_text:
        # All canonical tokens are deletions
        can_tokens = str(canonical_text).split()
        return [(i, 'd', token) for i, token in enumerate(can_tokens)]
    
    can_tokens = str(canonical_text).split()
    other_tokens = str(other_text).split()
    
    # Get alignment
    _, alignment = dp_align.dp_align(can_tokens, other_tokens, output_align=True)
    
    errors = []
    can_idx = 0
    align_idx = 0
    
    for test_token, ref_token, tag in alignment:
        if tag == 'c':
            # Correct match
            can_idx += 1
            align_idx += 1
        elif tag == 's':
            # Substitution
            errors.append((can_idx, 's', (ref_token, test_token)))
            can_idx += 1
            align_idx += 1
        elif tag == 'd':
            # Deletion
            errors.append((can_idx, 'd', ref_token))
            can_idx += 1
            # align_idx stays same (no test token consumed)
        elif tag == 'i':
            # Insertion - use alignment index as position identifier
            errors.append((align_idx, 'i', test_token))
            align_idx += 1
            # can_idx stays same (no canonical token consumed)
    
    return errors


def compute_phonological_metrics(
    canonical_text: str,
    reference_text: str,
    hypothesis_text: str,
) -> dict[str, PhonologicalMetrics]:
    """
    Compute phonological metrics comparing REF vs CAN (ground truth) 
    against HYP vs CAN (predictions).
    
    Args:
        canonical_text: What the child should have said (CAN)
        reference_text: What the child actually said (REF) - ground truth
        hypothesis_text: What ASR predicted (HYP) - predictions
    
    Returns:
        Dictionary with keys 'substitutions', 'deletions', 'insertions'
        Each value is a PhonologicalMetrics object
    """
    # Handle empty/missing texts
    if not canonical_text:
        canonical_text = ""
    if not reference_text:
        reference_text = ""
    if not hypothesis_text:
        hypothesis_text = ""
    
    # Get errors for REF vs CAN (ground truth)
    ref_errors = _get_alignment_errors(canonical_text, reference_text)
    
    # Get errors for HYP vs CAN (predictions)
    hyp_errors = _get_alignment_errors(canonical_text, hypothesis_text)
    
    # Convert errors to sets for comparison: (position, error_type)
    ref_error_set = {(pos, et) for pos, et, _ in ref_errors}
    hyp_error_set = {(pos, et) for pos, et, _ in hyp_errors}
    
    # Compute TP, FP, FN for each error type
    results = {}
    
    for error_type in ['s', 'd', 'i']:
        # Filter errors by type
        ref_errors_type = {(pos, et) for pos, et in ref_error_set if et == error_type}
        hyp_errors_type = {(pos, et) for pos, et in hyp_error_set if et == error_type}
        
        # True Positives: errors present in both REF and HYP at same position
        tp = len(ref_errors_type & hyp_errors_type)
        
        # False Positives: errors in HYP but not in REF
        fp = len(hyp_errors_type - ref_errors_type)
        
        # False Negatives: errors in REF but not in HYP
        fn = len(ref_errors_type - hyp_error_set)
        
        # Map error type to full name
        error_name = {
            's': 'substitutions',
            'd': 'deletions',
            'i': 'insertions'
        }[error_type]
        
        results[error_name] = PhonologicalMetrics(tp=tp, fp=fp, fn=fn)
    
    return results


def compute_phonological_metrics_row(row) -> dict:
    """
    Compute phonological metrics for a single row (dataframe row).
    
    Args:
        row: DataFrame row with 'canonical_text'/'CAN', 'reference_text'/'REF', 'hypothesis_text'/'HYP'
    
    Returns:
        Dictionary with metrics for substitutions, deletions, insertions
        Keys: S_TP, S_FP, S_FN, S_Precision, S_Recall, S_F1,
              D_TP, D_FP, D_FN, D_Precision, D_Recall, D_F1,
              I_TP, I_FP, I_FN, I_Precision, I_Recall, I_F1
    """
    # Handle both naming conventions: lowercase with _text suffix or uppercase
    can = row.get('canonical_text', row.get('CAN', ''))
    ref = row.get('reference_text', row.get('REF', ''))
    hyp = row.get('hypothesis_text', row.get('HYP', ''))
    
    metrics = compute_phonological_metrics(can, ref, hyp)
    
    result = {}
    
    for error_name in ['substitutions', 'deletions', 'insertions']:
        prefix = error_name[0].upper()  # 'S', 'D', 'I'
        m = metrics[error_name]
        
        result[f'{prefix}_TP'] = m.tp
        result[f'{prefix}_FP'] = m.fp
        result[f'{prefix}_FN'] = m.fn
        result[f'{prefix}_Precision'] = m.precision
        result[f'{prefix}_Recall'] = m.recall
        result[f'{prefix}_F1'] = m.f1
    
    return result

