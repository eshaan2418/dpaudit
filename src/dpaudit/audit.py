"""Empirical privacy audits.

1. Loss-threshold membership inference (Yeom et al. 2018): score = −loss. Reports ROC AUC and
   TPR at low FPR on balanced member / non-member sets.
2. One-run auditing (Steinke, Nasr, Jagielski 2023): m canaries are each included in training
   independently with probability 1/2. After training, the auditor ranks canaries by score,
   guesses "IN" for the top k₊ and "OUT" for the bottom k₋, and abstains on the rest. Under
   ε-DP, correct guesses are stochastically dominated by Binomial(r, e^ε/(1+e^ε)) for r
   guesses, so observing v correct gives the lower bound

       ε_lb = max{ ε : P[Binomial(r, e^ε/(1+e^ε)) ≥ v] ≤ β }   (confidence 1 − β).

   This is the δ = 0 form of their Theorem 3. For δ = 1e-5 and r ≤ 1,000 guesses the δ
   correction to the guess-accuracy bound is negligible next to the binomial
   confidence interval, but strictly the reported value is a bound for the δ = 0 analysis.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom


def roc_auc(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    """P(score_in > score_out) + 0.5·P(tie), via ranks."""
    s = np.concatenate([scores_in, scores_out])
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n1, n0 = len(scores_in), len(scores_out)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tpr_at_fpr(scores_in: np.ndarray, scores_out: np.ndarray, fpr: float) -> float:
    thr = np.quantile(scores_out, 1 - fpr)
    return float((scores_in > thr).mean())


def eps_lower_bound(r: int, v: int, beta: float = 0.05) -> float:
    """Largest ε such that observing ≥ v correct of r guesses has probability ≤ β under ε-DP."""
    if r == 0 or v <= r / 2:
        return 0.0

    def tail(eps: float) -> float:
        p = math.exp(eps) / (1 + math.exp(eps))
        return float(binom.sf(v - 1, r, p))  # P[X >= v]

    lo, hi = 0.0, 20.0
    if tail(lo) > beta:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if tail(mid) <= beta:
            lo = mid
        else:
            hi = mid
    return lo


def one_run_audit(scores: np.ndarray, included: np.ndarray, k_plus: int, k_minus: int,
                  beta: float = 0.05) -> dict:
    order = np.argsort(-scores, kind="mergesort")
    guess_in = order[:k_plus]
    guess_out = order[len(order) - k_minus :] if k_minus else np.array([], dtype=int)
    v = int(included[guess_in].sum() + (~included[guess_out]).sum())
    r = k_plus + k_minus
    return {"guesses": r, "correct": v, "accuracy": v / r if r else float("nan"),
            "eps_lower": eps_lower_bound(r, v, beta)}


def best_one_run_audit(scores: np.ndarray, included: np.ndarray, ks: tuple[int, ...],
                       beta: float = 0.05) -> dict:
    """Try several abstention levels; Bonferroni-correct β across them."""
    b = beta / len(ks)
    res = [one_run_audit(scores, included, k, k, b) | {"k": k} for k in ks]
    return max(res, key=lambda d: d["eps_lower"])
