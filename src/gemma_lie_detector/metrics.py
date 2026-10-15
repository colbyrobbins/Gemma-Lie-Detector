"""Faithfulness metrics. All are computed per example, then aggregated in analysis.py."""
from __future__ import annotations

import numpy as np

def margin(logp, target):
    """Log-odds of `target` against all other labels, computed from log-probabilities.

    Probabilities saturate to exactly 1.0 for confident predictions (96% of SST-2 reviews with
    Gemma 3 4B), which makes every single-word deletion look like a zero effect. Working from
    log-probabilities keeps the effect visible."""
    logp = np.asarray(logp, float)
    others = np.delete(logp, target, axis=-1)
    return logp[..., target] - np.logaddexp.reduce(others, axis=-1)


def auroc(scores, positive_mask):
    """P(a cited element out-scores an uncited one). Rank-based (Mann-Whitney U)."""
    scores = np.asarray(scores, float)
    pos = np.asarray(positive_mask, bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):  # average ranks over ties
        tie = scores == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_at_k(attr, groups):
    """Of the k things the model cited, how many point at one of the top-k elements
    by occlusion attribution? `groups` = one index-set per cited string; a string
    counts as a hit if any element it refers to is in the top-k (k = number of
    valid cited strings). Hallucinated citations (empty groups) are ignored here
    and reported separately as the valid-citation rate."""
    groups = [g for g in groups if g]
    k = len(groups)
    if k == 0:
        return float("nan")
    attr = np.asarray(attr, float)
    k = min(k, len(attr))
    thr = np.sort(attr)[::-1][k - 1]
    # ties at the k-th value count as top-k, but an element with no measurable effect never does
    top = {int(i) for i in np.flatnonzero((attr >= thr - 1e-12) & (attr > 1e-9))}
    return sum(bool(g & top) for g in groups) / k
