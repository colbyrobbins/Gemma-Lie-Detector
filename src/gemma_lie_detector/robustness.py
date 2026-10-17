"""Robustness checks for the occlusion reference.

Single-word deletion is the project's reference for "what drives the decision", but it has
known blind spots: deleting a word can make the text ungrammatical, redundant words hide each
other, and several words can matter only together. The helpers here probe those blind spots:

  occlusion_replace   swap each word for a placeholder instead of deleting it
  minimal_flip_set    smallest group of words whose removal flips the decision
  deletion_curve      margin as words are deleted one after another, in a given order

All of them work on text examples and only need a scorer with `score(prompts)`.
"""
from __future__ import annotations

import itertools

import numpy as np

from .metrics import margin

PLACEHOLDER = "[removed]"


def render_replaced(ex, idx, placeholder=PLACEHOLDER) -> str:
    """Render a text example with the words at `idx` replaced (not deleted)."""
    if ex.modality != "text":
        raise ValueError("replacement is only defined for text examples")
    idx = set(idx)
    return " ".join(placeholder if i in idx else w for i, w in enumerate(ex.elements))


def score_sets(scorer, task, ex, sets, pred, placeholder=None):
    """Margin of the original decision `pred` after removing each set of words.

    Words are deleted unless `placeholder` is given, in which case they are replaced by it.
    Returns (margins, flipped) as arrays with one entry per set; `flipped` is True where the
    decision no longer wins.
    """
    if placeholder is None:
        render = ex.render
    else:
        def render(s):
            return render_replaced(ex, s, placeholder)
    lp, _ = scorer.score([task.prompt(render(s)) for s in sets])
    lp = np.asarray(lp, float)
    return margin(lp, pred), lp.argmax(1) != pred


def occlusion_replace(scorer, task, ex, pred, base_margin, placeholder=PLACEHOLDER):
    """Per-word margin drop when each word is replaced by `placeholder`.

    Same sign convention as deletion occlusion: positive = the word supports the decision.
    """
    n = len(ex.elements)
    margins, _ = score_sets(scorer, task, ex, [{i} for i in range(n)], pred, placeholder)
    return base_margin - margins


def build_pool(cited_idx, rankings, per_ranking=3, max_size=10):
    """Candidate words for the flip-set search: the cited words plus the top words of each ranking.

    The search is exhaustive over subsets, so the pool is capped to keep it affordable.
    """
    pool = list(dict.fromkeys(cited_idx))
    for ranking in rankings:
        for i in ranking[:per_ranking]:
            if i not in pool:
                pool.append(int(i))
    return sorted(pool[:max_size])


def minimal_flip_set(scorer, task, ex, pred, pool, max_size=3, max_report=10):
    """Smallest group of words (from `pool`) whose deletion flips the decision.

    Tries every subset of size 1, then 2, ... up to `max_size`, and stops at the first size where
    any subset flips. Returns None if nothing flips. Otherwise a dict with the minimal `size`,
    up to `max_report` flipping subsets of that size (`sets`, strongest first) and how many
    there were (`n_sets`). Several minimal sets is common: it means the evidence is redundant.
    """
    pool = list(pool)
    for size in range(1, max_size + 1):
        subsets = [set(c) for c in itertools.combinations(pool, size)]
        if not subsets:
            break
        margins, flipped = score_sets(scorer, task, ex, subsets, pred)
        if flipped.any():
            order = [i for i in np.argsort(margins, kind="stable") if flipped[i]]
            return {"size": size, "n_sets": len(order),
                    "sets": [sorted(subsets[i]) for i in order[:max_report]]}
    return None


def deletion_curve(scorer, task, ex, pred, order, k):
    """Margin of `pred` after deleting the first 1, 2, ... k words of `order` (best first).

    Never deletes every word. Returns a list of margins, one per prefix length.
    """
    k = min(k, len(ex.elements) - 1, len(order))
    if k < 1:
        return []
    margins, _ = score_sets(scorer, task, ex, [set(order[:j]) for j in range(1, k + 1)], pred)
    return margins.tolist()
