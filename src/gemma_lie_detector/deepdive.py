"""Deep dive: is the occlusion reference trustworthy, and do other methods agree with it?

For every review that already has a core-experiment record (so Gemma's citations are reused, not
regenerated) this compares four ways of scoring which words drive the decision:

  delete    delete one word at a time (the core experiment's reference)
  replace   swap one word at a time for a placeholder
  ig_pad    integrated gradients, padding-embedding baseline
  ig_zero   integrated gradients, all-zeros baseline

and judges them, along with Gemma's own citations, with behavioural tests:
  - top-m test:   delete each method's top-m words (m = number of cited words) and see how far
                  the margin falls, and whether the decision flips
  - flip set:     the smallest group of words whose deletion flips the decision, and whether
                  Gemma's citations overlap it
  - curves:       margin as the top 1..k words of each method are deleted, against random order

One JSON line is written per review, and finished reviews are skipped on a re-run.
"""
from __future__ import annotations

import json
import zlib
from itertools import combinations

import numpy as np

from .core import match_groups
from .experiments import _run
from .metrics import auroc, precision_at_k, spearman
from .robustness import build_pool, deletion_curve, minimal_flip_set, occlusion_replace, score_sets


def _order(attr):
    """Word indices from most to least supportive of the decision (stable for ties)."""
    return np.argsort(-np.asarray(attr, float), kind="stable").tolist()


def analyze_deep(ex, task, scorer, attributor, core_rec, k_curve=6, n_random=3, flip_size=3, seed=0):
    """Compare attribution methods on one review. `core_rec` is its record from the core experiment.

    `attributor` provides `.baselines` and `.attribute(task, ex, target, baseline)`; pass None to skip
    integrated gradients. Returns one JSON-serialisable record; reviews that cannot be tested
    (no valid citations, or every word cited) come back as `{"id": ..., "skipped": True}`.
    """
    n = len(ex.elements)
    cited = core_rec.get("cited_idx") or []
    if not core_rec.get("parse_ok") or not 0 < len(cited) < n:
        return {"id": ex.id, "skipped": True}

    pred = task.labels.index(core_rec["pred"])
    base_margin = core_rec["base_margin"]
    m = len(cited)
    groups = match_groups(ex, core_rec["cited_raw"])
    cited_mask = np.isin(np.arange(n), cited)
    rng = np.random.default_rng(zlib.crc32(f"{seed}-{ex.id}-deep".encode()))

    # 1. Word scores from each method (all "margin drop" style: positive = supports the decision).
    attrs = {"delete": np.asarray(core_rec["attr_logodds"], float),
             "replace": occlusion_replace(scorer, task, ex, pred, base_margin)}
    ig_err = {}
    if attributor is not None:
        for b in attributor.baselines:
            attrs[f"ig_{b}"], ig_err[b] = attributor.attribute(task, ex, pred, baseline=b)

    rec = dict(id=ex.id, skipped=False, pred=core_rec["pred"], base_margin=base_margin, n_elements=n,
               cited_idx=cited, cited_raw=core_rec["cited_raw"],
               ref_cited_mdrop=core_rec.get("del_cited_mdrop"), ref_cited_flip=core_rec.get("del_cited_flip"),
               ref_rand_mdrop=core_rec.get("del_rand_mdrop"))

    # 2. How well does each method agree with Gemma's citations, and with the other methods?
    for name, a in attrs.items():
        rec[f"attr_{name}"] = a.tolist()
        rec[f"prec_{name}"] = precision_at_k(a, groups)
        rec[f"auroc_{name}"] = auroc(a, cited_mask)
    for b, err in ig_err.items():
        rec[f"ig_err_{b}"] = err
    for a_name, b_name in combinations(attrs, 2):
        rec[f"corr_{a_name}__{b_name}"] = spearman(attrs[a_name], attrs[b_name])

    # 3. Judge every method with the same behavioural test: delete its top-m words, m = len(cited).
    for name, a in attrs.items():
        margins, flipped = score_sets(scorer, task, ex, [set(_order(a)[:m])], pred)
        rec[f"topm_mdrop_{name}"] = float(base_margin - margins[0])
        rec[f"topm_flip_{name}"] = bool(flipped[0])
        rec[f"curve_{name}"] = deletion_curve(scorer, task, ex, pred, _order(a), k_curve)
    random_curves = [deletion_curve(scorer, task, ex, pred, rng.permutation(n).tolist(), k_curve)
                     for _ in range(n_random)]
    rec["curve_random"] = np.mean(random_curves, axis=0).tolist() if random_curves[0] else []

    # 4. Smallest group of words whose deletion flips the decision, and do the citations overlap it?
    rankings = [_order(attrs["delete"])] + ([_order(attrs["ig_pad"])] if "ig_pad" in attrs else [])
    pool = build_pool(cited, rankings)
    flip = minimal_flip_set(scorer, task, ex, pred, pool, max_size=flip_size)
    rec["flip_pool"] = pool
    if flip is None:
        rec.update(flip_min_size=None, flip_sets=[], flip_n_sets=0,
                   flip_cited_overlap=None, flip_cited_exact=None)
    else:
        overlaps = [len(set(s) & set(cited)) / len(s) for s in flip["sets"]]
        rec.update(flip_min_size=flip["size"], flip_sets=flip["sets"], flip_n_sets=flip["n_sets"],
                   flip_cited_overlap=max(overlaps),
                   flip_cited_exact=any(set(s) <= set(cited) for s in flip["sets"]))
    return rec


def run_deep(examples, task, scorer, attributor, core_path, out_path, **kw):
    """Run `analyze_deep` over `examples`, reusing citations from the core results file at `core_path`.

    Appends one line per review to `out_path` and skips reviews already there, so it can be
    stopped and restarted. Reviews missing from the core file are skipped without being recorded,
    so they are picked up automatically once the core run reaches them.
    """
    with open(core_path, encoding="utf-8") as f:
        core = {r["id"]: r for r in map(json.loads, filter(str.strip, f))}
    ready = [ex for ex in examples if ex.id in core]

    def analyze(ex):
        return analyze_deep(ex, task, scorer, attributor, core[ex.id], **kw)

    _run(analyze, ready, out_path)
