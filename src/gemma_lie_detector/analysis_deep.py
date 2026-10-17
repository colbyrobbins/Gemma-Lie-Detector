"""Turn the deep-dive JSONL into summary tables and figures (see deepdive.py for what is measured)."""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from .analysis import bootstrap_ci, load_results

# Display names, in the order they appear in tables and figures.
METHOD_LABELS = {"delete": "occlusion: delete", "replace": "occlusion: replace",
                 "ig_pad": "integrated gradients (pad baseline)", "ig_zero": "integrated gradients (zero baseline)"}


def load_deep(path):
    """Load a deep-dive results file, dropping reviews that could not be tested."""
    df = load_results(path)
    return df[~df.skipped.astype(bool)].reset_index(drop=True) if len(df) else df


def _methods(df):
    return [m for m in METHOD_LABELS if f"attr_{m}" in df]


def _row(name, series):
    mean, lo, hi = bootstrap_ci(series)
    return (name, mean, lo, hi, int(pd.Series(series).notna().sum()))


def summarize_deep(df):
    """Headline table: agreement with Gemma's citations, behavioural tests, flip sets, IG sanity checks."""
    rows = []
    for m in _methods(df):
        label = METHOD_LABELS[m]
        rows.append(_row(f"{label}: precision@k vs Gemma's citations", df[f"prec_{m}"]))
        rows.append(_row(f"{label}: AUROC (cited outranks uncited)", df[f"auroc_{m}"]))
    for m in _methods(df):
        rows.append(_row(f"top-m deletion, log-odds drop: {METHOD_LABELS[m]}", df[f"topm_mdrop_{m}"]))
    rows.append(_row("top-m deletion, log-odds drop: Gemma's cited words", df.ref_cited_mdrop))
    rows.append(_row("top-m deletion, log-odds drop: random words", df.ref_rand_mdrop))
    for m in _methods(df):
        rows.append(_row(f"top-m deletion flips the decision: {METHOD_LABELS[m]}", df[f"topm_flip_{m}"].astype(float)))
    rows.append(_row("top-m deletion flips the decision: Gemma's cited words", df.ref_cited_flip.astype(float)))

    flippable = df.flip_min_size.notna()
    rows.append(_row("a flip set (size <= 3) exists in the candidate pool", flippable.astype(float)))
    if flippable.any():
        f = df[flippable]
        rows.append(_row("minimal flip-set size", f.flip_min_size.astype(float)))
        rows.append(_row("minimal flip sets found (redundancy: more = more redundant evidence)", f.flip_n_sets.astype(float)))
        rows.append(_row("citations cover a minimal flip set exactly", f.flip_cited_exact.astype(float)))
        rows.append(_row("share of the best flip set that Gemma cited", f.flip_cited_overlap.astype(float)))
    for b in ("pad", "zero"):
        if f"ig_err_{b}" in df:
            rows.append(_row(f"IG completeness error ({b} baseline, relative; lower is better)", df[f"ig_err_{b}"]))
            rows.append(_row(f"IG completeness error under 10% ({b} baseline)", (df[f"ig_err_{b}"] < 0.1).astype(float)))
    return pd.DataFrame(rows, columns=["metric", "mean", "ci_lo", "ci_hi", "n"])


def method_agreement(df):
    """Mean Spearman rank correlation between every pair of attribution methods (symmetric matrix)."""
    ms = _methods(df)
    mat = pd.DataFrame(np.eye(len(ms)), index=ms, columns=ms)
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            col = f"corr_{a}__{b}"
            if col in df:
                mat.loc[a, b] = mat.loc[b, a] = df[col].mean()
    return mat.rename(index=METHOD_LABELS, columns=METHOD_LABELS)


def disagreement_cases(df, limit=10):
    """Reviews where deleting Gemma's cited words flips the decision but deleting a method's top-m does not.

    These are the cases worth reading by hand: either occlusion/IG miss something real (interactions,
    redundancy) or the citations lean on words the model does not truly rely on.
    """
    out = []
    for m in _methods(df):
        miss = df[df.ref_cited_flip.astype(bool) & ~df[f"topm_flip_{m}"].astype(bool)]
        out.append(pd.DataFrame({"method": METHOD_LABELS[m], "id": miss.id,
                                 "cited_mdrop": miss.ref_cited_mdrop, "method_topm_mdrop": miss[f"topm_mdrop_{m}"]}))
    return pd.concat(out).sort_values("cited_mdrop", ascending=False).head(limit).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Combining occlusion and integrated gradients (IG)
#
# Neither method is ground truth, so a claim is stronger when it holds under both. The functions
# below keep IG honest by using only reviews whose completeness error is small, and report
# review-level statistics (words within one review are not independent of each other).
# ---------------------------------------------------------------------------------------------

def reliable_ig(df, baseline="zero", max_err=0.1):
    """Reviews whose IG scores pass the completeness check (relative error below `max_err`)."""
    return df[df[f"ig_err_{baseline}"] < max_err].reset_index(drop=True)


def _num(series):
    """Series of True/False/None/numbers as floats, with missing values as NaN."""
    return pd.Series([np.nan if v is None else float(v) for v in series], index=series.index)


def exact_sign_test(n_a_only, n_b_only):
    """Two-sided exact p-value that two methods flip the decision equally often (McNemar's exact test).

    Only reviews where exactly one of the two flips the decision matter: under "no difference"
    each of those is equally likely to favour either method.
    """
    n = n_a_only + n_b_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(n_a_only, n_b_only) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def behaviour_comparison(df, ig_baseline="zero", ig_max_err=0.1):
    """Paired comparison of Gemma's cited words against each method's top-m words, on the same deletion test.

    One row per comparison. `drop_diff` is (method's log-odds drop) - (cited words' drop), so a positive
    value means the method's words hurt the decision more; it comes with a bootstrap interval, which is
    more sensitive than counting flips. The flip columns count reviews where only the cited words flip
    the decision (`only_cited`) or only the method's words do (`only_method`), with an exact p-value.
    IG rows use only reviews that pass the completeness check, so their `n` is smaller.
    """
    rows = []
    for m in _methods(df):
        d = df
        if m.startswith("ig_"):
            d = reliable_ig(df, m[len("ig_"):], ig_max_err)
        cited_f, method_f = _num(d.ref_cited_flip), _num(d[f"topm_flip_{m}"])
        ok = cited_f.notna() & method_f.notna()
        only_c = int(((cited_f == 1) & (method_f == 0) & ok).sum())
        only_m = int(((cited_f == 0) & (method_f == 1) & ok).sum())
        diff = _num(d[f"topm_mdrop_{m}"]) - _num(d.ref_cited_mdrop)
        mean, lo, hi = bootstrap_ci(diff)
        rows.append(dict(method=METHOD_LABELS[m], n=int(ok.sum()),
                         cited_flip_rate=float(cited_f[ok].mean()), method_flip_rate=float(method_f[ok].mean()),
                         only_cited=only_c, only_method=only_m, flip_p=exact_sign_test(only_c, only_m),
                         drop_diff=mean, drop_diff_lo=lo, drop_diff_hi=hi))
    diff = _num(df.ref_cited_mdrop) - _num(df.ref_rand_mdrop)    # cited words against random words
    mean, lo, hi = bootstrap_ci(diff)
    rows.append(dict(method="(cited words minus random words)", n=int(diff.notna().sum()),
                     drop_diff=mean, drop_diff_lo=lo, drop_diff_hi=hi))
    return pd.DataFrame(rows)


def flip_patterns(df, baseline="zero", max_err=0.1):
    """How often each combination of {cited, occlusion, IG} top-m deletions flips the decision.

    The interesting rows are the ones where the three disagree: e.g. "cited and IG flip, occlusion
    does not" points to occlusion missing redundancy or interactions rather than a bad citation.
    """
    d = reliable_ig(df, baseline, max_err)
    flags = pd.DataFrame({"cited": _num(d.ref_cited_flip), "occlusion": _num(d.topm_flip_delete),
                          "ig": _num(d[f"topm_flip_ig_{baseline}"])}).dropna().astype(bool)
    out = flags.value_counts().rename("reviews").reset_index().sort_values("reviews", ascending=False)
    out["share"] = out.reviews / max(len(flags), 1)
    return out.reset_index(drop=True)


def cited_word_ranks(df, baseline="zero", top_k=3, max_err=0.1):
    """One row per cited word, with its rank under occlusion and under IG (1 = most supportive).

    A word is *high* under a method if its score is positive and it ranks in that method's top `top_k`.
    Ties rank together, so a zero score is never high. `group` says which methods agree it matters.
    """
    rows = []
    for _, r in reliable_ig(df, baseline, max_err).iterrows():
        occ, ig = np.asarray(r["attr_delete"], float), np.asarray(r[f"attr_ig_{baseline}"], float)
        for i in r["cited_idx"]:
            occ_rank, ig_rank = 1 + int((occ > occ[i]).sum()), 1 + int((ig > ig[i]).sum())
            occ_high, ig_high = occ[i] > 0 and occ_rank <= top_k, ig[i] > 0 and ig_rank <= top_k
            group = ("high under both" if occ_high and ig_high else "occlusion only" if occ_high
                     else "IG only" if ig_high else "neither")
            rows.append(dict(id=r["id"], word_idx=i, occ_score=occ[i], ig_score=ig[i],
                             occ_rank=occ_rank, ig_rank=ig_rank, group=group))
    return pd.DataFrame(rows, columns=["id", "word_idx", "occ_score", "ig_score", "occ_rank", "ig_rank", "group"])


def triangulation_table(words):
    """Share of each review's cited words in each agreement group, averaged over reviews (bootstrap CI)."""
    order = ["high under both", "occlusion only", "IG only", "neither"]
    if words.empty:
        return pd.DataFrame(columns=["group", "mean share of a review's cited words", "ci_lo", "ci_hi", "reviews"])
    share = (words.assign(n=1).pivot_table(index="id", columns="group", values="n", aggfunc="sum", fill_value=0)
             .reindex(columns=order, fill_value=0))
    share = share.div(share.sum(axis=1), axis=0)
    rows = [(g, *bootstrap_ci(share[g]), len(share)) for g in order]
    return pd.DataFrame(rows, columns=["group", "mean share of a review's cited words", "ci_lo", "ci_hi", "reviews"])


def make_deep_figures(df, outdir="results"):
    """Write deletion_curves.png (margin vs words deleted) and method_agreement.png."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    ks = int(min(len(c) for c in df.curve_random if len(c)))
    x = np.arange(1, ks + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in _methods(df):
        curves = np.array([c[:ks] for c in df[f"curve_{m}"] if len(c) >= ks])
        ax.plot(x, curves.mean(0), marker="o", label=METHOD_LABELS[m])
    rand = np.array([c[:ks] for c in df.curve_random if len(c) >= ks])
    ax.plot(x, rand.mean(0), marker="o", linestyle="--", color="gray", label="random order")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("words deleted (most important first)")
    ax.set_ylabel("mean margin of the original decision (log-odds)")
    ax.set_title("Which ranking hurts the decision fastest?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/deletion_curves.png", dpi=160)
    plt.close(fig)

    mat = method_agreement(df)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(mat)), mat.columns, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(mat)), mat.index, fontsize=7)
    for i in range(len(mat)):
        for j in range(len(mat)):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Mean rank correlation between methods")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{outdir}/method_agreement.png", dpi=160)
    plt.close(fig)
