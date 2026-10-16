"""Turn the deep-dive JSONL into summary tables and figures (see deepdive.py for what is measured)."""
from __future__ import annotations

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
