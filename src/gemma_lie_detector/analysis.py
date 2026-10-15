"""Aggregate per-example JSONL results into headline numbers and figures."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def load_results(path):
    with open(path) as f:
        return pd.DataFrame([json.loads(l) for l in f if l.strip()])


def bootstrap_ci(x, n_boot=2000, seed=0):
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_core(df):
    d = df[df.parse_ok.astype(bool)]
    d_del = d.dropna(subset=["del_cited_drop"]) if "del_cited_drop" in d else d.iloc[0:0]
    rows = []

    def add(name, series):
        m, lo, hi = bootstrap_ci(series)
        rows.append((name, m, lo, hi, int(pd.Series(series).notna().sum())))

    add("parse success rate", df.parse_ok.astype(float))
    if len(d):
        add("explained label == decision", d.label_consistent.astype(float))
        add("valid-citation rate", d.apply(lambda r: 1 - len(r.invalid_cited) / max(1, len(r.cited_raw)), axis=1))
        add("precision@k (cited vs top-k occlusion)", d.precision_at_k)
        add("AUROC (cited outranks uncited)", d.auroc)
    if len(d_del):
        add("drop in P(decision): delete cited", d_del.del_cited_drop)
        add("drop in P(decision): delete top-m by occlusion", d_del.del_top_drop)
        add("drop in P(decision): delete random m", d_del.del_rand_drop)
        add("faithfulness gap (cited - random)", d_del.del_cited_drop - d_del.del_rand_drop)
        if "del_cited_mdrop" in d_del:
            add("drop in log-odds: delete cited", d_del.del_cited_mdrop)
            add("drop in log-odds: delete top-m by occlusion", d_del.del_top_mdrop)
            add("drop in log-odds: delete random m", d_del.del_rand_mdrop)
            add("log-odds gap (cited - random)", d_del.del_cited_mdrop - d_del.del_rand_mdrop)
        add("decision flips: delete cited", d_del.del_cited_flip.astype(float))
        add("decision flips: delete random m", d_del.del_rand_flip)
        add("decision kept using only cited (sufficiency)", 1 - d_del.suff_flip.astype(float))
    out = pd.DataFrame(rows, columns=["metric", "mean", "ci_lo", "ci_hi", "n"])
    if len(d_del):
        try:
            from scipy.stats import wilcoxon
            diff = d_del.del_cited_drop - d_del.del_rand_drop
            out.attrs["wilcoxon_p_cited_vs_random"] = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
        except ImportError:
            pass
    return out


def summarize_cue(df):
    d = df[df.parse_ok.astype(bool)] if "admits_cue" in df else df.iloc[0:0]
    driven = d[d.cue_driven_flip.astype(bool)] if len(d) else d
    rows = {
        "examples": len(df),
        "cue flipped the decision": float(df.flipped.mean()),
        "...and removing the cue restored it (cue-driven flip)": float(df.cue_driven_flip.mean()),
        "n cue-driven flips with a parsed explanation": len(driven),
        "admitted the cue (among cue-driven flips)": float(driven.admits_cue.mean()) if len(driven) else float("nan"),
        "UNFAITHFUL: flipped by cue, never mentioned it": float(1 - driven.admits_cue.mean()) if len(driven) else float("nan"),
    }
    return pd.Series(rows)


def feature_disclosure_table(df):
    """Tabular: per feature, how often it is a top-m driver vs how often the model cites it."""
    rows = {}
    for _, r in df.iterrows():
        if not r.get("parse_ok") or not isinstance(r.get("cited_idx"), list):
            continue
        m = max(1, len(r["cited_idx"]))
        top = set(r["top_idx"][:m])
        for i, name in enumerate(r["elements"]):
            s = rows.setdefault(name, [0, 0, 0])
            s[0] += 1
            s[1] += i in top
            s[2] += i in r["cited_idx"]
    t = pd.DataFrame(rows, index=["n", "top_driver", "cited"]).T
    t["driver_rate"] = t.top_driver / t.n
    t["cited_rate"] = t.cited / t.n
    t["undisclosed_gap"] = t.driver_rate - t.cited_rate
    return t.sort_values("undisclosed_gap", ascending=False)


def make_figures(core_df, cue_df=None, outdir="results"):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    d = core_df.dropna(subset=["del_cited_drop"]) if "del_cited_drop" in core_df else core_df.iloc[0:0]
    if len(d):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        m = "mdrop" if "del_cited_mdrop" in d else "drop"
        cols = [(f"del_cited_{m}", "cited by Gemma"), (f"del_top_{m}", "top by occlusion"), (f"del_rand_{m}", "random")]
        stats = [bootstrap_ci(d[c]) for c, _ in cols]
        ax.bar([l for _, l in cols], [s[0] for s in stats],
               yerr=[[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]], capsize=4)
        ax.set_ylabel("drop in log-odds of decision" if m == "mdrop" else "drop in P(decision)")
        ax.set_title("Does deleting what Gemma cites change its answer?")
        fig.tight_layout()
        fig.savefig(f"{outdir}/deletion.png", dpi=160)
        plt.close(fig)
    if cue_df is not None and "admits_cue" in cue_df:
        s = summarize_cue(cue_df)
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        vals = [s["admitted the cue (among cue-driven flips)"], s["UNFAITHFUL: flipped by cue, never mentioned it"]]
        ax.bar(["admitted cue", "hid cue"], vals)
        ax.set_ylim(0, 1)
        ax.set_title("When a planted hint flips Gemma, does it say so?")
        fig.tight_layout()
        fig.savefig(f"{outdir}/cue_admission.png", dpi=160)
        plt.close(fig)
