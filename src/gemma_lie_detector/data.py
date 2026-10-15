"""Dataset loaders. Each returns a list of `Example`s ready for the pipeline."""
from __future__ import annotations

import numpy as np

from .core import Example

JOB_LEVELS = {0: "unskilled and non-resident", 1: "unskilled and resident",
              2: "skilled", 3: "highly skilled"}


def load_sst2(n=500, seed=0, min_words=8, max_words=40):
    """SST-2 validation sentences (GLUE). Needs `datasets`."""
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="validation")
    rows = [(i, r["sentence"].strip(), r["label"]) for i, r in enumerate(ds)
            if min_words <= len(r["sentence"].split()) <= max_words]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    return [Example(f"sst2-{i}", "text", "positive" if y == 1 else "negative", tuple(s.split()))
            for i, s, y in rows[:n]]


def _tabular_examples(df, label_col, n, seed, prefix, balanced=True):
    rng = np.random.default_rng(seed)
    feats = [c for c in df.columns if c != label_col]
    labels = df[label_col].astype(str).str.lower()
    idx = np.arange(len(df))
    if balanced:
        groups = [idx[(labels == l).to_numpy()] for l in sorted(labels.unique())]
        take = np.concatenate([rng.permutation(g)[: n // len(groups)] for g in groups])
        idx = rng.permutation(take)
    else:
        idx = rng.permutation(idx)[:n]
    return [Example(f"{prefix}-{i}", "tabular", labels.iloc[i], tuple(feats),
                    tuple(str(df.iloc[i][c]) for c in feats)) for i in idx]


def load_credit_kaggle(csv_path, n=300, seed=0):
    """Kaggle `uciml/german-credit` CSV (columns: Age, Sex, Job, Housing,
    Saving accounts, Checking account, Credit amount, Duration, Purpose, Risk)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = [c.strip().lower() for c in df.columns]
    if "job" in df:
        df["job"] = df["job"].map(lambda j: JOB_LEVELS.get(int(j), str(j)) if str(j).lstrip("-").isdigit() else j)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna("unknown")
    return _tabular_examples(df, "risk", n, seed, "credit")


def load_credit_openml(n=300, seed=0):
    """Same German credit data from OpenML (no Kaggle login). Needs scikit-learn."""
    from sklearn.datasets import fetch_openml
    df = fetch_openml("credit-g", version=1, as_frame=True, parser="auto").frame.copy()
    df.columns = [c.replace("_", " ") for c in df.columns]
    return _tabular_examples(df, "class", n, seed, "credit")
