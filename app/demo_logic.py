"""Pure-Python logic behind the Streamlit demo (no Streamlit import, so it is unit-testable).

Everything shown is read from recorded runs of Gemma; nothing is recomputed live. Only deletions
that were actually run are available: the words Gemma cited, the top occlusion words, and the
average over random words of the same count.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, os.path.join(HERE, "..", "src"))   # use the project's own cue and admission code

from gemma_lie_detector.core import Example, match_groups, with_cue  # noqa: E402
from gemma_lie_detector.experiments import cue_admission  # noqa: E402
from gemma_lie_detector.prompts import parse_explanation  # noqa: E402


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_core(path=None):
    """Core-experiment records (one per review): decision, citations, occlusion scores, deletion results."""
    return load_jsonl(path or os.path.join(DATA, "demo_core.jsonl"))


def load_ig(path=None):
    """Integrated-gradients records from the deep dive, keyed by review id (empty if not available yet)."""
    recs = load_jsonl(path or os.path.join(DATA, "demo_deepdive.jsonl"))
    return {r["id"]: r for r in recs if not r.get("skipped")}


def testable(rec):
    """Reviews with at least one valid cited word, which the demo needs."""
    return bool(rec.get("parse_ok", True) and rec.get("cited_idx"))


def top_words(scores, m):
    """Indices of the m highest scores (ties broken by position, as in the experiments)."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:m]


def highlight_classes(rec, scores):
    """Colour class of every word: 'both' (purple), 'cited' (red), 'method' (blue) or ''.

    `scores` is the attribution being compared with Gemma's citations (occlusion or IG). The top
    words are as many as Gemma cited, so red, blue and purple are directly comparable and the
    'remove' choices below delete exactly the words that are highlighted.
    """
    cited = set(rec["cited_idx"])
    top = set(top_words(scores, len(cited)))
    return ["both" if i in cited and i in top else "cited" if i in cited else "method" if i in top else ""
            for i in range(len(rec["elements"]))]


def deletion(rec, mode):
    """Recorded outcome of removing words. mode: 'cited' | 'top' | 'random'.

    Returns dict(new_margin, drop, flipped, note). For 'random' only the average margin drop and the
    share of random draws that flipped the decision were recorded, so `flipped` is True when most
    of the draws flipped it and the note gives the exact share.
    """
    base, words = rec["base_margin"], rec["elements"]
    if mode == "cited":
        drop, flipped = rec["del_cited_mdrop"], bool(rec["del_cited_flip"])
        note = "removed " + ", ".join(f"'{words[i]}'" for i in rec["cited_idx"])
    elif mode == "top":
        drop, flipped = rec["del_top_mdrop"], bool(rec["del_top_flip"])
        note = "removed " + ", ".join(f"'{words[i]}'" for i in top_words(rec["attr_logodds"], len(rec["cited_idx"])))
    elif mode == "random":
        drop, flipped = rec["del_rand_mdrop"], rec["del_rand_flip"] >= 0.5
        note = f"average over random draws; the decision changed in {rec['del_rand_flip']:.0%} of them"
    else:
        raise ValueError(mode)
    return {"drop": drop, "new_margin": base - drop, "flipped": flipped, "note": note}


def deletion_ig(rec, ig_rec):
    """Recorded outcome of deleting the words integrated gradients ranks highest (as many as Gemma cited).

    Same shape as `deletion`. Comes from the deep dive, where this deletion was run for every review.
    """
    words, m = rec["elements"], len(rec["cited_idx"])
    drop = ig_rec["topm_mdrop_ig_zero"]
    note = "removed " + ", ".join(f"'{words[i]}'" for i in top_words(ig_rec["attr_ig_zero"], m))
    return {"drop": drop, "new_margin": rec["base_margin"] - drop, "flipped": bool(ig_rec["topm_flip_ig_zero"]),
            "note": note}


IG_MAX_ERR = 0.1   # IG scores are trusted only when they add up to the margin change within 10%

# Which reviews the picker offers: label -> test on a core record.
REVIEW_FILTERS = {
    "All reviews": lambda r: True,
    "Deleting the cited words flips the decision": lambda r: bool(r["del_cited_flip"]),
    "Cited words flip it, occlusion's top words do not": lambda r: bool(r["del_cited_flip"]) and not r["del_top_flip"],
    "Occlusion's top words flip it, cited words do not": lambda r: bool(r["del_top_flip"]) and not r["del_cited_flip"],
}


def filter_reviews(core, name):
    """The reviews matching one of `REVIEW_FILTERS` (a way to jump to the interesting cases)."""
    keep = REVIEW_FILTERS[name]
    return [r for r in core if keep(r)]


def headline(core, ig_by_id, cues_by_kind):
    """The summary numbers on the Findings tab, all computed from the recorded runs.

    Rates are over every review in `core`. IG numbers use only reviews whose IG completeness error is
    under `IG_MAX_ERR`, and compare the cited words with IG's top words on those same reviews.
    """
    def rate(rows, key):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    out = {"n": len(core), "cited_flip": rate(core, "del_cited_flip"), "top_flip": rate(core, "del_top_flip"),
           "random_flip": rate(core, "del_rand_flip"), "cited_drop": rate(core, "del_cited_mdrop"),
           "top_drop": rate(core, "del_top_mdrop"), "random_drop": rate(core, "del_rand_mdrop")}
    core_by_id = {r["id"]: r for r in core}
    ok = [(core_by_id[i], g) for i, g in ig_by_id.items()
          if i in core_by_id and g.get("ig_err_zero") is not None and g["ig_err_zero"] < IG_MAX_ERR]
    out.update(ig_total=len(ig_by_id), ig_ok=len(ok),
               ig_cited_flip=rate([c for c, _ in ok], "del_cited_flip"), ig_flip=rate([g for _, g in ok], "topm_flip_ig_zero"))
    out["cues"] = {}
    for kind, cues in cues_by_kind.items():
        n, flipped, driven, mention = cue_summary([c for c in cues if c["id"] in core_by_id], kind)
        out["cues"][kind] = {"n": n, "flipped": flipped, "driven": driven, "mention": mention}
    return out


def other_label(rec, labels=("negative", "positive")):
    return labels[0] if rec["pred"] == labels[1] else labels[1]


def load_cues(data_dir=None):
    """Planted-cue records by cue type, from every demo_cue_<type>.jsonl present."""
    data_dir = data_dir or DATA
    out = {}
    for fn in sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else []:
        if fn.startswith("demo_cue_") and fn.endswith(".jsonl"):
            out[fn[len("demo_cue_"):-len(".jsonl")]] = load_jsonl(os.path.join(data_dir, fn))
    return out


def cue_outcome(cue, kind):
    """Recompute from the stored raw reply whether the explanation mentions the cue (strict rule).

    Older result files stored a looser flag, so the app always recomputes it.
    """
    return cue_admission(parse_explanation(cue["raw"]), kind)


def cue_summary(cues, kind):
    """(reviews, flipped by the cue, flips the cue caused, caused flips whose explanation mentions the cue)."""
    driven = [c for c in cues if c.get("cue_driven_flip")]
    return (len(cues), sum(bool(c.get("flipped")) for c in cues), len(driven),
            sum(cue_outcome(c, kind) for c in driven))


def cue_view(core_rec, cue, kind):
    """Everything the cue demo shows for one review, with the cue present and with it removed.

    The decision with the cue is the cue's label if it flipped Gemma, else the clean decision. After
    removing the cue it is the clean decision if the cue was what changed it (`cue_restored`).
    """
    clean = cue["clean_pred"]
    cued = with_cue(Example(core_rec["id"], "text", core_rec["gold"], tuple(core_rec["elements"])),
                    cue["cue_label"], kind)
    expl = parse_explanation(cue["raw"])
    cited = set().union(*match_groups(cued, expl["cited"])) if expl else set()
    return {"words": list(cued.elements), "cue_idx": set(cued.cue_idx), "cited_idx": cited,
            "decision_present": cue["cue_label"] if cue["flipped"] else clean,
            "decision_removed": clean if cue["cue_restored"] else cue["cue_label"],
            "margin_shift": cue["cue_margin_shift"], "cue_label": cue["cue_label"],
            "reason": expl["reason"] if expl else None, "cited_raw": expl["cited"] if expl else [],
            "raw": cue["raw"], "mentions_cue": cue_outcome(cue, kind)}


def confidence(margin, pred, labels=("negative", "positive")):
    """(label, text) for whichever label a margin favours, with a readable confidence.

    A margin is log-odds, so the probability is 1 / (1 + e^-margin). A negative margin means the other
    label now wins. Above 99.9% the exact figure is hidden ("99.9%+") because it would round to 100%;
    the margin itself still shows the difference.
    """
    other = labels[0] if pred == labels[1] else labels[1]
    p = 1.0 / (1.0 + math.exp(-abs(margin)))
    return (pred if margin >= 0 else other), ("99.9%+" if p >= 0.999 else f"{p:.1%}")
