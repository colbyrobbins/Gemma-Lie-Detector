"""Poster figures for the Gemma Lie Detector, drawn from the saved results in results/text_results/.

    python scripts/make_poster_figures.py            # writes results/poster_figures/*.png

Four figures, each built to carry one message on a poster:
  1_deletion_flips      how often deleting words flips the decision (Gemma's citations vs the rest)
  2_cue_admission       when a planted hint flips the decision, does the explanation mention it?
  3_deletion_curves     margin of the original decision as top-ranked words are deleted one by one
  4_citation_agreement  which methods agree that Gemma's cited words matter

Colour follows the method everywhere: orange = Gemma's citations, blue = occlusion,
green = integrated gradients, grey = random words / baselines.
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from gemma_lie_detector.analysis import bootstrap_ci, load_results, with_admission  # noqa: E402
from gemma_lie_detector.analysis_deep import cited_word_ranks, load_deep, reliable_ig, triangulation_table  # noqa: E402

RESULTS = "results/text_results"
OUT = "results/poster_figures"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
GEMMA, OCCLUSION, IG, NEUTRAL, LIGHT = "#eb6834", "#2a78d6", "#1baf7a", "#8d8c86", "#c9c8c2"

plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 13, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})


def _style(ax, grid_axis="x"):
    """Recessive grid on the value axis only; no spine on the category axis."""
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _bars(ax, labels, values, los, his, colors, fmt="{:.0%}"):
    """Horizontal bars with 95% interval whiskers and a value label at the end of each bar."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.55, color=colors, zorder=3)
    ax.errorbar(values, y, xerr=[np.subtract(values, los), np.subtract(his, values)], fmt="none",
                ecolor=INK, elinewidth=1.4, capsize=4, zorder=4)
    for yi, v, hi in zip(y, values, his):
        ax.text(hi + 0.015, yi, fmt.format(v), va="center", fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.62)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.tick_params(axis="y", length=0)
    _style(ax)


def fig_deletion_flips(core, deep):
    """Panel A: all 300 reviews. Panel B: the 188 reviews where integrated gradients can be trusted."""
    def rate(series):
        m, lo, hi = bootstrap_ci(np.asarray(series, float))
        return m, lo, hi

    a = [("Random words", core.del_rand_flip, NEUTRAL), ("Gemma's cited words", core.del_cited_flip, GEMMA),
         ("Occlusion's top words\n(replace)", deep.topm_flip_replace, OCCLUSION),
         ("Occlusion's top words\n(delete)", core.del_top_flip, OCCLUSION)]
    ok = reliable_ig(deep)
    b = [("Gemma's cited words", ok.ref_cited_flip, GEMMA), ("IG's top words", ok.topm_flip_ig_zero, IG)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.2), height_ratios=[len(a), len(b) + 0.4])
    for ax, rows in ((ax1, a), (ax2, b)):
        stats = [rate(s) for _, s, _ in rows]
        _bars(ax, [r[0] for r in rows], [s[0] for s in stats], [s[1] for s in stats], [s[2] for s in stats],
              [r[2] for r in rows])
    ax1.set_title(f"A.  All {len(core)} reviews", loc="left", fontweight="bold", fontsize=13)
    ax2.set_title(f"B.  The {len(ok)} reviews where integrated gradients (IG) is reliable", loc="left",
                  fontweight="bold", fontsize=13)
    ax2.set_xlabel("Share of reviews where deleting the words flips Gemma's decision (95% interval)")
    fig.suptitle("Gemma's cited words matter, but they are not the most influential words", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(h_pad=2)
    _save(fig, "1_deletion_flips")


def fig_cue_admission(cues):
    """100% bars: of the flips caused by a planted hint, how many explanations mention the hint?"""
    names = {"explicit": "Explicit hint\n\"A previous reviewer noted...\"", "opinion": "Opinion hint\n\"My own guess is...\""}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (kind, df) in enumerate(reversed(list(cues.items()))):
        d = df[df.parse_ok & df.cue_driven_flip.astype(bool)]
        n, k = len(d), int(d.admits_cue.sum())
        ax.barh(i, k / n, height=0.5, color=LIGHT, zorder=3)
        ax.barh(i, 1 - k / n, left=k / n, height=0.5, color=GEMMA, zorder=3)
        ax.text(k / n / 2, i, f"mentions the hint\n{k / n:.0%}  ({k} of {n})", ha="center", va="center", fontsize=12)
        ax.text(k / n + (1 - k / n) / 2, i, f"never\nmentions\n{1 - k / n:.0%}", ha="center", va="center",
                fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(cues)), [names.get(k, k) for k in reversed(list(cues))])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Share of hint-caused flips")
    ax.tick_params(axis="y", length=0)
    ax.set_title("About a third of the time, a planted hint changes the answer and is never mentioned",
                 loc="left", fontsize=14, fontweight="bold")
    _style(ax)
    fig.tight_layout()
    _save(fig, "2_cue_admission")


def fig_deletion_curves(deep):
    """Mean margin of the original decision as the top 1..k words of each ranking are deleted."""
    ok = reliable_ig(deep)
    ks = int(min(len(c) for c in ok.curve_random if len(c)))
    x = np.arange(1, ks + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    series = [("occlusion (delete)", "curve_delete", OCCLUSION), ("integrated gradients", "curve_ig_zero", IG),
              ("random order", "curve_random", NEUTRAL)]
    for label, col, colour in series:
        curves = np.array([c[:ks] for c in ok[col] if len(c) >= ks])
        mean = curves.mean(0)
        ax.plot(x, mean, color=colour, linewidth=2.5, marker="o", markersize=7, markeredgecolor="white",
                markeredgewidth=1.5, zorder=3)
        ax.text(ks + 0.12, mean[-1], label, color=INK, va="center", fontweight="bold")
    ax.axhline(0, color=INK, linewidth=1)
    ax.text(0.75, -1.6, "below 0: the decision has flipped", fontsize=11, color=MUTED, va="center")
    ax.set_ylim(-3, None)
    ax.set_xlim(0.7, ks + 2.1)
    ax.set_xticks(x)
    ax.set_xlabel("Words deleted (highest-ranked first)")
    ax.set_ylabel("Mean margin of the original decision\n(log-odds; below 0 = decision flipped)")
    ax.set_title(f"Occlusion's ranking breaks the decision fastest ({len(ok)} reviews)", loc="left",
                 fontsize=14, fontweight="bold")
    _style(ax, "y")
    fig.tight_layout()
    _save(fig, "3_deletion_curves")


def fig_citation_agreement(deep):
    """Where Gemma's cited words rank: high under both methods, one method, or neither (top 3, score > 0)."""
    tab = triangulation_table(cited_word_ranks(deep)).set_index("group")
    order = [("high under both", "Both methods\nrank it top 3", INK), ("occlusion only", "Occlusion only", OCCLUSION),
             ("IG only", "IG only", IG), ("neither", "Neither method", LIGHT)]
    n = int(tab.reviews.iloc[0])
    fig, ax = plt.subplots(figsize=(8, 2.1))
    left = 0.0
    for key, label, colour in order:
        w = float(tab.loc[key].iloc[0])
        ax.barh(0, w, left=left, height=0.55, color=colour, edgecolor="white", linewidth=2, zorder=3)
        ax.text(left + w / 2, 0, f"{w:.0%}", ha="center", va="center", fontweight="bold",
                color="white" if colour in (INK, OCCLUSION) else INK, fontsize=14)
        ax.text(left + w / 2, 0.32, label, ha="center", va="bottom", fontsize=11)
        left += w
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.35, 0.95)
    ax.axis("off")
    ax.set_title(f"Only {float(tab.loc['high under both'].iloc[0]):.0%} of Gemma's cited words are confirmed by both "
                 f"methods ({n} reviews)", loc="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "4_citation_agreement")


def main():
    core = load_results(f"{RESULTS}/text_core_v2.jsonl")
    deep = load_deep(f"{RESULTS}/text_deepdive_v1.jsonl")
    cues = {"explicit": with_admission(load_results(f"{RESULTS}/text_cue_v2.jsonl")),
            "opinion": with_admission(load_results(f"{RESULTS}/text_cue_opinion_v2.jsonl"))}
    core = core.set_index("id").loc[deep.id].reset_index()          # same reviews, same order
    fig_deletion_flips(core, deep)
    fig_cue_admission(cues)
    fig_deletion_curves(deep)
    fig_citation_agreement(deep)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
