"""Poster figures for the Gemma Lie Detector, drawn from the saved results in results/text_results/.

    python scripts/make_poster_figures.py            # writes results/poster_figures/*.png

Four figures, each built to carry one message on a poster:
  1_deletion_flips      how often deleting words flips the decision (Gemma's citations vs the rest)
  2_cue_admission       when a planted hint flips the decision, does the explanation mention it?
  3_who_flips           which reviews flip when Gemma's cited words vs occlusion's top words are deleted
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
from gemma_lie_detector.analysis_deep import (cited_word_ranks, exact_sign_test, load_deep, reliable_ig,  # noqa: E402
                                              triangulation_table)

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


def fig_who_flips(core):
    """Of the 300 reviews, which flip when Gemma's cited words are deleted, which when occlusion's top words are?

    Both tests delete the same number of words per review (the number Gemma cited), so the bar splits the
    reviews into four groups. Ordering the groups as [only cited | both | only occlusion | neither] keeps each
    method's flips contiguous, so a bracket under the bar can total them.
    """
    cited, top = core.del_cited_flip.astype(bool), core.del_top_flip.astype(bool)
    n = len(core)
    only_c, both = int((cited & ~top).sum()), int((cited & top).sum())
    only_o, neither = int((~cited & top).sum()), int((~cited & ~top).sum())
    p = exact_sign_test(only_c, only_o)

    groups = [(only_c, "Only\nGemma", GEMMA, INK, "center"), (both, "Both", INK, "white", "center"),
              (only_o, "Only\nocclusion", OCCLUSION, "white", "center"), (neither, "Neither", LIGHT, INK, "center")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    left = 0
    for count, label, colour, text_colour, align in groups:
        ax.barh(0, count, left=left, height=0.5, color=colour, edgecolor="white", linewidth=2, zorder=3)
        ax.text(left + count / 2, 0, str(count), ha="center", va="center", fontweight="bold", color=text_colour, fontsize=14)
        ax.text(left + count / 2, 0.36, label, ha="center", va="bottom", fontsize=11)
        left += count

    def bracket(x0, x1, y, text):
        """Bracket under the bar spanning x0..x1 with its label to the right."""
        ax.plot([x0, x0, x1, x1], [y + 0.07, y, y, y + 0.07], color=INK, linewidth=1.4, clip_on=False)
        ax.text(x1 + 5, y, text, va="center", fontsize=12, fontweight="bold")

    bracket(0, only_c + both, -0.45, f"Gemma's cited words flip it: {only_c + both} ({(only_c + both) / n:.0%})")
    bracket(only_c, only_c + both + only_o, -0.8, f"Occlusion's top words flip it: {both + only_o} ({(both + only_o) / n:.0%})")
    ax.text(0, -1.25, "Each test deletes the same number of words (the number Gemma cited).\n"
                      f"Exact sign test on the {only_c + only_o} reviews where only one test flips: p = {p:.3f}",
            fontsize=10, color=MUTED, va="center", linespacing=1.5)
    ax.set_xlim(0, n)
    ax.set_ylim(-1.5, 1.05)
    ax.axis("off")
    ax.set_title(f"Occlusion's top words flip {only_o} reviews that Gemma's cited words miss;\nthe reverse happens in {only_c} "
                 f"({n} reviews)", loc="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "3_who_flips")


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
    fig_who_flips(core)
    fig_citation_agreement(deep)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
