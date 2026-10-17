"""Tests for the deep-dive: robustness helpers, integrated gradients, and the comparison pipeline.

The mock model from test_pipeline.py (a bag-of-words sentiment scorer) stands in for Gemma, so
everything runs without a GPU. For that linear model the exact word attributions are known, which
lets a mock attributor play the role of integrated gradients.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_pipeline import MockScorer, W, _examples  # noqa: E402

from gemma_lie_detector.analysis_deep import (behaviour_comparison, cited_word_ranks,  # noqa: E402
                                              disagreement_cases, exact_sign_test, flip_patterns, load_deep,
                                              make_deep_figures, method_agreement, reliable_ig, summarize_deep,
                                              triangulation_table)
from gemma_lie_detector.attribution import (completeness_error, integrated_gradients,  # noqa: E402
                                            tokens_to_words, word_char_spans)
from gemma_lie_detector.core import Example  # noqa: E402
from gemma_lie_detector.deepdive import run_deep  # noqa: E402
from gemma_lie_detector.experiments import run_core  # noqa: E402
from gemma_lie_detector.metrics import spearman  # noqa: E402
from gemma_lie_detector.prompts import SENTIMENT  # noqa: E402
from gemma_lie_detector.robustness import (build_pool, deletion_curve, minimal_flip_set,  # noqa: E402
                                           occlusion_replace, render_replaced)


class MockAttributor:
    """Exact attributions for the linear mock model: each word contributes its weight to the margin."""
    baselines = ("pad", "zero")

    def attribute(self, task, ex, target, baseline="pad"):
        sign = 1 if target == 0 else -1          # label 0 is "positive"
        return np.array([sign * W.get(w.lower(), 0.0) for w in ex.elements]), 0.0


def test_integrated_gradients_numeric_core():
    w = np.array([1.0, -2.0, 0.5])
    x, base = np.array([1.0, 2.0, -1.0]), np.zeros(3)

    def quad(points):   # f(x) = sum(w * x^2); gradient is linear in the path, so the midpoint rule is exact
        return (w * points ** 2).sum(-1), 2 * w * points

    attr = integrated_gradients(quad, x, base, steps=4)
    assert completeness_error(attr, (w * x ** 2).sum(), 0.0) < 1e-9

    def smooth(points):  # f(x) = sum(tanh(x)), not linear: needs enough steps
        return np.tanh(points).sum(-1), 1 - np.tanh(points) ** 2

    a_small = integrated_gradients(smooth, x, base, steps=10, chunk=4)    # chunk does not divide steps
    a_same = integrated_gradients(smooth, x, base, steps=10, chunk=10)
    assert np.allclose(a_small, a_same)
    assert completeness_error(integrated_gradients(smooth, x, base, steps=64), np.tanh(x).sum(), 0.0) < 1e-3


def test_token_to_word_mapping():
    words = ["a", "great", "film"]
    spans = word_char_spans(words, start=5)
    assert spans == [(5, 6), (7, 12), (13, 17)]
    # offsets as a tokenizer might report them: one token includes the leading space, one is the separator
    offsets = [(0, 5), (5, 6), (6, 12), (12, 17)]
    assert tokens_to_words(offsets, spans) == [-1, 0, 1, 2]


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_robustness_helpers():
    ex = Example("r", "text", "positive", ("great", "wonderful", "great", "dull", "dull"))
    sc = MockScorer()
    assert render_replaced(ex, {1}) == "great [removed] great dull dull"

    # margin of the original decision is 6 - 4 = 2; replacing a positive word lowers it by its weight
    drops = occlusion_replace(sc, SENTIMENT, ex, pred=0, base_margin=2.0)
    assert np.allclose(drops, [2, 2, 2, -2, -2])

    # one positive word cannot flip it (margin 0 is a tie), two can: three equivalent minimal pairs
    flip = minimal_flip_set(sc, SENTIMENT, ex, pred=0, pool=range(5), max_size=3)
    assert flip["size"] == 2 and flip["n_sets"] == 3
    assert all(set(s) <= {0, 1, 2} for s in flip["sets"])
    assert minimal_flip_set(sc, SENTIMENT, ex, pred=0, pool=[3, 4], max_size=2) is None

    curve = deletion_curve(sc, SENTIMENT, ex, pred=0, order=[0, 1, 2, 3, 4], k=10)
    assert len(curve) == 4 and np.allclose(curve, [0, -2, -4, -2])   # capped: never deletes every word
    assert build_pool([4], [[0, 1, 2, 3]], per_ranking=2) == [0, 1, 4]


def test_deepdive_pipeline_and_resume(tmp_path):
    exs = _examples(30)
    core_path, deep_path = str(tmp_path / "core.jsonl"), str(tmp_path / "deep.jsonl")
    run_core(exs, SENTIMENT, MockScorer("honest"), core_path)
    run_deep(exs, SENTIMENT, MockScorer(), MockAttributor(), core_path, deep_path, k_curve=4)
    run_deep(exs, SENTIMENT, MockScorer(), MockAttributor(), core_path, deep_path, k_curve=4)   # resume
    with open(deep_path) as f:
        assert sum(1 for line in f if line.strip()) == len(exs)   # no duplicates after a re-run

    df = load_deep(deep_path)
    assert len(df) > 0 and not df.skipped.any()
    table = summarize_deep(df).set_index("metric")
    # an honest model citing real drivers agrees with every method, and IG on the linear mock is exact
    assert table.loc["integrated gradients (pad baseline): AUROC (cited outranks uncited)", "mean"] > 0.9
    assert table.loc["IG completeness error under 10% (pad baseline)", "mean"] == 1.0
    assert table.loc["top-m deletion, log-odds drop: Gemma's cited words", "mean"] > \
        table.loc["top-m deletion, log-odds drop: random words", "mean"]
    agree = method_agreement(df)
    assert agree.shape == (4, 4) and (np.diag(agree.values) == 1).all()
    assert list(disagreement_cases(df).columns) == ["method", "id", "cited_mdrop", "method_topm_mdrop"]
    make_deep_figures(df, str(tmp_path / "figs"))
    assert (tmp_path / "figs" / "deletion_curves.png").exists()


def test_deepdive_skips_untestable_reviews(tmp_path):
    ex = Example("only-cited", "text", "positive", ("great",))     # the single word is the whole review
    core_path, deep_path = str(tmp_path / "core.jsonl"), str(tmp_path / "deep.jsonl")
    run_core([ex], SENTIMENT, MockScorer("honest"), core_path)
    run_deep([ex], SENTIMENT, MockScorer(), None, core_path, deep_path)
    assert len(load_deep(deep_path)) == 0                           # recorded as skipped, excluded on load


def test_attributor_defaults_and_flip_pool_uses_first_ig_method(tmp_path):
    from types import SimpleNamespace
    from gemma_lie_detector.attribution import GemmaAttributor

    stub = SimpleNamespace(model=SimpleNamespace(requires_grad_=lambda flag: None))
    att = GemmaAttributor(stub)
    assert att.baselines == ("zero",) and att.steps == 64     # defaults chosen from the smoke tests
    assert GemmaAttributor(stub, baselines=("pad", "zero")).baselines == ("pad", "zero")

    class ZeroOnly(MockAttributor):
        baselines = ("zero",)

    exs = _examples(10)
    core_path, deep_path = str(tmp_path / "core.jsonl"), str(tmp_path / "deep.jsonl")
    run_core(exs, SENTIMENT, MockScorer("honest"), core_path)
    run_deep(exs, SENTIMENT, MockScorer(), ZeroOnly(), core_path, deep_path, k_curve=3)
    df = load_deep(deep_path)
    assert "attr_ig_zero" in df and "attr_ig_pad" not in df and df.flip_pool.map(len).min() > 0
    assert method_agreement(df).shape == (3, 3)


def test_combined_method_analysis(tmp_path):
    assert abs(exact_sign_test(9, 17) - 0.1686) < 1e-3       # the early 100-review split: not significant
    assert exact_sign_test(0, 0) == 1.0 and exact_sign_test(6, 6) == 1.0 and exact_sign_test(0, 12) < 0.001

    exs = _examples(40)
    core_path, deep_path = str(tmp_path / "core.jsonl"), str(tmp_path / "deep.jsonl")
    run_core(exs, SENTIMENT, MockScorer("honest"), core_path)
    run_deep(exs, SENTIMENT, MockScorer(), MockAttributor(), core_path, deep_path, k_curve=3)
    df = load_deep(deep_path)

    # the completeness filter drops reviews whose IG scores cannot be trusted
    bad = df.copy()
    bad.loc[:2, "ig_err_zero"] = 0.5
    assert len(reliable_ig(bad)) == len(df) - 3

    table = behaviour_comparison(df)
    assert len(table) == 5 and table.iloc[-1].method.startswith("(cited")
    assert (table.iloc[:4].only_cited + table.iloc[:4].only_method <= table.iloc[:4].n).all()
    assert table.iloc[:4].flip_p.between(0, 1).all()

    patterns = flip_patterns(df)
    assert abs(patterns.share.sum() - 1) < 1e-9 and patterns.reviews.sum() == len(df)

    # an honest model with exact attributions cites words that both methods rank high
    words = cited_word_ranks(df)
    assert len(words) == df.cited_idx.map(len).sum()
    assert (words.group == "high under both").mean() > 0.8
    shares = triangulation_table(words)
    assert abs(shares.iloc[:, 1].sum() - 1) < 1e-9 and (shares.reviews == len(df)).all()
    assert triangulation_table(words.iloc[0:0]).empty
