import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import demo_logic as dl


def test_demo_data_and_deletion():
    core = [r for r in dl.load_core() if dl.testable(r)]
    assert core, "app/data/demo_core.jsonl missing"
    r = core[0]
    assert dl.deletion(r, "cited")["flipped"] == bool(r["del_cited_flip"])
    top = dl.deletion(r, "top")
    assert abs(top["new_margin"] - (r["base_margin"] - r["del_top_mdrop"])) < 1e-9
    assert dl.deletion(r, "random")["flipped"] == (r["del_rand_flip"] >= 0.5)
    classes = dl.highlight_classes(r, r["attr_logodds"])
    assert len(classes) == r["n_elements"]
    assert {i for i, c in enumerate(classes) if c in ("cited", "both")} == set(r["cited_idx"])
    assert sum(c in ("method", "both") for c in classes) == len(r["cited_idx"])   # as many top words as cited words


def test_top_words_ties_and_ig_loading(tmp_path):
    assert dl.top_words([0.0, 5.0, 5.0, 1.0], 2) == [1, 2]            # ties broken by position
    assert dl.load_ig(str(tmp_path / "missing.jsonl")) == {}           # no IG data yet is fine
    (tmp_path / "d.jsonl").write_text('{"id": "a", "attr_ig_pad": [1.0]}\n{"id": "b", "skipped": true}\n')
    assert list(dl.load_ig(str(tmp_path / "d.jsonl"))) == ["a"]


def test_cue_view_and_summary():
    core = {r["id"]: r for r in dl.load_core()}
    cues = dl.load_cues()["explicit"]
    n, flipped, driven, mention = dl.cue_summary(cues, "explicit")
    assert n == len(cues) and driven <= flipped <= n and mention <= driven
    cue = next(c for c in cues if c.get("cue_driven_flip"))
    view = dl.cue_view(core[cue["id"]], cue, "explicit")
    assert view["decision_present"] == cue["cue_label"] and view["decision_removed"] == cue["clean_pred"]
    n_words = len(core[cue["id"]]["elements"])
    assert len(view["words"]) > n_words and view["cue_idx"] == set(range(n_words, len(view["words"])))
    assert view["mentions_cue"] == dl.cue_outcome(cue, "explicit")


def test_confidence_text():
    assert dl.confidence(0.0, "positive") == ("positive", "50.0%")
    assert dl.confidence(2.0, "positive") == ("positive", "88.1%")
    assert dl.confidence(-2.0, "positive") == ("negative", "88.1%")      # negative margin: the other label wins
    assert dl.confidence(11.5, "negative") == ("negative", "99.9%+")


def test_filters_ig_deletion_and_headline():
    core = [r for r in dl.load_core() if dl.testable(r)]
    ig = dl.load_ig()
    assert ig, "app/data/demo_deepdive.jsonl missing"

    only_cited = dl.filter_reviews(core, "Cited words flip it, occlusion's top words do not")
    only_top = dl.filter_reviews(core, "Occlusion's top words flip it, cited words do not")
    assert only_cited and only_top and len(dl.filter_reviews(core, "All reviews")) == len(core)
    assert all(r["del_cited_flip"] and not r["del_top_flip"] for r in only_cited)
    assert all(r["del_top_flip"] and not r["del_cited_flip"] for r in only_top)

    rec = core[0]
    out = dl.deletion_ig(rec, ig[rec["id"]])
    assert abs(out["new_margin"] - (rec["base_margin"] - ig[rec["id"]]["topm_mdrop_ig_zero"])) < 1e-9
    assert out["note"].count("'") == 2 * len(rec["cited_idx"])        # as many words removed as Gemma cited

    h = dl.headline(core, ig, dl.load_cues())
    assert h["n"] == len(core) and h["cited_flip"] > 2 * h["random_flip"]
    assert 0 < h["ig_ok"] <= h["ig_total"] and set(h["cues"]) >= {"explicit"}
    c = h["cues"]["explicit"]
    assert c["mention"] <= c["driven"] <= c["flipped"] <= c["n"]
