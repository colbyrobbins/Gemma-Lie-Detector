"""Gemma Lie Detector: viewer for the recorded runs.   Run:  streamlit run app/streamlit_app.py"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_logic as dl

# Chips use solid, clearly different colours: pale red, pale blue, and a dark filled purple for "both".
CHIP = {"cited": "background:#ffb3b3;color:#111;",
        "method": "background:#9fd0ff;color:#111;",
        "both": "background:#7b2cbf;color:#fff;font-weight:700;",
        "": ""}
CUE_BORDER = "outline:2px solid #f0a000;"
CUE_NAMES = {"explicit": "Explicit cue (\"A previous reviewer noted...\")",
             "opinion": "Opinion cue (\"My own guess is...\")"}

st.set_page_config(page_title="Gemma Lie Detector", layout="centered")
st.title("Gemma Lie Detector")
st.subheader("When Gemma cites words in an explanation, do they actually drive the decision?")
st.caption("Everything here is read from recorded runs of Gemma on 300 SST-2 movie reviews; nothing is recomputed live.")

core = [r for r in dl.load_core() if dl.testable(r)]
core_by_id = {r["id"]: r for r in dl.load_core()}
ig_by_id = dl.load_ig()
cues_by_kind = dl.load_cues()
if not core:
    st.error("No data found in app/data/demo_core.jsonl.")
    st.stop()


def chips(words, classes, outlined=()):
    """Render words as coloured chips; `outlined` indices get an orange outline (the planted cue)."""
    html = [f"<span style='{CHIP[c]}{CUE_BORDER if i in outlined else ''}padding:2px 5px;margin:1px;"
            f"border-radius:4px;display:inline-block'>{w}</span>" for i, (w, c) in enumerate(zip(words, classes))]
    st.markdown(" ".join(html), unsafe_allow_html=True)


def snippet(rec):
    """Label for the review picker: the start of the review."""
    return " ".join(rec["elements"])[:90]


tab_find, tab_cited, tab_cue = st.tabs(["Findings", "Do the cited words matter?", "Planted cue test"])

# ---------------------------------------------------------------- summary of the recorded runs
with tab_find:
    h = dl.headline(list(core_by_id.values()), ig_by_id, cues_by_kind)
    st.markdown("#### Do the cited words matter?")
    st.caption(f"Share of the {h['n']} reviews where deleting words flips Gemma's decision. "
               "The same number of words is deleted each time.")
    a, b, c = st.columns(3)
    a.metric("Gemma's cited words", f"{h['cited_flip']:.0%}")
    b.metric("Random words", f"{h['random_flip']:.0%}")
    c.metric("Top words by occlusion", f"{h['top_flip']:.0%}",
             help="Occlusion deletes one word at a time and ranks words by how far the decision's margin falls. "
                  "These are the words it ranks highest.")
    st.markdown("Gemma's citations matter far more than random words, but less than the words occlusion ranks highest.")

    st.markdown("#### Do other methods agree?")
    st.caption(f"Integrated gradients (IG) is trusted on {h['ig_ok']} of {h['ig_total']} reviews, where its scores "
               "add up to the change in margin. Both numbers are decision flips on those reviews.")
    a, b = st.columns(2)
    a.metric("Gemma's cited words", f"{h['ig_cited_flip']:.0%}")
    b.metric("Top words by IG", f"{h['ig_flip']:.0%}")
    st.markdown("IG ranks differently from occlusion, and its top words flip the decision less often than Gemma's citations do.")

    st.markdown("#### Does Gemma admit a planted hint?")
    st.caption("Among reviews where the hint caused the decision to flip, the share whose explanation refers to it "
               "as a hint, note or guess.")
    cols = st.columns(max(len(h["cues"]), 1))
    for col, (kind, v) in zip(cols, h["cues"].items()):
        col.metric(CUE_NAMES.get(kind, kind), f"{v['mention'] / v['driven']:.0%}" if v["driven"] else "n/a",
                   help=f"{v['mention']} of {v['driven']} hint-caused flips mention the hint. "
                        f"The hint flipped the decision in {v['flipped']} of {v['n']} reviews.")
    st.markdown("About a third of the time Gemma changes its answer because of the hint and never refers to it as a hint. "
                "Merely quoting the hint's words, such as 'negative', does not count as referring to it.")

# ---------------------------------------------------------------- cited words vs occlusion vs IG
with tab_cited:
    counts = {name: len(dl.filter_reviews(core, name)) for name in dl.REVIEW_FILTERS}
    view_name = st.selectbox("Show", list(counts), format_func=lambda n: f"{n} ({counts[n]})", key="review_filter")
    rec = st.selectbox("Select a review", dl.filter_reviews(core, view_name), format_func=snippet, key="cited_review")
    m = len(rec["cited_idx"])
    ig = ig_by_id.get(rec["id"])
    ig_ok = ig is not None and ig["ig_err_zero"] < dl.IG_MAX_ERR   # IG scores add up to the margin change

    st.markdown(f"**Gemma's Decision:** {rec['pred']}")
    st.markdown(f"**Confidence margin:** {rec['base_margin']:.1f} ({dl.confidence(rec['base_margin'], rec['pred'])[1]} sure)",
                help="The margin is the log-odds of the decision against the other label. 0 is a coin flip, "
                     "2 is about 88%, 5 about 99%, 10 about 99.995%. Negative means the other label wins.")
    chips(rec["elements"], dl.highlight_classes(rec, rec["attr_logodds"]))
    st.caption("Red: words Gemma cited. Blue: top words by occlusion. Purple: both (agreement).")
    st.markdown(f"**Gemma's Explanation:** {rec['reason']}  \nCited: {', '.join(rec['cited_raw'])}")

    st.markdown("#### Occlusion test")
    options = {"The words Gemma cited": "cited", f"The top {m} words by occlusion": "top",
               f"{m} random words (average)": "random"}
    if ig is not None:
        options[f"The top {m} words by integrated gradients"] = "ig"
    left, right = st.columns(2)
    with left:
        mode = options[st.radio("Remove", list(options), key="remove_choice")]
    result = dl.deletion_ig(rec, ig) if mode == "ig" else dl.deletion(rec, mode)
    with right:
        st.metric("Margin", f"{result['new_margin']:.1f}", f"{-result['drop']:+.1f}")
        label, sure = dl.confidence(result["new_margin"], rec["pred"])
        st.caption(f"Gemma then favours {label} ({sure} sure).")
        if result["flipped"]:
            st.error(f"Result: decision changes to {dl.other_label(rec)}")
        else:
            st.success(f"Result: decision stays {rec['pred']}")
    st.caption(result["note"].capitalize())
    if mode == "ig" and not ig_ok:
        st.warning("Integrated gradients' scores do not add up to the margin change for this review, so its top words may be unreliable.")

    st.markdown("#### Integrated gradients (IG) findings")
    if ig is None:
        st.info("Integrated gradients results for this review are not in app/data yet.")
    else:
        chips(rec["elements"], dl.highlight_classes(rec, ig["attr_ig_zero"]))
        st.caption("Same colors, with integrated gradients in place of occlusion.")
        if not ig_ok:
            st.warning("For this review integrated gradients' scores do not add up to the change in margin "
                       f"(error {ig['ig_err_zero']:.0%}), so treat the blue words with caution.")

# ---------------------------------------------------------------- planted cue
with tab_cue:
    st.markdown("A hint pointing at the opposite label is added to the review. If it flips Gemma's decision and "
                "removing it brings the original decision back, the hint caused the flip. Does Gemma's explanation admit it?")
    usable = {k: [c for c in v if c["id"] in core_by_id] for k, v in cues_by_kind.items()}
    usable = {k: v for k, v in usable.items() if v}
    if not usable:
        st.info("No planted-cue results in app/data yet.")
    else:
        kind = (st.selectbox("Cue type", list(usable), format_func=lambda k: CUE_NAMES.get(k, k))
                if len(usable) > 1 else next(iter(usable)))
        n, n_flipped, n_driven, n_mention = dl.cue_summary(usable[kind], kind)
        a, b, c = st.columns(3)
        a.metric("Flipped by cue", f"{n_flipped}/{n}", help="Reviews where adding the cue changed Gemma's decision.")
        b.metric("Caused by cue", f"{n_driven}/{n_flipped}",
                 help="Flips where removing the cue brought the original decision back.")
        c.metric("Cue mentioned", f"{n_mention}/{n_driven}",
                 help="Cue-caused flips whose explanation refers to the planted cue.")

        driven = [c for c in usable[kind] if c.get("cue_driven_flip")]
        if driven:
            cue = st.selectbox("Select a review", driven, format_func=lambda c: snippet(core_by_id[c["id"]]), key="cue_review")
            view = dl.cue_view(core_by_id[cue["id"]], cue, kind)
            present = st.radio("Planted cue", ["Present", "Removed"], horizontal=True, key="cue_toggle") == "Present"
            if present:
                classes = ["cited" if i in view["cited_idx"] else "" for i in range(len(view["words"]))]
                chips(view["words"], classes, outlined=view["cue_idx"])
                st.caption("Orange outline: the planted cue. Red: words Gemma cited.")
                st.markdown(f"**Gemma's Decision:** {view['decision_present']}")
                st.caption(f"The cue moved the log-odds toward '{view['cue_label']}' by {view['margin_shift']:+.1f}.")
                if view["reason"] is None:
                    st.code(view["raw"])
                else:
                    st.markdown(f"**Gemma's Explanation:** {view['reason']}  \nCited: {', '.join(view['cited_raw'])}")
                if view["mentions_cue"]:
                    st.success("The explanation mentions the planted cue.")
                else:
                    st.error("The explanation never mentions the planted cue.")
            else:
                keep = [i for i in range(len(view["words"])) if i not in view["cue_idx"]]
                chips([view["words"][i] for i in keep], [""] * len(keep))
                st.markdown(f"**Gemma's Decision:** {view['decision_removed']}")
                st.caption("With the cue removed, the original decision comes back.")
