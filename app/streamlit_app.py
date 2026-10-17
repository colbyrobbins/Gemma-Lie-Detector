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

st.set_page_config(page_title="Gemma Lie Detector", layout="centered")
st.title("Gemma Lie Detector")
st.subheader("When Gemma cites words in an explanation, do they actually drive the decision?")
st.caption("Everything here is read from recorded runs of Gemma on SST-2 movie reviews; nothing is recomputed live.")

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


tab_cited, tab_cue = st.tabs(["Do the cited words matter?", "Planted cue test"])

# ---------------------------------------------------------------- cited words vs occlusion vs IG
with tab_cited:
    rec = st.selectbox("Select a review", core, format_func=snippet, key="cited_review")
    m = len(rec["cited_idx"])

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
    left, right = st.columns(2)
    with left:
        mode = options[st.radio("Remove", list(options), key="remove_choice")]
    result = dl.deletion(rec, mode)
    with right:
        st.metric("Margin", f"{result['new_margin']:.1f}", f"{-result['drop']:+.1f}")
        label, sure = dl.confidence(result["new_margin"], rec["pred"])
        st.caption(f"Gemma then favours {label} ({sure} sure).")
        if result["flipped"]:
            st.error(f"Result: decision changes to {dl.other_label(rec)}")
        else:
            st.success(f"Result: decision stays {rec['pred']}")
    st.caption(result["note"].capitalize())

    st.markdown("#### Integrated gradients (IG) findings")
    ig = ig_by_id.get(rec["id"])
    if ig is None:
        st.info("Integrated gradients results for this review are not in app/data yet.")
    else:
        chips(rec["elements"], dl.highlight_classes(rec, ig["attr_ig_zero"]))
        st.caption("Same colors, with integrated gradients in place of occlusion.")

# ---------------------------------------------------------------- planted cue
with tab_cue:
    st.markdown("A hint pointing at the opposite label is added to the review. If it flips Gemma's decision and "
                "removing it brings the original decision back, the hint caused the flip. Does Gemma's explanation admit it?")
    usable = {k: [c for c in v if c["id"] in core_by_id] for k, v in cues_by_kind.items()}
    usable = {k: v for k, v in usable.items() if v}
    if not usable:
        st.info("No planted-cue results in app/data yet.")
    else:
        kind = st.selectbox("Cue type", list(usable)) if len(usable) > 1 else next(iter(usable))
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
