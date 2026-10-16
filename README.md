# Gemma Lie Detector

**When Gemma explains a text classification by citing specific words, do those words actually drive the decision?**

Language models will happily tell you *why* they answered the way they did ("I chose negative because of *tedious* and *dull*"). This project tests whether that story is true. We treat the explanation as a claim about the model's own behaviour and check the claim with interventions: delete what the model says mattered and see whether the answer changes; plant a misleading hint and see whether the model admits it was swayed.

> Status: **work in progress** (final project for Duke AIPI Responsible AI). Code, tests and Colab notebooks are in place; results sections below are filled in after the Gemma runs.
>
> **Scope:** the evaluated study is on **text** (SST-2 movie reviews). Tabular support (German credit) is implemented and tested on a mock model but has **not been evaluated on Gemma**; it is future work. Images are out of scope.

## How it works

One model (`google/gemma-3-4b-it`), one prompt format, three kinds of evidence.

1. **Decision and explanation come from the same prompt.** Gemma is asked for JSON `{"label", "cited", "reason"}` and its reply is forced to start with `{"label": "`. The next-token probabilities over the label words give a *measurable decision*; greedy decoding of the rest gives the *explanation*. So the decision being explained is exactly the one the explanation was written beside.
2. **Ground truth = occlusion.** Every input is a list of ablatable *elements* (words for text, feature columns for tabular). Removing one element at a time gives a causal importance score for each, measured as the drop in the decision's log-odds.
3. **Compare the claim to the ground truth.** Because one-word deletion is an imperfect yardstick, a separate deep dive cross-checks it (see below).

| Test | Question | Metric |
|---|---|---|
| Agreement | Do the cited elements rank at the top of the occlusion ranking? | precision@k, AUROC (cited vs uncited), valid-citation rate |
| Deletion (comprehensiveness) | Does deleting the cited elements hurt the decision more than deleting random ones? | drop in P(decision), flip rate, gap vs random and vs top-by-occlusion |
| Sufficiency | Does the decision survive if *only* the cited elements are kept? | flip rate |
| Planted cue | When a hint pointing at the opposite label flips the decision, does the explanation admit it? | cue-driven flip rate, admission rate |
| Disclosure (tabular) | Which features drive decisions but are never cited (e.g. `age`, `sex`)? | driver rate vs cited rate per feature |

A flip counts as *cue-driven* only if removing the cue restores the original decision, so the test is causal rather than anecdotal.

## Deep dive: can the occlusion reference be trusted?

Deleting one word at a time can mislead (broken grammar, redundant words hiding each other, words that matter only in combination). Notebook `03_text_deepdive` re-uses the saved citations and asks how much that matters:

| Check | What it does |
|---|---|
| Replace instead of delete | Swaps each word for a placeholder; if rankings agree with deletion, the grammar-breaking artifact is not driving the result |
| Integrated gradients | A gradient-based attribution (pad and zero baselines) as an independent second opinion, with a completeness check that flags unreliable runs |
| Smallest flip set | Searches subsets of up to 3 candidate words for the fewest whose deletion flips the decision, and checks whether Gemma cited them |
| One behavioural judge | Deletes each method's top-m words (and Gemma's cited words, and random words) and compares how far the margin falls, plus deletion curves |

The question becomes three-way: do Gemma's citations, occlusion and integrated gradients agree, and which of them best predicts what really breaks the decision?

## Modalities

| Modality | Element | Ablation | Data | Status |
|---|---|---|---|---|
| Text | word | delete or replace | SST-2 (GLUE validation) | **evaluated** |
| Tabular | feature column | value becomes "not provided" | German credit (Kaggle `uciml/german-credit` or OpenML `credit-g`) | implemented and unit-tested, **not evaluated** (future work) |
| Image | named object / region | mask or blur | COCO subset | out of scope |

## Quick start

**Run Gemma (Colab, free T4 is enough to start)**

1. Accept the license at <https://huggingface.co/google/gemma-3-4b-it> (while logged in) and create a read token.
2. In Colab add a secret named `HF_TOKEN` with notebook access on, and select a GPU runtime.
3. Open `notebooks/01_text_core.ipynb` and run top to bottom, then `03_text_deepdive.ipynb` (it reuses 01's citations). `02_tabular.ipynb` exists but has not been run on Gemma. Start with `N = 50`; the initial test cell shows raw model output so you can check parsing before the long run. Results are written as resumable JSONL under `results/text_results/` (text) and committed, so a fresh clone resumes where the last run stopped.

**Browse the results (no GPU)**

```bash
pip install streamlit
streamlit run app/streamlit_app.py
```

The app shows recorded runs only (a snapshot in `app/data/`). One tab lets you pick a review, remove Gemma's cited words, the top occlusion words or random words, and see the recorded change in its decision, next to the integrated-gradients view. The other tab replays the planted-cue test: toggle the hint on and off and see whether Gemma's explanation admits it. Only deletions that were actually run are available.

**Run the tests (no GPU, no downloads)**

```bash
pip install -r requirements.txt
pip install -e .
pytest
```

The tests use a mock bag-of-words model with an "honest" and a "liar" personality and check that the audit tells them apart.

## Repo layout

```
src/gemma_lie_detector/
  core.py         Example, ablation, matching cited strings to elements, planted cues
  prompts.py      shared prompt/prefix, JSON parsing
  model.py        Gemma wrapper: score() -> label probabilities, generate() -> explanation
  metrics.py      AUROC, precision@k, log-odds, rank correlation
  experiments.py  per-example audit + planted-cue test, resumable JSONL runners
  robustness.py   replacement occlusion, smallest flip set, deletion curves
  attribution.py  integrated gradients (numeric core, token-to-word mapping, Gemma adapter)
  deepdive.py     compares all attribution methods on one review; resumable runner
  data.py         SST-2 and German-credit loaders
  analysis.py     summary tables with bootstrap CIs, disclosure table, figures
  analysis_deep.py  deep-dive tables and figures
app/              Streamlit viewer for recorded runs (app/data = snapshot)
notebooks/        01_text_core, 03_text_deepdive (Colab); 02_tabular (not yet run)
tests/            mock-model tests
results/          JSONL outputs and figures
```

## Design decisions

- **Occlusion as ground truth, not as truth.** Occlusion is a causal intervention on the real model, which is why it is the yardstick. It is not perfect (see limitations), so the deep dive cross-checks it with replacement occlusion, integrated gradients and a smallest-flip-set search.
- **Margins from log-probabilities, never from probabilities.** A first 50-review run showed that Gemma's P(decision) is exactly 1.0 in float32 for 96% of SST-2 reviews, so every single-word deletion scored zero and the occlusion ranking was empty for two thirds of the reviews. The scorer now returns log-probabilities and importance is the drop in the decision's log-odds (its *margin*), which stays informative at any confidence. Deletion results are reported as margin drop and as flip rate.
- **Label read from logits, not parsed from text.** This gives a continuous decision score and removes parse failures from the ablation loop. A `label_mass` check (should be near 1) guards the assumption that the model starts its answer with a label word.
- **Tied attributions count as ties.** In precision@k, elements tied with the k-th attribution all count as top-k, but an element with zero measurable effect never does.
- **The deep dive reuses the core citations.** It reads Gemma's cited words from the core results file instead of regenerating them, so both experiments judge the very same explanations.
- **Integrated gradients are reported with a completeness check.** IG's scores should sum to the change in margin from baseline to input; reviews where they do not are flagged rather than trusted, because bfloat16 gradients can be noisy. Two baselines (padding embedding, zeros) are used because IG results depend on the baseline.
- **The flip-set search is exhaustive but capped.** It tries every subset of up to 3 words from a small candidate pool (Gemma's cited words plus the top words of occlusion and IG), so "no flip set found" means none in that pool, not none anywhere.
- **Everything resumable.** Colab disconnects; each example is appended to a JSONL file and skipped on re-run.

## Known limitations

- **One-at-a-time occlusion misses interactions** (negation, "not good") and under-credits words that are repeated or redundant. The flip-set search and deletion curves in the deep dive probe this, but only within their limits.
- **Integrated gradients depend on the baseline and on numerical noise.** Disagreement with occlusion is not proof that either is wrong.
- **Deleting words makes ungrammatical text**; this is an out-of-distribution intervention, so a drop can partly reflect confusion rather than importance. The random-deletion baseline controls for this to a degree.
- **Cue admission is detected with a keyword rule** plus whether the cue was cited, so it can miss paraphrases. Plan: hand-check a sample of about 30 explanations and report agreement.
- **Small model, short inputs, one evaluated dataset.** Findings are about `gemma-3-4b-it` on SST-2, not LLMs in general. The tabular pipeline has not been run on Gemma.
- **Numerical precision.** Gemma 3 may be unstable in fp16 on a T4; the scorer raises on non-finite logits and the notebook lets you switch dtype or model.
- **Early results are from small runs** (50 reviews to start); check the confidence intervals before reading anything into a difference.

## Related work

- Turpin et al., 2023. *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.* <https://arxiv.org/abs/2305.04388> (biasing features that models do not mention; the inspiration for the planted-cue test)
- Madsen, Chandar, Reddy, 2024. *Are self-explanations from Large Language Models faithful?* Findings of ACL. <https://arxiv.org/abs/2401.07927>
- DeYoung et al., 2020. *ERASER: A Benchmark to Evaluate Rationalized NLP Models.* (comprehensiveness and sufficiency)
- Jacovi and Goldberg, 2020. *Towards Faithfully Interpretable NLP Systems.*

**What is new here:** a causally verified planted-cue test and a three-way comparison of Gemma's citations, occlusion and integrated gradients, with a flip-set search to stress-test the occlusion reference, all on an open Google model.

## License and model terms

Code: see `LICENSE`. Gemma weights are governed by Google's Gemma terms of use and are not redistributed here.
