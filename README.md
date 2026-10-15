# Gemma Lie Detector

**When Gemma explains a decision on text or tabular inputs by citing specific elements, do those elements actually drive the decision?**

Language models will happily tell you *why* they answered the way they did ("I chose negative because of *tedious* and *dull*"). This project tests whether that story is true. We treat the explanation as a claim about the model's own behaviour and check the claim with interventions: delete what the model says mattered and see whether the answer changes; plant a misleading hint and see whether the model admits it was swayed.

> Status: **work in progress** (final project for Duke AIPI Responsible AI). Code, tests and Colab notebooks are in place; results sections below are filled in after the Gemma runs.

## How it works

One model (`google/gemma-3-4b-it`), one prompt format, three kinds of evidence.

1. **Decision and explanation come from the same prompt.** Gemma is asked for JSON `{"label", "cited", "reason"}` and its reply is forced to start with `{"label": "`. The next-token probabilities over the label words give a *measurable decision*; greedy decoding of the rest gives the *explanation*. So the decision being explained is exactly the one the explanation was written beside.
2. **Ground truth = occlusion.** Every input is a list of ablatable *elements* (words for text, feature columns for tabular). Removing one element at a time gives a causal importance score for each, measured as the drop in the decision's log-odds.
3. **Compare the claim to the ground truth.**

| Test | Question | Metric |
|---|---|---|
| Agreement | Do the cited elements rank at the top of the occlusion ranking? | precision@k, AUROC (cited vs uncited), valid-citation rate |
| Deletion (comprehensiveness) | Does deleting the cited elements hurt the decision more than deleting random ones? | drop in P(decision), flip rate, gap vs random and vs top-by-occlusion |
| Sufficiency | Does the decision survive if *only* the cited elements are kept? | flip rate |
| Planted cue | When a hint pointing at the opposite label flips the decision, does the explanation admit it? | cue-driven flip rate, admission rate |
| Disclosure (tabular) | Which features drive decisions but are never cited (e.g. `age`, `sex`)? | driver rate vs cited rate per feature |

A flip counts as *cue-driven* only if removing the cue restores the original decision, so the test is causal rather than anecdotal.

## Modalities

| Modality | Element | Ablation | Data | Status |
|---|---|---|---|---|
| Text | word | delete | SST-2 (GLUE validation) | implemented |
| Tabular | feature column | value becomes "not provided" | German credit (Kaggle `uciml/german-credit` or OpenML `credit-g`) | implemented |
| Image | named object / region | mask or blur | COCO subset | planned pilot |

## Quick start

**Run Gemma (Colab, free T4 is enough to start)**

1. Accept the license at <https://huggingface.co/google/gemma-3-4b-it> (while logged in) and create a read token.
2. In Colab add a secret named `HF_TOKEN` with notebook access on, and select a GPU runtime.
3. Open `notebooks/01_text_core.ipynb` (then `02_tabular.ipynb`) and run top to bottom. Start with `N = 50`; the initial test cell shows raw model output so you can check parsing before the long run. Results are written as resumable JSONL under `results/`.

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
  metrics.py      AUROC, precision@k, log-odds
  experiments.py  per-example audit + planted-cue test, resumable JSONL runners
  data.py         SST-2 and German-credit loaders
  analysis.py     summary tables with bootstrap CIs, disclosure table, figures
notebooks/        01_text_core, 02_tabular (Colab)
tests/            mock-model tests
results/          JSONL outputs and figures
```

## Design decisions

- **Occlusion as ground truth, not as truth.** Occlusion is a causal intervention on the real model, which is why it is the yardstick. It is not perfect (see limitations); a second attribution method (Integrated Gradients) is a possible text-only cross-check.
- **Log-odds for ranking, probability for deletion.** Probabilities saturate on confident predictions and flatten the ranking; log-odds do not. Deletion results are reported as probability drop and flip rate because those are easier to read.
- **Label read from logits, not parsed from text.** This gives a continuous decision score and removes parse failures from the ablation loop. A `label_mass` check (should be near 1) guards the assumption that the model starts its answer with a label word.
- **Tied attributions count as ties.** In precision@k, elements tied with the k-th attribution all count as top-k, but an element with zero measurable effect never does.
- **Everything resumable.** Colab disconnects; each example is appended to a JSONL file and skipped on re-run.

## Known limitations

- **One-at-a-time occlusion misses interactions** (negation, "not good") and under-credits words that are repeated or redundant.
- **Deleting words makes ungrammatical text**; this is an out-of-distribution intervention, so a drop can partly reflect confusion rather than importance. The random-deletion baseline controls for this to a degree.
- **Cue admission is detected with a keyword rule** plus whether the cue was cited, so it can miss paraphrases. Plan: hand-check a sample of about 30 explanations and report agreement.
- **Small model, short inputs, two datasets.** Findings are about `gemma-3-4b-it` on SST-2 and German credit, not LLMs in general.
- **Numerical precision.** Gemma 3 may be unstable in fp16 on a T4; the scorer raises on non-finite logits and the notebook lets you switch dtype or model.

## Related work

- Turpin et al., 2023. *Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting.* <https://arxiv.org/abs/2305.04388> (biasing features that models do not mention; the inspiration for the planted-cue test)
- Madsen, Chandar, Reddy, 2024. *Are self-explanations from Large Language Models faithful?* Findings of ACL. <https://arxiv.org/abs/2401.07927>
- DeYoung et al., 2020. *ERASER: A Benchmark to Evaluate Rationalized NLP Models.* (comprehensiveness and sufficiency)
- Jacovi and Goldberg, 2020. *Towards Faithfully Interpretable NLP Systems.*

**What is new here:** one protocol applied across modalities, a causally verified planted-cue test, and a per-feature disclosure table on a high-stakes tabular task, all on an open Google model.

## License and model terms

Code: see `LICENSE`. Gemma weights are governed by Google's Gemma terms of use and are not redistributed here.
