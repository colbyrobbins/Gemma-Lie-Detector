"""Integrated gradients (Sundararajan et al., 2017) for Gemma's decision.

Occlusion asks "what happens if this word is missing?". Integrated gradients (IG) instead looks
inside the model: it walks from a blank input to the real review in small steps and adds up how
the decision's margin responds to each word's embedding along the way. The two methods fail in
different ways, so agreement between them (and with Gemma's own citations) is informative.

The file has three layers:
  integrated_gradients()   model-agnostic numeric core (numpy), tested on toy functions
  word_char_spans(), tokens_to_words()   text bookkeeping that maps tokens back to words, tested
  GemmaAttributor          the torch adapter that plugs Gemma into the numeric core

Sign convention matches occlusion: a positive score means the word supports Gemma's decision.
"""
from __future__ import annotations

import numpy as np

from .model import GEMMA_FORMAT
from .prompts import PREFIX


def integrated_gradients(value_and_grad, x, baseline, steps=32, chunk=8):
    """Integrated gradients of a scalar function along the straight path baseline -> x.

    `value_and_grad(points)` takes a batch of inputs `[B, *x.shape]` and returns
    `(values [B], gradients [B, *x.shape])`. The integral is approximated with the midpoint rule
    over `steps` points, evaluated `chunk` at a time to bound memory. Returns an array shaped like
    `x`: each entry is that input value's share of f(x) - f(baseline).
    """
    x, baseline = np.asarray(x, float), np.asarray(baseline, float)
    alphas = (np.arange(steps) + 0.5) / steps
    grad_sum = np.zeros_like(x)
    for start in range(0, steps, chunk):
        a = alphas[start:start + chunk].reshape((-1,) + (1,) * x.ndim)
        _, grads = value_and_grad(baseline + a * (x - baseline))
        grad_sum += np.asarray(grads, float).sum(axis=0)
    return (x - baseline) * grad_sum / steps


def completeness_error(attr, f_x, f_baseline):
    """Relative gap between the attributions' sum and f(x) - f(baseline).

    IG should satisfy sum(attr) = f(x) - f(baseline) ("completeness"). A large gap means too few
    steps or numerical noise (e.g. bfloat16 gradients), so the attribution should not be trusted.
    """
    target = f_x - f_baseline
    return float(abs(np.sum(attr) - target) / max(abs(target), 1e-9))


def word_char_spans(words, start=0):
    """(start, end) character offsets of each word when the words are joined by single spaces."""
    spans, pos = [], start
    for w in words:
        spans.append((pos, pos + len(w)))
        pos += len(w) + 1
    return spans


def tokens_to_words(token_offsets, word_spans):
    """For each token, the index of the word it overlaps most, or -1 for non-word tokens.

    Tokenizers disagree on whether a token's offsets include the leading space, so overlap
    (rather than exact equality) is used to assign tokens to words.
    """
    out = []
    for t0, t1 in token_offsets:
        best, best_overlap = -1, 0
        for i, (w0, w1) in enumerate(word_spans):
            overlap = min(t1, w1) - max(t0, w0)
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        out.append(best)
    return out


class GemmaAttributor:
    """Integrated gradients of Gemma's decision margin with respect to the review's token embeddings.

    Only the review's tokens move along the path; the instructions around them stay fixed. Token
    scores are summed into word scores so they line up with the occlusion and citation results.
    Needs the GPU-loaded `GemmaScorer`; it reuses that scorer's model and tokenizer.
    """

    def __init__(self, scorer, steps=64, chunk=4, baselines=("zero",)):
        """`baselines` picks what "blank" means: "zero" (all-zero embeddings) or "pad" (the padding token's embedding).

        Defaults reflect smoke tests on Gemma 3 4B in bfloat16: the zero baseline passed the completeness
        check at 64 steps, while the pad baseline failed it and got worse with more steps (noisy gradients).
        """
        self.scorer, self.steps, self.chunk = scorer, steps, chunk
        self.baselines = tuple(baselines)
        scorer.model.requires_grad_(False)   # gradients are needed for the inputs only

    def _margin(self, embeds, mask, target):
        """Margin (log-odds) of label `target`, computed from input embeddings. Matches metrics.margin."""
        s, torch = self.scorer, self.scorer.torch
        try:
            out = s.model(inputs_embeds=embeds, attention_mask=mask, logits_to_keep=1)
        except TypeError:
            out = s.model(inputs_embeds=embeds, attention_mask=mask)
        logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        lab = torch.stack([torch.logsumexp(logp[:, ids], dim=-1) for ids in s.label_ids], dim=-1)
        lab = torch.log_softmax(lab, dim=-1)
        others = torch.cat([lab[:, :target], lab[:, target + 1:]], dim=-1)
        return lab[:, target] - torch.logsumexp(others, dim=-1)

    def attribute(self, task, ex, target, baseline="pad"):
        """Word-level IG scores for label index `target`. Returns (scores [n_words], completeness_error)."""
        s, torch = self.scorer, self.scorer.torch
        prompt = task.prompt(ex.render())
        text = GEMMA_FORMAT.format(prompt=prompt) + PREFIX

        # Tokenize with character offsets so every token can be traced back to a review word.
        enc = s.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        if ids != s._encode([prompt])["input_ids"][0].tolist():
            raise RuntimeError("tokenization differs from the scorer's; attributions would not match decisions")
        review_start = text.index(f"{task.input_header}:\n") + len(task.input_header) + 2
        tok_word = tokens_to_words(enc["offset_mapping"], word_char_spans(ex.elements, review_start))
        positions = [t for t, w in enumerate(tok_word) if w >= 0]
        pos_t = torch.tensor(positions, device=s.model.device)

        input_ids = torch.tensor([ids], device=s.model.device)
        mask = torch.ones_like(input_ids)
        with torch.no_grad():
            emb = s.model.get_input_embeddings()(input_ids)       # [1, T, D], includes Gemma's scaling
            pad_vec = s.model.get_input_embeddings()(torch.tensor([[s.tok.pad_token_id]], device=emb.device))[0, 0]
        x = emb[0, positions].float().cpu().numpy()               # the review's real embeddings [P, D]
        if baseline == "pad":
            base = np.tile(pad_vec.float().cpu().numpy(), (len(positions), 1))
        elif baseline == "zero":
            base = np.zeros_like(x)
        else:
            raise ValueError(f"unknown baseline {baseline!r}")

        def with_points(points):
            """Full embedding sequences with the review positions replaced by `points` [B, P, D]."""
            full = emb.expand(len(points), -1, -1).clone()
            full[:, pos_t] = points.to(emb.dtype)
            return full, mask.expand(len(points), -1)

        def value_and_grad(points):
            pts = torch.tensor(points, dtype=torch.float32, device=emb.device, requires_grad=True)
            margins = self._margin(*with_points(pts), target)
            margins.sum().backward()
            return margins.detach().float().cpu().numpy(), pts.grad.cpu().numpy()

        def value(point):
            with torch.no_grad():
                pts = torch.tensor(point[None], dtype=torch.float32, device=emb.device)
                return float(self._margin(*with_points(pts), target)[0])

        token_attr = integrated_gradients(value_and_grad, x, base, self.steps, self.chunk).sum(axis=-1)
        err = completeness_error(token_attr, value(x), value(base))

        word_attr = np.zeros(len(ex.elements))
        np.add.at(word_attr, np.asarray(tok_word)[positions], token_attr)   # tokens -> words
        return word_attr, err
