"""Gemma wrapper. Exposes the two operations the experiments need:

    score(prompts)    -> (probs [N, n_labels], label_mass [N])
    generate(prompts) -> list[str]   (PREFIX + greedy continuation)

`probs` is the next-token distribution after PREFIX, restricted to the label
words and renormalised; `label_mass` is how much of the full-vocabulary
probability sat on those words (a sanity check: should be ~1).
"""
from __future__ import annotations

import numpy as np

from .prompts import PREFIX

GEMMA_FORMAT = "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


class GemmaScorer:
    def __init__(self, labels, model_id="google/gemma-3-4b-it", dtype="auto",
                 batch_size=16, device_map="auto"):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        if device_map == "auto" and torch.cuda.is_available():
            # Put the whole model on the GPU. With "auto", a second copy of the model (e.g. re-running
            # the load cell) silently offloads layers to CPU and makes everything very slow.
            device_map = {"": 0}
        self.labels = tuple(labels)
        self.batch_size = batch_size
        if dtype == "auto":  # T4 has no native bf16; Colab L4/A100 do.
            native_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            dtype = "bfloat16" if native_bf16 else "float16"
        self.dtype = dtype
        dt = getattr(torch, dtype)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tok = getattr(self.processor, "tokenizer", self.processor)
        self.tok.padding_side = "left"

        import transformers
        loaders = [getattr(transformers, n) for n in ("AutoModelForImageTextToText", "AutoModelForCausalLM")
                   if hasattr(transformers, n)]
        last_err = None
        for cls in loaders:
            for kw in ("dtype", "torch_dtype"):
                try:
                    self.model = cls.from_pretrained(model_id, device_map=device_map, **{kw: dt}).eval()
                    break
                except (TypeError, ValueError, KeyError) as e:
                    last_err = e
            else:
                continue
            break
        else:
            raise RuntimeError(f"could not load {model_id}: {last_err}")

        self.label_ids = []
        for lab in self.labels:
            ids = sorted({self.tok.encode(v, add_special_tokens=False)[0] for v in (lab, lab.capitalize())})
            self.label_ids.append(ids)
        flat = [i for ids in self.label_ids for i in ids]
        assert len(flat) == len(set(flat)), "label words share a first token; pick different label words"

    def _encode(self, prompts):
        texts = [GEMMA_FORMAT.format(prompt=p) + PREFIX for p in prompts]
        enc = self.processor(text=texts, return_tensors="pt", padding=True, add_special_tokens=False)
        return enc.to(self.model.device)

    def score(self, prompts):
        torch = self.torch
        probs, mass = [], []
        with torch.inference_mode():
            for chunk in _chunks(list(prompts), self.batch_size):
                enc = self._encode(chunk)
                try:
                    out = self.model(**enc, logits_to_keep=1)
                except TypeError:
                    out = self.model(**enc)
                logits = out.logits[:, -1, :].float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError(
                        f"non-finite logits with dtype={self.dtype}; reload with dtype='bfloat16' "
                        "(Colab L4/A100) or a smaller model such as google/gemma-3-1b-it")
                logp = torch.log_softmax(logits, dim=-1)
                lab_lp = torch.stack([torch.logsumexp(logp[:, ids], dim=-1) for ids in self.label_ids], dim=-1)
                mass.append(lab_lp.exp().sum(-1).cpu().numpy())
                probs.append(torch.softmax(lab_lp, dim=-1).cpu().numpy())
        return np.concatenate(probs), np.concatenate(mass)

    def generate(self, prompts, max_new_tokens=96):
        torch = self.torch
        outs = []
        with torch.inference_mode():
            for chunk in _chunks(list(prompts), self.batch_size):
                enc = self._encode(chunk)
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
                new = gen[:, enc["input_ids"].shape[1]:]
                outs += [PREFIX + self.tok.decode(row, skip_special_tokens=True) for row in new]
        return outs
