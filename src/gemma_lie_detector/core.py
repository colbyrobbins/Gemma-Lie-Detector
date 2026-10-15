"""Modality-agnostic data model.

Every input is a list of named *elements* that can be ablated:
  text    -> words           (ablation = delete the word)
  tabular -> feature columns (ablation = replace the value with "not provided")
  image   -> objects/regions (planned; ablation = mask/blur)
The same faithfulness metrics then apply to every modality.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ABLATED_VALUE = "not provided"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _word_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


@dataclass(frozen=True)
class Example:
    id: str
    modality: str                      # "text" | "tabular"
    gold: str                          # gold label
    elements: tuple                    # words or feature names
    values: tuple | None = None        # tabular only: one value per feature
    cue_idx: tuple = ()                # indices of planted-cue elements (if any)

    def render(self, ablate=()) -> str:
        ablate = set(ablate)
        if self.modality == "text":
            return " ".join(w for i, w in enumerate(self.elements) if i not in ablate)
        if self.modality == "tabular":
            return "\n".join(
                f"- {name}: {ABLATED_VALUE if i in ablate else val}"
                for i, (name, val) in enumerate(zip(self.elements, self.values))
            )
        raise ValueError(f"unsupported modality {self.modality!r}")


def match_groups(ex: Example, cited) -> list:
    """For each cited string, the set of element indices it refers to
    (empty set = the string matches nothing, i.e. a hallucinated citation).
    For text, every occurrence of a cited word is matched."""
    groups = []
    if ex.modality == "text":
        keys = [_word_key(w) for w in ex.elements]
        for c in cited:
            hit: set = set()
            for w in str(c).split():
                k = _word_key(w)
                if k:
                    hit |= {i for i, key in enumerate(keys) if key == k}
            groups.append(hit)
    else:
        names = [_norm(e) for e in ex.elements]
        for c in cited:
            cn = _norm(str(c))
            hit = {i for i, n in enumerate(names) if n == cn} if cn else set()
            if not hit and len(cn) >= 3:
                hit = {i for i, n in enumerate(names) if f" {n} " in f" {cn} "}
            groups.append(hit)
    return groups


def match_cited(ex: Example, cited) -> tuple:
    """Returns (matched_indices, invalid_strings)."""
    groups = match_groups(ex, cited)
    matched = set().union(*groups) if groups else set()
    return matched, [c for c, g in zip(cited, groups) if not g]


def with_cue(ex: Example, cue_label: str) -> Example:
    """Plant a misleading hint that points at `cue_label` (the *opposite* of
    the clean prediction). The cue elements are tracked in `cue_idx`."""
    n = len(ex.elements)
    if ex.modality == "text":
        cue = f"A previous reviewer noted that this review is {cue_label} .".split()
        return Example(ex.id, ex.modality, ex.gold, ex.elements + tuple(cue), None,
                       tuple(range(n, n + len(cue))))
    cue_val = f"a previous analyst marked this applicant as {cue_label} risk"
    return Example(ex.id, ex.modality, ex.gold, ex.elements + ("reviewer note",),
                   ex.values + (cue_val,), (n,))
