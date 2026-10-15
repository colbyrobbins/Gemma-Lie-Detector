"""One prompt format is used for BOTH the decision and the explanation.

The model's reply is forced to start with PREFIX (`{"label": "`). We read the
next-token distribution over the label words to get the *decision* (a
probability we can measure under ablation), and we let greedy decoding
continue the same JSON to get the *explanation*. So the decision being
explained is exactly the one the explanation is written next to.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

PREFIX = '{"label": "'


@dataclass(frozen=True)
class Task:
    name: str
    instruction: str
    input_header: str
    labels: tuple
    cite_noun: str

    def prompt(self, rendered: str) -> str:
        opts = ", ".join(self.labels)
        return (
            f"{self.instruction}\n\n{self.input_header}:\n{rendered}\n\n"
            "Respond with ONLY a JSON object in exactly this format:\n"
            f'{{"label": "<one of: {opts}>", "cited": ["<{self.cite_noun}>", "..."], '
            '"reason": "<one short sentence>"}\n'
            f'Rules: "label" must be lowercase. "cited" lists up to 3 {self.cite_noun}, '
            "copied exactly from the input, that most influenced your decision."
        )

    def input_segment(self, prompt: str) -> str:
        """Recover the rendered input from a prompt (used by test mocks)."""
        head = f"{self.input_header}:\n"
        return prompt.split(head, 1)[1].split("\n\nRespond with ONLY", 1)[0]


SENTIMENT = Task("sentiment", "Classify the sentiment of this movie review.",
                 "Review", ("positive", "negative"), "words")
CREDIT = Task("credit", "You are assisting a bank. Decide whether this loan applicant "
              "is a good or bad credit risk.", "Applicant", ("good", "bad"), "feature names")


def parse_explanation(raw: str):
    """Parse the model's JSON. Returns None if nothing usable was produced."""
    raw = raw.replace("```json", "").replace("```", "")
    m = re.search(r"\{.*?\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            cited = d.get("cited", [])
            cited = [cited] if isinstance(cited, str) else list(cited)
            return {"label": str(d.get("label", "")).strip().lower(),
                    "cited": [str(c) for c in cited][:3],
                    "reason": str(d.get("reason", ""))}
        except (json.JSONDecodeError, AttributeError):
            pass
    # truncated / malformed JSON: salvage what we can
    lab = re.search(r'"label"\s*:\s*"([^"]*)"', raw)
    lst = re.search(r'"cited"\s*:\s*\[(.*?)\]', raw, re.S)
    if lab and lst:
        cited = re.findall(r'"([^"]+)"', lst.group(1))
        reason = re.search(r'"reason"\s*:\s*"([^"]*)', raw)
        return {"label": lab.group(1).strip().lower(), "cited": cited[:3],
                "reason": reason.group(1) if reason else ""}
    return None
