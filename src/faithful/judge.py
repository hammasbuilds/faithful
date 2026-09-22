"""Asking a 14B instruct model whether a summary is true to its article.

The model is required to do two things: return a verdict, and **quote the words
it objects to**. The quote is the interesting part, because it is checkable
without any judgement at all — either that string is in the summary or it is
not.

That turns a vague question into a measurable one. A detector that says
"unfaithful" and then quotes a phrase the summary never contained has
hallucinated its own evidence, and a hallucination detector that hallucinates
is worth knowing about. `Verdict.quote_is_real` is the whole reason this module
asks for a quote rather than a score.

Nothing here reaches a real model during tests: `Recorded` raises on any prompt
it was not given an answer for, so a test that accidentally depends on a live
model fails loudly instead of quietly costing an hour.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

OLLAMA = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b-instruct"

PROMPT = """You are checking whether a summary is faithful to its article.

A summary is UNFAITHFUL if it states anything the article does not support —
an invented detail, a changed number, a relationship the article never draws,
or a person doing something the article attributes to someone else.
A summary is FAITHFUL if everything it says is supported, even if it leaves
most of the article out. Leaving things out is not unfaithful.

ARTICLE:
{document}

SUMMARY:
{summary}

Answer with exactly two lines and nothing else:
VERDICT: FAITHFUL or UNFAITHFUL
QUOTE: the exact words from the SUMMARY that are unsupported, or NONE
"""

FAITHFUL = "FAITHFUL"
UNFAITHFUL = "UNFAITHFUL"
NONE = "NONE"

_VERDICT = re.compile(r"VERDICT:\s*(FAITHFUL|UNFAITHFUL)", re.I)
_QUOTE = re.compile(r"QUOTE:\s*(.+)", re.I | re.S)

# The model sometimes restates the instruction before answering it:
#   QUOTE: the exact words from the SUMMARY that are unsupported, or NONE: NONE (...)
# Taking the line whole reports the prompt's own wording as the model's quote,
# which is then of course not in the summary — so an abstention is scored as
# invented evidence, the one number this module exists to measure.
_ECHO = re.compile(r"^the exact words from the summary\b[^:]*:\s*", re.I)

# "NONE (but the summary is inaccurate ...)" — the structured answer is NONE and
# the rest is commentary. The bracket is required: a summary may genuinely open
# a sentence with "None of the passengers survived", and collapsing that to an
# abstention would throw away a real quote.
_NONE_THEN_ASIDE = re.compile(r"^none\s*[(\[]", re.I)


class ModelUnavailableError(RuntimeError):
    """The model could not be reached."""


class Model(Protocol):
    def __call__(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class Verdict:
    """What the model said, and whether its evidence exists."""

    faithful: bool
    quote: str
    summary: str
    raw: str

    @property
    def quote_is_real(self) -> bool:
        """Whether the quoted words actually appear in the summary.

        Checked on a normalised copy: the model routinely re-punctuates or
        re-cases what it quotes, and calling that an invention would be
        measuring the model's typography rather than its reasoning.
        """
        if not self.quote or self.quote.upper() == NONE:
            return True
        return _normalise(self.quote) in _normalise(self.summary)

    @property
    def abstained(self) -> bool:
        """Called it unfaithful but would not say which words."""
        return not self.faithful and (not self.quote or self.quote.upper() == NONE)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def parse(raw: str, summary: str) -> Verdict:
    """Read the model's two lines. An unparseable answer counts as faithful.

    That default is deliberate and it is the conservative one: this measures a
    detector, and a detector that produced no usable finding did not find
    anything. Treating garbage as a positive would inflate recall with noise.
    """
    found = _VERDICT.search(raw)
    faithful = not (found and found.group(1).upper() == UNFAITHFUL)

    quote = ""
    marked = _QUOTE.search(raw)
    if marked:
        quote = marked.group(1).split("\n")[0].strip()
        quote = _ECHO.sub("", quote).strip()
        if _NONE_THEN_ASIDE.match(quote):
            quote = NONE
        # The model sometimes appends its reasoning after the quoted span:
        #   QUOTE: "dpp accepts 77 allegations" (the correct name is "Ratcliffe")
        # Taking the whole line then reports a quote that is not in the summary
        # and scores the model as having invented evidence, when it quoted
        # correctly and explained itself afterwards. If the line opens with a
        # quotation mark, keep only what is inside it.
        if quote[:1] in ('"', "'", "“"):
            closing = {'"': '"', "'": "'", "“": "”"}[quote[0]]
            end = quote.find(closing, 1)
            if end > 0:
                quote = quote[1:end]
        quote = quote.strip().strip('"').strip()

    return Verdict(faithful=faithful, quote=quote, summary=summary, raw=raw)


def ollama(
    model: str = DEFAULT_MODEL,
    timeout: float = 900.0,
    keep_alive: str = "2h",
) -> Model:
    """A live local model. Deterministic settings, so a rerun reproduces.

    The timeout is generous because it has to cover a cold start: 9 GB of
    weights onto a 16 GB card takes several minutes, and the first call pays
    for all of it. A 120-second timeout — the obvious value — fails every time
    on the first request and never on any request after it, which reads like a
    flaky server rather than a model that is still loading.

    `keep_alive` holds the model in VRAM between calls. Without it ollama
    unloads after five idle minutes, so a long run pays the cold start again
    every time a corpus batch takes a while to prepare.
    """

    def call(prompt: str) -> str:
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": keep_alive,
                "options": {"temperature": 0, "seed": 7, "num_predict": 120},
            }
        ).encode()
        request = urllib.request.Request(
            OLLAMA, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())["response"]
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelUnavailableError(f"{model} at {OLLAMA}: {exc}") from exc

    return call


class Recorded:
    """A scripted model. Raises on anything it was not told about."""

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    def __call__(self, prompt: str) -> str:
        for key, answer in self._answers.items():
            if key in prompt:
                return answer
        raise ModelUnavailableError(
            "Recorded model got a prompt it has no answer for. A test must not "
            "reach a real model by accident."
        )


def ask(model: Model, document: str, summary: str, max_chars: int = 6000) -> Verdict:
    """One faithfulness judgement.

    Long articles are cut from the end. That is a real limitation and it is
    stated rather than hidden: a claim supported only by the last paragraph of
    a very long article will be judged unsupported.
    """
    prompt = PROMPT.format(document=document[:max_chars], summary=summary)
    return parse(model(prompt), summary)
