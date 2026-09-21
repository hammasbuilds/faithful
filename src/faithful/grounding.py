"""Does every claim in the summary trace back to the article?

This is the deterministic half of the product, and it uses no model at all. A
summary is *ungrounded* when it contains content the article does not: a number
that appears nowhere, or content words the source never used.

The point of writing it is not that it is better than a model. It is to
establish how much of the problem needs one. If a regex catches most
hallucinations, a 14B judging summaries is an expensive way to lose.

The prediction, made before measuring: this should do well on **extrinsic**
hallucination, where the summary introduces content that is simply absent, and
badly on **intrinsic**, where every word came from the article and only the
relation between them is invented. "The Frenchman scored" is ungrounded if the
article never says French, and perfectly grounded if it calls someone else
French.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9][a-z0-9'’-]*")
_NUMBER = re.compile(r"\b\d[\d,.]*\b")

# Written-out numbers are numbers. A summary saying "three people died" when the
# article says two is exactly the failure this is looking for, and it never
# reaches the digit pattern.
WORD_NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "hundred": "100", "thousand": "1000",
}

STOP = frozenset(
    """
    a an and are as at be been being but by for from had has have he her his
    if in into is it its of on or she that the their them they this to was
    were will with who which what when where would could should may might
    must not no than then there these those you your we our us i
    """.split()
)

MIN_STEM = 3


def _stem(token: str) -> str:
    """A crude suffix strip, deliberately not a real stemmer.

    Morphology is the main source of false alarms: an article saying "pupil"
    and a summary saying "pupils" is not a hallucination. Anything cleverer
    would need a dictionary, and the point of this module is to be cheap.
    """
    for suffix in ("'s", "’s", "ing", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM:
            base = token[: -len(suffix)]
            return base + "y" if suffix == "ies" else base
    return token


def words(text: str) -> list[str]:
    """Content words, trimmed of edge punctuation.

    The trim matters more than it looks. `pupils'` keeps its apostrophe through
    the word pattern, no suffix rule strips a bare quote, and the token never
    matches `pupil` in the article — so a possessive plural was being reported
    as invented content.

    Written-out numbers are excluded here because `numbers()` already owns
    them. Counting "two" as both an ungrounded word and an ungrounded number
    doubled the score of every numeric mismatch.
    """
    out = []
    for raw in _WORD.findall(text.lower()):
        word = raw.strip("'’-")
        if word and word not in STOP and word not in WORD_NUMBERS and len(word) > 1:
            out.append(word)
    return out


def tokens(text: str) -> set[str]:
    """Content-word stems."""
    return {_stem(w) for w in words(text)}


def numbers(text: str) -> set[str]:
    """Every quantity, digits and words alike, normalised."""
    lowered = text.lower()
    found = {n.replace(",", "").rstrip(".") for n in _NUMBER.findall(lowered)}
    found |= {
        WORD_NUMBERS[w] for w in _WORD.findall(lowered) if w in WORD_NUMBERS
    }
    return {n for n in found if n}


@dataclass
class Check:
    """What the article failed to support."""

    loose_tokens: set[str] = field(default_factory=set)
    loose_numbers: set[str] = field(default_factory=set)

    @property
    def ungrounded(self) -> int:
        return len(self.loose_tokens) + len(self.loose_numbers)

    @property
    def evidence(self) -> str:
        parts = []
        if self.loose_numbers:
            parts.append("numbers not in the article: " + ", ".join(sorted(self.loose_numbers)))
        if self.loose_tokens:
            parts.append("words not in the article: " + ", ".join(sorted(self.loose_tokens)))
        return "; ".join(parts) or "every content word traces to the article"


def check(summary: str, document: str) -> Check:
    """Content in the summary that the article does not contain.

    Matching is done on stems; the *evidence* reports the words as the summary
    actually wrote them. Telling someone their summary contains "flood" when it
    says "flooding" is an accusation about a word they did not use, and
    evidence a reader cannot find is not evidence.
    """
    source_tokens = tokens(document)
    surface: dict[str, str] = {}
    for word in words(summary):
        surface.setdefault(_stem(word), word)

    loose = {stem for stem in surface if stem not in source_tokens}
    return Check(
        loose_tokens={surface[stem] for stem in loose},
        loose_numbers=numbers(summary) - numbers(document),
    )


def ungrounded(summary: str, document: str, threshold: int = 1) -> bool:
    """Flag the summary when it carries ``threshold`` or more loose items.

    The threshold is the whole tuning knob, and its behaviour is the finding:
    at 1 it flags nearly everything, at 4 it flags almost nothing, and the
    interesting question is whether any setting separates faithful summaries
    from unfaithful ones on real data.
    """
    if threshold < 1:
        raise ValueError("a threshold below 1 flags summaries with no evidence at all")
    return check(summary, document).ungrounded >= threshold
