"""Human-labelled summarisation faithfulness, at three grains.

`data/corpus.json` merges three published corpora. They were chosen because
they disagree about what is worth recording, and each disagreement buys
something:

======================  ========  =====================================
Grain                   Scale     What it can answer
======================  ========  =====================================
Binary (AggreFact)      ~60,000   Does a checker work at all, on two
                                  domains and nine sub-benchmarks
Typed (FRANK)           ~2,200    *Which kind* of error does it miss —
                                  entity, relation, circumstance, or
                                  content that is simply not there
Spans (Google/XSum)     ~500      Where exactly, and is the invention
                                  ``extrinsic`` or ``intrinsic``
======================  ========  =====================================

The three-grain shape is the point. A binary label tells you a checker scored
0.6; it cannot tell you that it scored 0.65 on invented content and *below
chance* on recombined content, which is the actual finding here and is only
visible at the second and third grains.

Domains are CNN/DailyMail (multi-sentence, extractive-ish) and BBC/XSum
(single-sentence, aggressively abstractive). Any number quoted on one alone is
a fact about that one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

DATA = Path(__file__).resolve().parents[2] / "data"
CORPUS = DATA / "corpus.json"

EXTRINSIC = "extrinsic"
INTRINSIC = "intrinsic"

CNNDM = "cnndm"
XSUM = "xsum"

# FRANK's category for content that appears nowhere in the article. The
# taxonomy's own name for what the span annotations call extrinsic.
OUT_OF_ARTICLE = "OutE"


class Span(NamedTuple):
    """One annotated stretch of a summary, and what kind of error it is.

    A NamedTuple rather than a dict so the items that hold it stay hashable —
    a frozen dataclass carrying a list of dicts cannot go in a set, which is
    exactly what any measurement wants to do with it.
    """

    type: str
    span: str


class CorpusMissingError(FileNotFoundError):
    """The merged corpus is not on disk."""


@dataclass(frozen=True)
class Item:
    """One summary, its article, and what the annotators said."""

    id: str
    benchmark: str          # XSumFaith, FRANK, SummEval, FactCC, Polytope, ...
    domain: str             # cnndm | xsum
    split: str              # val | test
    model: str              # the system that wrote the summary
    document: str
    summary: str
    faithful: bool
    error_kinds: tuple[str, ...]
    spans: tuple[Span, ...]

    @property
    def typed(self) -> bool:
        """Carries FRANK's error taxonomy."""
        return bool(self.error_kinds)

    @property
    def marked(self) -> bool:
        """Carries span-level annotation."""
        return bool(self.spans)

    @property
    def kinds(self) -> set[str]:
        return {s.type for s in self.spans}

    @property
    def extrinsic_only(self) -> bool:
        return self.kinds == {EXTRINSIC}

    @property
    def intrinsic_only(self) -> bool:
        return self.kinds == {INTRINSIC}

    @property
    def out_of_article(self) -> bool:
        """FRANK marked content that is not in the article at all."""
        return OUT_OF_ARTICLE in self.error_kinds


@lru_cache(maxsize=1)
def load(path: str | None = None) -> tuple[Item, ...]:
    target = Path(path) if path else CORPUS
    if not target.exists():
        raise CorpusMissingError(
            f"{target} is missing. Run scripts/build_corpus.py once the "
            "AggreFact, FRANK and XSum annotation files are in data/."
        )
    raw = json.loads(target.read_text(encoding="utf-8"))
    documents = raw["documents"]
    return tuple(
        Item(
            id=row["id"],
            benchmark=row["benchmark"],
            domain=row["domain"],
            split=row["split"],
            model=row["model"],
            document=documents[row["doc"]],
            summary=row["summary"],
            faithful=row["faithful"],
            error_kinds=tuple(row["error_kinds"]),
            spans=tuple(Span(s["type"], s["span"]) for s in row["spans"]),
        )
        for row in raw["items"]
    )


def by_domain(domain: str, path: str | None = None) -> list[Item]:
    return [i for i in load(path) if i.domain == domain]


def by_benchmark(name: str, path: str | None = None) -> list[Item]:
    return [i for i in load(path) if i.benchmark == name]


def typed(path: str | None = None) -> list[Item]:
    """Items carrying FRANK's error taxonomy."""
    return [i for i in load(path) if i.typed]


def marked(path: str | None = None) -> list[Item]:
    """Items carrying span-level extrinsic/intrinsic marks."""
    return [i for i in load(path) if i.marked]


@dataclass(frozen=True)
class Graded:
    """An XSum summary with three worker votes and typed spans.

    Separate from `Item` because it answers a different question. `Item` says
    whether a summary is faithful; `Graded` says *how* it is unfaithful, and
    that distinction is the only thing that can tell an invented fact from a
    recombined one.
    """

    id: str
    system: str
    document: str
    summary: str
    votes: tuple[str, ...]
    faithful: bool
    spans: tuple[Span, ...]

    @property
    def unanimous(self) -> bool:
        return bool(self.votes) and len(set(self.votes)) == 1

    @property
    def kinds(self) -> set[str]:
        return {s.type for s in self.spans}

    @property
    def extrinsic_only(self) -> bool:
        return self.kinds == {EXTRINSIC}

    @property
    def intrinsic_only(self) -> bool:
        return self.kinds == {INTRINSIC}


@lru_cache(maxsize=1)
def graded(path: str | None = None) -> tuple[Graded, ...]:
    """The XSum span-annotated set, at full size."""
    target = Path(path) if path else CORPUS
    if not target.exists():
        raise CorpusMissingError(f"{target} is missing. Run scripts/build_corpus.py.")
    raw = json.loads(target.read_text(encoding="utf-8"))
    documents = raw["documents"]
    return tuple(
        Graded(
            id=row["id"],
            system=row["system"],
            document=documents[row["doc"]],
            summary=row["summary"],
            votes=tuple(row["votes"]),
            faithful=row["faithful"],
            spans=tuple(Span(s["type"], s["span"]) for s in row["spans"]),
        )
        for row in raw.get("graded", [])
    )
