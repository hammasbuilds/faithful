"""Fold three published faithfulness corpora into one file, at three grains.

The point of using three rather than one is that they disagree about what to
record, and each disagreement is useful:

* **AggreFact** gives the scale and the breadth — nine sub-benchmarks, two
  domains (CNN/DailyMail and BBC/XSum), one binary label per summary. This is
  what any headline number should be measured on.
* **FRANK** gives a *typed* error taxonomy over 2,246 of those summaries:
  entity, relation, circumstance, coreference, grammar, and `OutE` for content
  that is simply not in the article. A binary label cannot tell you which kind
  of wrongness a checker misses; this can.
* **Google's XSum annotations** give *span-level* marks on 500 articles, each
  tagged `extrinsic` (invented) or `intrinsic` (recombined). The finest grain,
  the smallest sample.

Binary at 60k, typed at 2k, spans at 500. A claim that survives all three is
worth more than a claim measured once.

Run once:

    python scripts/build_corpus.py
"""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
AGGREFACT = DATA / "aggrefact.csv"
FRANK = DATA / "frank.json"
XSUM_SPANS = DATA / "hallucination_annotations_xsum_summaries.csv"
XSUM_FACT = DATA / "xsum_factuality.csv"
XSUM_PARQUET = DATA / "xsum_test.parquet"
OUT = DATA / "corpus.json"

# FRANK scores each category 1.0 = no error of that kind. Below this, the
# annotators saw the error often enough to call it present.
PRESENT = 0.999

ERROR_KINDS = ("OutE", "EntE", "RelE", "CircE", "CorefE", "GramE", "LinkE", "Other")


def read_csv_lenient(path: Path) -> list[dict]:
    """Read a CSV, discarding a trailing partial row.

    The AggreFact file is fetched in byte ranges over a throttled link, so a
    build can legitimately run against a file whose last line is half-written.
    Dropping it is correct; silently dropping a *complete* row would not be.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    end = raw.rfind("\n")
    return list(csv.DictReader(io.StringIO(raw[: end if end > 0 else None])))


def frank_errors() -> dict[str, dict]:
    """hash -> the error kinds annotators marked, averaged over models."""
    if not FRANK.exists():
        return {}
    records = json.loads(FRANK.read_text(encoding="utf-8"))
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_hash[row["hash"]].append(row)

    out: dict[str, dict] = {}
    for key, rows in by_hash.items():
        kinds = sorted(
            kind for kind in ERROR_KINDS if any(float(r.get(kind, 1.0)) < PRESENT for r in rows)
        )
        out[key] = {
            "error_kinds": kinds,
            "factuality": sum(float(r["Factuality"]) for r in rows) / len(rows),
            "domain": rows[0]["dataset"],
        }
    return out


def xsum_spans() -> dict[str, list[dict]]:
    """bbcid -> span marks, each typed intrinsic or extrinsic."""
    if not XSUM_SPANS.exists():
        return {}
    marks: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv_lenient(XSUM_SPANS):
        kind = row.get("hallucination_type", "")
        if kind in {"intrinsic", "extrinsic"} and row.get("hallucinated_span"):
            marks[row["bbcid"]].append(
                {"type": kind, "span": row["hallucinated_span"], "system": row["system"]}
            )
    return marks


def xsum_graded(documents: dict[str, str], doc_key: dict[str, str]) -> list[dict]:
    """The Google XSum set at full size: majority vote plus typed spans.

    Three workers judged each summary and separately marked spans as intrinsic
    or extrinsic. This is the only grain that can separate *invented* content
    from *recombined* content, so it is kept whole rather than intersected with
    anything.
    """
    if not (XSUM_FACT.exists() and XSUM_PARQUET.exists()):
        return []
    import pyarrow.parquet as pq  # noqa: PLC0415

    votes: dict[tuple[str, str], list[str]] = defaultdict(list)
    text: dict[tuple[str, str], str] = {}
    for row in read_csv_lenient(XSUM_FACT):
        if row["is_factual"] in {"yes", "no"}:
            votes[(row["bbcid"], row["system"])].append(row["is_factual"])
            text[(row["bbcid"], row["system"])] = row["summary"]

    marks: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in read_csv_lenient(XSUM_SPANS):
        kind = row.get("hallucination_type", "")
        if kind in {"intrinsic", "extrinsic"} and row.get("hallucinated_span"):
            marks[(row["bbcid"], row["system"])].append(
                {"type": kind, "span": row["hallucinated_span"]}
            )

    table = pq.read_table(XSUM_PARQUET)
    articles = dict(
        zip(
            (str(x) for x in table.column("id").to_pylist()),
            table.column("document").to_pylist(),
            strict=True,
        )
    )

    out = []
    for (bbcid, system), summary in sorted(text.items()):
        article = articles.get(bbcid)
        if not article:
            continue
        if article not in doc_key:
            doc_key[article] = str(len(documents))
            documents[doc_key[article]] = article
        cast = votes[(bbcid, system)]
        yes = sum(1 for v in cast if v == "yes")
        out.append(
            {
                "id": bbcid,
                "doc": doc_key[article],
                "system": system,
                "summary": summary,
                "votes": cast,
                "faithful": yes > len(cast) - yes,
                "spans": marks.get((bbcid, system), []),
            }
        )
    return out


def main() -> None:
    if not AGGREFACT.exists():
        raise SystemExit(f"{AGGREFACT} is missing — fetch it first")

    typed = frank_errors()
    spans = xsum_spans()

    # Articles are stored once and referenced. Roughly three summaries are
    # judged per article across the benchmarks, so repeating the text inline
    # tripled the file for nothing.
    documents: dict[str, str] = {}
    doc_key: dict[str, str] = {}

    items = []
    for row in read_csv_lenient(AGGREFACT):
        label = row.get("label", "")
        if label not in {"0", "1"} or not row.get("doc") or not row.get("summary"):
            continue
        key = row["id"]
        extra = typed.get(key, {})

        text = row["doc"]
        if text not in doc_key:
            doc_key[text] = str(len(documents))
            documents[doc_key[text]] = text

        items.append(
            {
                "id": key,
                "doc": doc_key[text],
                "benchmark": row["dataset"],  # XSumFaith, FRANK, SummEval, ...
                "domain": row["origin"],  # cnndm | xsum
                "split": row["cut"],  # val | test
                "model": row["model_name"],
                "summary": row["summary"],
                "faithful": label == "1",
                "error_kinds": extra.get("error_kinds", []),
                "spans": spans.get(key, []) if row["dataset"] == "XSumFaith" else [],
            }
        )

    # The span grain is kept as its own set, at its own full size.
    #
    # Filtering it down to the rows that also appear in AggreFact was a mistake
    # worth recording: it left three intrinsic-only summaries, and no claim can
    # rest on three. The whole point of a three-grain corpus is that the finest
    # grain answers a question the coarsest cannot, which it can only do if it
    # is allowed to keep every item it has.
    graded = xsum_graded(documents, doc_key)

    OUT.write_text(
        json.dumps({"documents": documents, "items": items, "graded": graded}, indent=0),
        encoding="utf-8",
    )
    print(f"{len(graded):,} span-annotated summaries (XSum, own grain)")

    print(f"{len(documents):,} distinct articles")
    typed_n = sum(1 for i in items if i["error_kinds"])
    span_n = sum(1 for i in items if i["spans"])
    print(f"{len(items):,} labelled summaries -> {OUT}")
    print(f"  with typed errors : {typed_n:,}")
    print(f"  with span marks   : {span_n:,}")
    print(f"  {OUT.stat().st_size / 1_048_576:.1f} MB")


if __name__ == "__main__":
    main()
