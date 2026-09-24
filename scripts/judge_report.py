"""What the 14B judge did, read off the recorded answers.

    python scripts/judge_report.py

Reads `data/judge_runs.jsonl` and prints the tables. It works on a partial run —
the file is appended to as answers arrive — and says how far through it is, so
a run that is still going can be inspected without stopping it.

The question this half exists to answer is not "is a 14B better than counting
words". It is **whether the detector's own evidence is real**. The prompt makes
the model quote the exact words it objects to, and that quote is checkable
without any judgement at all: either the string is in the summary or it is not.
A hallucination detector that hallucinates its evidence is worth knowing about,
and nothing else in the pipeline would have caught it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path(__file__).resolve().parents[1] / "data"
RUNS = DATA / "judge_runs.jsonl"


def planned() -> int:
    """How many answers this run is aiming at, read from the corpus.

    This was hard-coded to 600 - the sample size of the first run. When the run
    was widened to the whole corpus the report went on dividing by 600, so it
    announced "1,675 of 600 planned (279%)" and declared itself finished while
    eight thousand items were still queued. A constant that describes one run
    is wrong for the next one.

    Falls back to the number of answers on disk when the corpus is not
    available, which makes the report readable from the jsonl alone.
    """
    try:
        # Imported here, not at module scope: the report is meant to be readable
        # from data/judge_runs.jsonl alone, on a machine where the corpus has
        # not been fetched. A top-level import would make that an ImportError
        # instead of the "completeness unknown" line below.
        from faithful import corpus  # noqa: PLC0415

        items = corpus.load()
        # Distinct summaries, not rows: the report deduplicates, so the target
        # has to be deduplicated the same way. Comparing 9,979 answers against
        # 10,066 items made a finished run report as permanently PARTIAL.
        keys = {
            (
                i.benchmark,
                i.domain,
                hashlib.blake2b(i.summary.encode("utf-8"), digest_size=4).hexdigest(),
            )
            for i in items
        }
        return len(keys)
    except Exception:
        return 0


def rule(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def load() -> tuple[list[dict], int]:
    """Every answer, one per distinct summary. Returns (rows, duplicates_dropped).

    The file can hold the same id more than once, and it is not a bug in the
    runner: `item_id` is (benchmark, domain, hash-of-summary), and the corpus
    contains 73 summaries that appear 2-3 times inside the same benchmark and
    domain - 160 rows for 73 distinct texts, identical strings, not hash
    collisions. Each copy is a separate corpus item, so each gets judged and
    written under the shared id.

    Counting all of them would weight those 73 summaries two or three times in
    every number below. It is only 0.7% of the file, which is exactly the size
    of error that never gets checked and quietly moves a third decimal place.
    The duplication is reported rather than silently collapsed, because a
    faithfulness benchmark containing duplicate summaries is worth knowing
    about on its own.
    """
    if not RUNS.exists():
        raise SystemExit(f"{RUNS} is missing. Run scripts/judge_run.py first.")
    seen: dict[str, dict] = {}
    total = 0
    for line in RUNS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        # Keep the first answer for an id. They are answers to the same summary
        # text, so which one is kept is arbitrary; taking the first makes the
        # report deterministic for a given file.
        seen.setdefault(row["id"], row)
    return list(seen.values()), total - len(seen)


def confusion(rows) -> dict[str, int]:
    """Positive class is UNFAITHFUL: the thing the detector is for."""
    tp = sum(1 for r in rows if not r["gold_faithful"] and not r["said_faithful"])
    fp = sum(1 for r in rows if r["gold_faithful"] and not r["said_faithful"])
    fn = sum(1 for r in rows if not r["gold_faithful"] and r["said_faithful"])
    tn = sum(1 for r in rows if r["gold_faithful"] and r["said_faithful"])
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def scores(rows) -> dict[str, float]:
    c = confusion(rows)
    n = sum(c.values())
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "accuracy": (c["tp"] + c["tn"]) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def the_run(rows, duplicates: int = 0) -> None:
    rule("the run")
    if duplicates:
        print(f"duplicate answers  {duplicates:>5} dropped — the corpus holds 73 summaries")
        print("                         that appear 2-3 times in the same benchmark and")
        print("                         domain, so each was judged once per copy")
    total = planned()
    if total:
        print(
            f"answers recorded   {len(rows):>5} of {total:,} distinct summaries"
            f"   ({len(rows) / total:.0%})"
        )
        if len(rows) < total:
            print("  PARTIAL — the run has not finished. Everything below is on what exists.")
    else:
        print(f"answers recorded   {len(rows):>5}")
        print("  corpus not readable, so completeness is unknown")
    gold = Counter(r["gold_faithful"] for r in rows)
    dom = Counter(r["domain"] for r in rows)
    print(f"gold faithful      {gold[True]:>5}   unfaithful {gold[False]}")
    print(f"domains            {dict(dom)}")
    print(f"benchmarks         {len(Counter(r['benchmark'] for r in rows))}")


def the_scores(rows) -> None:
    rule("what it got right")
    print(f"{'split':<12}{'n':>6}{'accuracy':>11}{'precision':>11}{'recall':>9}{'F1':>8}")
    s = scores(rows)
    print(
        f"{'pooled':<12}{s['n']:>6}{s['accuracy']:>11.3f}"
        f"{s['precision']:>11.3f}{s['recall']:>9.3f}{s['f1']:>8.3f}"
    )
    per = {}
    for domain in ("cnndm", "xsum"):
        part = [r for r in rows if r["domain"] == domain]
        if not part:
            continue
        per[domain] = scores(part)
        d = per[domain]
        print(
            f"{domain:<12}{d['n']:>6}{d['accuracy']:>11.3f}"
            f"{d['precision']:>11.3f}{d['recall']:>9.3f}{d['f1']:>8.3f}"
        )

    if len(per) == 2:
        pooled = s["accuracy"]
        both = [d["accuracy"] for d in per.values()]
        if pooled > max(both):
            print("\n  ^ pooled accuracy beats BOTH domains — the same Simpson's")
            print("    paradox the model-free half shows. Pooling lets the detector")
            print("    act as a domain detector, because the two domains differ")
            print("    sharply in how often they are faithful.")
        else:
            print(
                f"\n  pooled {pooled:.3f} sits between the domains"
                f" ({min(both):.3f}-{max(both):.3f}), so no pooling effect here."
            )
    print("\n  Positive class is UNFAITHFUL throughout: that is the thing the")
    print("  detector exists to find, and scoring the other way round makes a")
    print("  detector that flags nothing look excellent.")


def the_evidence(rows) -> None:
    rule("whether the detector's own evidence is real")
    flagged = [r for r in rows if not r["said_faithful"]]
    invented = [r for r in flagged if not r["quote_is_real"]]
    abstained = [r for r in flagged if r["abstained"]]

    if not flagged:
        print("nothing was called unfaithful")
        return
    print(f"called unfaithful                    {len(flagged):>5}")
    print(
        f"  quote is not in the summary        {len(invented):>5}"
        f"   {len(invented) / len(flagged):>6.1%} of flags"
    )
    print(
        f"  refused to name any words at all   {len(abstained):>5}"
        f"   {len(abstained) / len(flagged):>6.1%} of flags"
    )

    verifiable = sum(1 for r in rows if "summary" in r)
    if verifiable < len(rows):
        print(f"\n  note: {len(rows) - verifiable} rows predate the summary being stored")
        print("  beside the answer. Their quote check is the one made at run time,")
        print("  against the real summary, and it stands — but it cannot be")
        print("  recomputed here. Rows written since carry the summary and can be.")

    print("\n  ^ this is the number this half was built for. The prompt requires")
    print("    the model to quote the exact words it objects to, and that quote")
    print("    is checkable with no judgement at all: the string is in the")
    print("    summary or it is not.")
    print("\n    A detector that says 'unfaithful' and then quotes a phrase the")
    print("    summary never contained has invented its own evidence. Nothing")
    print("    else in this pipeline would have noticed, because the verdict")
    print("    would still have been scored as correct whenever it happened to")
    print("    land on an unfaithful summary.")

    if invented:
        print("\n  examples:")
        for r in invented[:3]:
            print(f"    [{r['domain']}] quoted: {r['quote'][:72]!r}")


def by_benchmark(rows) -> None:
    rule("by source benchmark")
    groups = defaultdict(list)
    for r in rows:
        groups[r["benchmark"]].append(r)
    print(f"{'benchmark':<14}{'n':>5}{'accuracy':>11}{'F1':>8}{'invented evidence':>20}")
    for name, part in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        s = scores(part)
        flagged = [r for r in part if not r["said_faithful"]]
        bad = sum(1 for r in flagged if not r["quote_is_real"])
        share = f"{bad}/{len(flagged)}" if flagged else "—"
        print(f"{name:<14}{s['n']:>5}{s['accuracy']:>11.3f}{s['f1']:>8.3f}{share:>20}")
    print("\n  ^ small cells. Read the big ones and treat anything under ~30 as")
    print("    a hint rather than a measurement.")


def main() -> None:
    rows, duplicates = load()
    the_run(rows, duplicates)
    the_scores(rows)
    the_evidence(rows)
    by_benchmark(rows)
    print()


if __name__ == "__main__":
    main()
