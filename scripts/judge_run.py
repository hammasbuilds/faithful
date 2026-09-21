"""Run the 14B judge over a stratified sample and record every answer.

    python scripts/judge_run.py --n 600

Two things make this worth writing as a script rather than a loop in a
notebook.

**It is resumable.** Every answer is appended to `data/judge_runs.jsonl` as it
arrives, keyed by item id. A rerun skips what is already there. A 600-item run
is roughly an hour on this card and it should not have to survive a session
ending, a reboot, or somebody else wanting the GPU.

**The sample is stratified and its recipe is in the file.** The corpus is 58%
faithful overall but 79% faithful in cnndm and 23% in xsum, so a random draw
would be mostly cnndm-faithful and would flatter anything measured on it. The
sample takes equal numbers from each (domain, label) cell, which makes accuracy
mean something and makes the two domains comparable to each other.

`scripts/judge_report.py` reads the jsonl and prints the tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faithful import corpus, judge

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "judge_runs.jsonl"

# Equal cells rather than the corpus's own mix. See the module docstring.
CELLS = (("cnndm", True), ("cnndm", False), ("xsum", True), ("xsum", False))


def sample(items, n: int, seed: int = 7) -> list:
    """n items, split as evenly as the corpus allows across the four cells."""
    buckets: dict[tuple[str, bool], list] = defaultdict(list)
    for item in items:
        buckets[(item.domain, item.faithful)].append(item)

    rng = random.Random(seed)
    want = n // len(CELLS)
    picked = []
    for cell in CELLS:
        pool = buckets[cell]
        if len(pool) < want:
            print(f"  note: {cell} has only {len(pool)}, wanted {want}")
        picked += rng.sample(pool, min(want, len(pool)))
    rng.shuffle(picked)
    return picked


def already_done() -> set[str]:
    if not OUT.exists():
        return set()
    done = set()
    with OUT.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                done.add(json.loads(line)["id"])
    return done


def item_id(item) -> str:
    """Stable across runs and across processes.

    The first version used `hash(item.summary)`, and Python salts string
    hashing per process. That is fine inside one run and wrong for the thing
    this id exists to do: the ids written by one process never match the ids
    computed by the next, so `already_done` matches nothing and a resumed run
    silently repeats every item it has already paid for.

    It did not bite because the first 366 happened to complete in a single
    process. The next resume would have re-run all of them — about an hour of a
    contended GPU — and the only evidence would have been duplicate rows nobody
    was counting.
    """
    digest = hashlib.blake2b(item.summary.encode("utf-8"), digest_size=4).hexdigest()
    return f"{item.benchmark}:{item.domain}:{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--model", default=judge.DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    items = corpus.load()
    chosen = sample(items, args.n, args.seed)
    done = already_done()
    todo = [i for i in chosen if item_id(i) not in done]

    print(f"corpus {len(items):,}   sample {len(chosen)}   done {len(done)}   todo {len(todo)}")
    if not todo:
        print("nothing to do. run scripts/judge_report.py")
        return

    model = judge.ollama(args.model)
    started = time.time()
    failures = 0

    with OUT.open("a", encoding="utf-8") as handle:
        for n, item in enumerate(todo, 1):
            try:
                verdict = judge.ask(model, item.document, item.summary)
            except judge.ModelUnavailableError as exc:
                failures += 1
                print(f"  [{n}] model unavailable: {exc}")
                if failures >= 3:
                    print("  three failures in a row region — stopping. Rerun to resume.")
                    return
                continue
            failures = 0

            handle.write(
                json.dumps(
                    {
                        "id": item_id(item),
                        # The summary is stored so that `quote_is_real` stays
                        # re-derivable. Without it the only way to re-check a
                        # recorded quote is to rejoin to the corpus by id, and
                        # an id that is even slightly wrong silently checks the
                        # quote against somebody else's summary.
                        "summary": item.summary,
                        "benchmark": item.benchmark,
                        "domain": item.domain,
                        "gold_faithful": item.faithful,
                        "said_faithful": verdict.faithful,
                        "quote": verdict.quote,
                        "quote_is_real": verdict.quote_is_real,
                        "abstained": verdict.abstained,
                    }
                )
                + "\n"
            )
            handle.flush()

            if n % 25 == 0 or n == len(todo):
                rate = (time.time() - started) / n
                left = (len(todo) - n) * rate
                print(f"  {n}/{len(todo)}  {rate:.1f}s/item  ~{left / 60:.0f} min left")

    print(f"\nwrote {OUT}  ({time.time() - started:.0f}s)")
    print("now: python scripts/judge_report.py")


if __name__ == "__main__":
    main()
