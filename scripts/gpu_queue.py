"""Wait for the GPU, then run the judge. Never evict anyone.

    python scripts/gpu_queue.py --n 600

This machine runs more than one session and they share one 16 GB card. A run
that simply fires requests at ollama will either sit blocked behind whatever is
already loaded, or — worse — cause a model swap that throws out work somebody
else is in the middle of.

So this polls instead. It asks ollama what is resident; if that is a model this
run does not want, it waits and asks again. When the card is free, or already
holds the model we want, it hands off to `judge_run`, which is itself resumable
and appends as it goes.

Nothing here kills a process, unloads a model, or shortens anyone's keep_alive.
The only lever it uses is patience.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faithful import judge

PS = "http://127.0.0.1:11434/api/ps"
HERE = Path(__file__).resolve().parent


def resident() -> list[str]:
    """Models currently held in VRAM, as ollama reports them.

    A failure to reach ollama is reported as 'unknown' rather than 'nothing is
    loaded', because treating an unreachable server as an idle one is how a
    polite queue turns into a rude one.
    """
    try:
        with urllib.request.urlopen(PS, timeout=15) as response:
            return [m["name"] for m in json.loads(response.read()).get("models", [])]
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError):
        return ["unknown"]


def wait_for_card(want: str, every: int, limit: float, settle: int = 1) -> bool:
    """Block until the card is free or already holds `want`. True if it is.

    `settle` is how many consecutive free polls are required before claiming it.
    Taking the card the instant it reads free is not the same as queueing last:
    a session that finishes one job and starts the next leaves a gap of a few
    seconds, and a queue that pounces on that gap has jumped ahead of work that
    was already in progress. Requiring the card to stay quiet across several
    polls lets anyone mid-sequence keep it, and costs this run only the wait.
    """
    started = time.time()
    told = None
    free_in_a_row = 0
    while time.time() - started < limit:
        loaded = resident()
        if not loaded or loaded == [want]:
            free_in_a_row += 1
            if free_in_a_row >= settle:
                return True
            print(f"  card looks free ({free_in_a_row}/{settle}) — holding back in case "
                  f"somebody else is between jobs")
        else:
            if free_in_a_row:
                print("  somebody else took it — good, waiting again")
            free_in_a_row = 0
            if loaded != told:
                waited = (time.time() - started) / 60
                print(f"  [{waited:5.1f} min] card holds {loaded} — waiting, not evicting")
                told = loaded
        time.sleep(every)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--model", default=judge.DEFAULT_MODEL)
    parser.add_argument("--poll", type=int, default=60, help="seconds between checks")
    parser.add_argument("--max-wait", type=float, default=12 * 3600)
    parser.add_argument(
        "--settle",
        type=int,
        default=10,
        help="consecutive free polls required before claiming the card",
    )
    args = parser.parse_args()

    print(f"queued for {args.model}; polling every {args.poll}s, "
          f"claiming only after {args.settle} consecutive free polls "
          f"({args.settle * args.poll / 60:.0f} min quiet)")
    if not wait_for_card(args.model, args.poll, args.max_wait, args.settle):
        print(f"\ngave up after {args.max_wait / 3600:.0f}h. The card was never free.")
        print("Nothing was run and nothing was evicted. Rerun to queue again.")
        raise SystemExit(2)

    print("card is free — starting the judge")
    raise SystemExit(
        subprocess.call(
            [sys.executable, str(HERE / "judge_run.py"), "--n", str(args.n),
             "--model", args.model]
        )
    )


if __name__ == "__main__":
    main()
