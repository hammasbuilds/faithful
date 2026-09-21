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


def wait_for_card(want: str, every: int, limit: float) -> bool:
    """Block until the card is free or already holds `want`. True if it is."""
    started = time.time()
    told = None
    while time.time() - started < limit:
        loaded = resident()
        if not loaded or loaded == [want]:
            return True
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
    args = parser.parse_args()

    print(f"queued for {args.model}; polling every {args.poll}s")
    if not wait_for_card(args.model, args.poll, args.max_wait):
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
