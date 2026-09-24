"""Every number this repository claims, printed from the corpus.

Run it and you get the README's tables. Nothing here is typed by hand.

    python scripts/measure.py
"""

from __future__ import annotations

import statistics
from collections import Counter

from faithful.corpus import graded, load
from faithful.grounding import check

_CACHE: dict[int, int] = {}


def score(item) -> int:
    key = id(item)
    if key not in _CACHE:
        _CACHE[key] = check(item.summary, item.document).ungrounded
    return _CACHE[key]


def auc(positive, negative) -> tuple[float, float, float]:
    """P(a random unfaithful scores above a random faithful). 0.5 is a coin flip.

    Reported instead of precision and recall because these corpora are badly
    imbalanced — XSum runs 77% unfaithful — and on a set that lopsided a
    checker that flags everything posts a fine F1 while being worthless.
    """
    p = [score(i) for i in positive]
    n = [score(i) for i in negative]
    if not p or not n:
        return 0.0, 0.0, 0.0
    wins = ties = 0
    for a in p:
        for b in n:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(p) * len(n)), statistics.mean(p), statistics.mean(n)


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def grain_one(items) -> None:
    rule(f"GRAIN 1 - binary labels, AggreFact, n={len(items):,}")
    print("benchmarks:", dict(Counter(i.benchmark for i in items).most_common()))

    faithful = [i for i in items if i.faithful]
    unfaithful = [i for i in items if not i.faithful]
    pooled, mu, mf = auc(unfaithful, faithful)
    print(
        f"\nPOOLED   AUC {pooled:.3f}   faithful {len(faithful) / len(items) * 100:.0f}%"
        f"   mean ungrounded {mu:.2f} vs {mf:.2f}"
    )

    for domain in ("cnndm", "xsum"):
        sub = [i for i in items if i.domain == domain]
        a, m_u, m_f = auc([i for i in sub if not i.faithful], [i for i in sub if i.faithful])
        share = sum(1 for i in sub if i.faithful) / len(sub) * 100
        print(
            f"  {domain:<6} n={len(sub):>5}  AUC {a:.3f}"
            f"  faithful {share:>4.0f}%   ({m_u:.2f} vs {m_f:.2f})"
        )

    print("\n  ^ the pooled number beats BOTH domains. Pooling lets the ungrounded")
    print("    count act as a domain detector, and the domains differ sharply in")
    print("    base faithfulness. That gap is Simpson's paradox, measured.")


def grain_two(items) -> None:
    typed_items = [i for i in items if i.typed]
    rule(f"GRAIN 2 - typed errors, FRANK, n={len(typed_items):,}")
    base = [i for i in typed_items if i.faithful]

    print(f"{'error kind':<12}{'any':>6}{'AUC':>8}{'only':>7}{'AUC':>8}")
    for kind in ("OutE", "EntE", "RelE", "CircE", "CorefE", "GramE", "LinkE"):
        tagged = [i for i in typed_items if kind in i.error_kinds and not i.faithful]
        alone = [i for i in tagged if len(i.error_kinds) == 1]
        a_any, _, _ = auc(tagged, base)
        line = f"{kind:<12}{len(tagged):>6}{a_any:>8.3f}"
        if len(alone) >= 20:
            a_only, _, _ = auc(alone, base)
            line += f"{len(alone):>7}{a_only:>8.3f}"
        else:
            line += f"{len(alone):>7}{'--':>8}"
        print(line)

    print("\n  ^ read the 'any' column with care: a FRANK summary usually carries")
    print("    several error kinds at once, so those groups overlap heavily and")
    print("    their scores are dragged together. 'only' is the clean comparison,")
    print("    where the sample allows one at all.")


def grain_three(marks) -> None:
    rule(f"GRAIN 3 - spans, XSum, n={len(marks):,}")
    clean = [g for g in marks if g.faithful]
    print(f"faithful {len(clean)}   ({len(clean) / len(marks) * 100:.0f}%)")

    groups = (
        ("extrinsic only", [g for g in marks if g.extrinsic_only and not g.faithful]),
        ("intrinsic only", [g for g in marks if g.intrinsic_only and not g.faithful]),
        ("both kinds", [g for g in marks if len(g.kinds) == 2 and not g.faithful]),
    )
    for name, group in groups:
        a, m_u, m_f = auc(group, clean)
        print(f"  {name:<16} n={len(group):>5}  AUC {a:.3f}   ({m_u:.2f} vs {m_f:.2f})")

    print("\n  ^ intrinsic hallucination scores BELOW chance. Recombining the")
    print("    article's own words looks MORE grounded than honest paraphrase,")
    print("    so the checker rates the lie as safer than the truth.")


def the_trap(items) -> None:
    rule("THE TRAP - what the naive metric says on XSum alone")
    xsum = [i for i in items if i.domain == "xsum"]
    truth = [not i.faithful for i in xsum]

    for threshold in (1, 3, 5):
        flagged = [score(i) >= threshold for i in xsum]
        pairs = list(zip(flagged, truth, strict=True))
        tp = sum(1 for f, t in pairs if f and t)
        fp = sum(1 for f, t in pairs if f and not t)
        fn = sum(1 for f, t in pairs if not f and t)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(
            f"  threshold {threshold}: flags {sum(flagged) / len(xsum) * 100:>5.1f}%"
            f"  precision {precision:.3f}  recall {recall:.3f}  F1 {f1:.3f}"
        )

    share = sum(truth) / len(xsum) * 100
    a, _, _ = auc([i for i in xsum if not i.faithful], [i for i in xsum if i.faithful])
    print(f"\n  ^ F1 reads 0.87 on a set that is {share:.0f}% unfaithful, while the")
    print(f"    same checker scores AUC {a:.3f} there. Flagging everything looks")
    print("    good when almost everything is guilty: that F1 is mostly the base")
    print("    rate, not the checker.")


def main() -> None:
    items = load()
    grain_one(items)
    grain_two(items)
    grain_three(graded())
    the_trap(items)


if __name__ == "__main__":
    main()
