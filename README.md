# faithful

> Four ways a summarisation faithfulness checker looks better than it is — measured on 10,066 human-labelled summaries, and on a 14B judge over every one of them.

**Status:** complete. The model-free half is measured, and the 14B judge has now been run
over **all 9,979 distinct summaries** — 4 hours 25 minutes on one Quadro RTX 5000. Its own
evidence turns out to be false in 6% of its flags.

## The corpus

Three published corpora, merged, at three grains. They were chosen because they disagree
about what is worth recording, and each disagreement buys something.

| Grain | Source | n | What only it can answer |
|---|---|---:|---|
| **Binary** | AggreFact — 9 benchmarks | **10,066** | Does the checker work at all, across two domains |
| **Typed** | FRANK error taxonomy | **2,613** | *Which kind* of error it misses |
| **Spans** | Google XSum annotations | **1,858** | Invented content vs recombined content |

3,277 distinct articles. Two domains: **CNN/DailyMail** (multi-sentence, near-extractive)
and **BBC/XSum** (single-sentence, aggressively abstractive). Benchmarks inside AggreFact:
XSumFaith, SummEval, FactCC, FRANK, Polytope, Cao22, CLIFF, Wang20, Goyal21.

## What is being tested

A checker with **no model in it at all**: every content word and every number in the summary
must trace back to the article, or it is ungrounded. The point is not that it beats a model.
It is to establish how much of the problem needs one.

```
python scripts/measure.py     # prints every table below
```

## Result 1 — pooling the domains inflates the score

| | n | AUC | faithful | mean ungrounded (unfaithful vs faithful) |
|---|---:|---:|---:|---|
| **Pooled** | 10,066 | **0.820** | 58% | 4.00 vs 0.88 |
| cnndm | 6,231 | 0.669 | 79% | 1.36 vs 0.39 |
| xsum | 3,835 | 0.668 | 23% | 5.17 vs 3.59 |

**The pooled score beats both of the domains it is made of.** That is Simpson's paradox, on
a benchmark people publish against.

The mechanism is not subtle once seen. XSum summaries carry far more ungrounded words than
CNN/DM ones (5.17 vs 1.36) **and** are far more often unfaithful (77% vs 21%). Pool them and
"many ungrounded words" becomes a detector for "is XSum", which is a detector for "is
unfaithful". The checker scores 0.820 by identifying the **domain**.

Both conditions are needed, and both are pinned by a test.

## Result 2 — F1 on an imbalanced set

On XSum alone, flagging **97.4% of everything**:

| Threshold | Flags | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 1 | 97.4% | 0.777 | 0.984 | **0.868** |
| 3 | 78.1% | 0.815 | 0.828 | 0.821 |
| 5 | 49.0% | 0.856 | 0.546 | 0.667 |

**F1 0.868 — because 77% of the set really is unfaithful.** The same checker, on the same
data, scores **AUC 0.668**. One number says excellent, the other says barely better than a
coin flip, and only the second is about the checker.

## Result 3 — invented content is detectable, recombined content is not

The central result, and the reason the corpus needed a span grain.

| XSum spans | n | AUC | mean ungrounded vs faithful |
|---|---:|---:|---|
| extrinsic only *(invented)* | 928 | **0.655** | 5.21 vs 3.89 |
| **intrinsic only** *(recombined)* | 177 | **0.327** | **2.68 vs 3.89** |
| both kinds | 568 | 0.563 | 4.36 vs 3.89 |

An **extrinsic** hallucination introduces content the article never had, so a lexical check
sees new words. An **intrinsic** one recombines the article's own words into a claim it never
made — every word traces back, and the check sees nothing.

It does not merely fail. It scores **0.327, well below chance**, and the reason is mechanical:
an intrinsic hallucination reuses source vocabulary more tightly than an honest paraphrase
does. Look at the last column — intrinsically false summaries carry **fewer** ungrounded words
(2.68) than faithful ones (3.89).

**The checker rates the lie as safer than the truth.**

### The same conclusion from a different corpus and taxonomy

FRANK tags errors by type across both domains. `OutE` is content not in the article at all —
what XSum calls extrinsic. `EntE` is a wrong entity — recombination. Restricted to summaries
carrying exactly **one** kind of error, so the groups cannot overlap:

| FRANK, single-error only | n | AUC |
|---|---:|---:|
| **OutE** — out of article | 118 | **0.926** |
| CircE — wrong circumstance | 26 | 0.896 |
| **EntE** — wrong entity | 58 | **0.680** |

Different corpus, different annotators, different taxonomy, same conclusion. That is why
three grains rather than one.

## Result 4 — the detector invents its own evidence in 6% of its flags

`qwen2.5:14b-instruct` judged **all 9,979 distinct summaries** in the corpus. The prompt
requires a verdict *and* a quote of the exact words it objects to, and that quote is
checkable with no judgement at all: the string is in the summary or it is not.

| | |
|---|---:|
| Called unfaithful | 4,606 |
| **Quote is not in the summary** | **276 — 6.0% of flags** |
| Refused to name any words | 21 — 0.5% |

**A detector that says "unfaithful" and then quotes a phrase the summary never contained has
invented its own evidence.** Nothing else in the pipeline notices, because the verdict is
still scored correct whenever it happens to land on a genuinely unfaithful summary. The
accuracy column cannot see this; only asking the model to point at something can.

```
[xsum]  quoted: 'sadiq khan has praised the grenfell tower fire as "frightened"...'
[cnndm] quoted: 'the boy was rescued by his parents before firefighters and para...'
```

It is not evenly spread. Per benchmark, invented evidence as a share of flags:

| Benchmark | n | Accuracy | F1 | Invented evidence |
|---|---:|---:|---:|---:|
| XSumFaith | 2,343 | 0.846 | 0.913 | 79/2,025 — 3.9% |
| SummEval | 1,646 | 0.875 | 0.605 | 36/340 — **10.6%** |
| FactCC | 1,434 | 0.835 | 0.598 | 17/395 — 4.3% |
| FRANK | 1,392 | 0.817 | 0.804 | 59/611 — 9.7% |
| Polytope | 1,244 | 0.834 | 0.585 | 50/328 — **15.2%** |
| Cao22 | 696 | 0.675 | 0.653 | 12/354 — 3.4% |
| CLIFF | 600 | 0.812 | 0.729 | 13/191 — 6.8% |
| Wang20 | 474 | 0.781 | 0.791 | 8/253 — 3.2% |
| Goyal21 | 150 | 0.753 | 0.829 | 2/109 — 1.8% |

### And the judge's own accuracy

| Split | n | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| pooled | 9,979 | 0.825 | 0.773 | 0.836 | 0.803 |
| cnndm | 6,154 | 0.842 | **0.598** | 0.789 | 0.680 |
| xsum | 3,825 | 0.799 | **0.878** | 0.857 | 0.868 |

**cnndm has the higher accuracy and the worse precision.** 79% of cnndm is genuinely
faithful, so accuracy is carried by the majority class while precision on the thing the
detector exists to find collapses. That is Finding 2 again, on a 14B instead of on word
counting — the metric moved, the failure did not.

Positive class is UNFAITHFUL throughout. Scoring it the other way round makes a detector
that flags nothing look excellent.

### 73 summaries in this corpus are duplicates

Judging every row surfaced something about AggreFact itself: **160 of the 10,066 rows are
73 summaries repeated two or three times** inside the same benchmark and domain — identical
strings, not hash collisions. The report deduplicates and says how many it dropped. It moves
pooled accuracy by one thousandth, which is precisely the size of error nobody checks.

## What is not measured yet

Two things this repository does **not** claim:

- **Nothing here measures whether a 14B writes unfaithful summaries.** Every label in the
  corpus is on output from 2018–2021 systems. The corpus can score a 14B as a *judge* of
  those summaries; it carries no ground truth for summaries a 14B writes itself.
- **Long articles are truncated to 6,000 characters** for the judge. A claim supported only
  by the end of a very long article will be judged unsupported.

## Known limitation, kept as a test

The stemmer is a suffix strip, not a real stemmer. `withdrew` and `withdrawn` are the same
verb and no rule connects them, so a faithful summary using one against an article using the
other is reported as ungrounded. A dictionary would fix it; this module is deliberately
cheap, and `test_an_irregular_verb_defeats_the_stemmer_and_that_is_recorded` is where that
choice is priced.

## Build it

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev,build]"

bash scripts/fetch_data.sh        # AggreFact, FRANK, XSum annotations, XSum test split
python scripts/build_corpus.py    # -> data/corpus.json
python scripts/measure.py         # every table above

python -m pytest -q               # 31 passed
```

Runtime dependencies: **none**. `pyarrow` is needed once, to build the corpus; the library
itself is standard library only.

## Layout

```
src/faithful/
  corpus.py     the three grains, loaded and typed
  grounding.py  the model-free checker
  judge.py      the 14B judge, with the quote-is-real check
scripts/
  fetch_data.sh    the four downloads, with the byte-range workaround
  build_corpus.py  merge into data/corpus.json
  measure.py       print every number this README claims
```

## Licence

Code: MIT, see [LICENSE](LICENSE). The corpus is not committed; `scripts/fetch_data.sh`
downloads AggreFact, FRANK and the XSum hallucination annotations, each under its own
licence. `data/judge_runs.jsonl` and `data/judge_report.txt` are this repository's judge
output (CC BY 4.0); the summaries quoted inside them come from those benchmarks and stay
under the benchmarks' terms.
