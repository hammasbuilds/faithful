# faithful

> Three ways a summarisation faithfulness checker looks better than it is — measured on 10,066 human-labelled summaries.

**Status:** the model-free half is complete and measured. The 14B judge is written and
tested, and has not been run: the GPU is busy with another session's work, and evicting it
is not something this repository does. See [What is not measured yet](#what-is-not-measured-yet).

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

## Finding 1 — pooling the domains inflates the score

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

## Finding 2 — F1 flatters a checker on an imbalanced set

On XSum alone, flagging **97.4% of everything**:

| Threshold | Flags | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 1 | 97.4% | 0.777 | 0.984 | **0.868** |
| 3 | 78.1% | 0.815 | 0.828 | 0.821 |
| 5 | 49.0% | 0.856 | 0.546 | 0.667 |

**F1 0.868 — because 77% of the set really is unfaithful.** The same checker, on the same
data, scores **AUC 0.668**. One number says excellent, the other says barely better than a
coin flip, and only the second is about the checker.

## Finding 3 — invented content is detectable, recombined content is not

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

## What is not measured yet

**The 14B judge has not been run.** `src/faithful/judge.py` is written and unit-tested, and
asks `qwen2.5:14b-instruct` for a verdict plus **a quote of the words it objects to** — the
quote being checkable without any judgement at all. That yields a question worth asking:
*does a hallucination detector hallucinate its own evidence?* `Verdict.quote_is_real` exists
to count it.

It has not been run because another session holds the GPU (`qwen2.5-coder:14b`, 98%
utilisation), and taking it would destroy that session's work. The comparison is queued, not
abandoned.

Two further things this repository does **not** claim:

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

python -m pytest -q               # 19 passed
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
