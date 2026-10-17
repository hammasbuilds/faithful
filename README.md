# faithful

[![ci](https://github.com/hammasbuilds/faithful/actions/workflows/ci.yml/badge.svg)](https://github.com/hammasbuilds/faithful/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

We built a study of how summarisation faithfulness checkers are scored, on 10,066
human-labelled summaries from three published corpora, with a model-free checker and a 14B
judge run over every distinct summary. It is for anyone choosing or reporting a
hallucination detector for summaries. It measures four ways a checker looks better than it
is: pooled domains, F1 on an imbalanced set, recombined content it cannot see, and a judge
that invents its own evidence (**6.0% of its flags quote words the summary never
contained**).

## What it does

- Merges AggreFact (9 benchmarks, binary labels), FRANK (typed errors) and Google's XSum
  hallucination spans into one corpus at three grains: 10,066 binary, 2,613 typed, 1,858
  span-annotated summaries over 3,277 articles, in two domains (CNN/DailyMail and BBC/XSum).
- Scores a checker with **no model in it**: every content word and number in the summary must
  trace back to the article (stem match), or it is ungrounded. This sets how much of the
  problem needs a model at all.
- Runs `qwen2.5:14b-instruct` as a judge that must give a verdict **and** quote the exact
  words it objects to, then checks mechanically whether that quote is in the summary.
- Reports AUC, precision, recall and F1 per domain and per benchmark, with UNFAITHFUL as the
  positive class, and drops exact duplicate summaries (73 repeated summaries, 160 rows).

## Sample input / output

The model-free checker on one article and three summaries (real run of
`faithful.grounding.check`):

```text
article: Arsenal beat Chelsea 2-1 at the Emirates on Saturday. Bukayo Saka scored the
         winner in the 78th minute after Cole Palmer had equalised for Chelsea.

faithful                 ungrounded=1  words not in the article: late
extrinsic (invented)     ungrounded=8  numbers not in the article: 3, 60000; words not in the article: 000, 3-1, 60, fans, front, wembley
intrinsic (recombined)   ungrounded=0  every content word traces to the article
```

The invented summary is caught; the recombined one ("Cole Palmer scored the winner as
Chelsea beat Arsenal 2-1") reuses the article's own words and looks cleaner than the
faithful one. That is Result 3 below, at scale.

One judge record from [`data/judge_runs.jsonl`](data/judge_runs.jsonl):

```json
{"id": "FRANK:cnndm:f91a1f79", "gold_faithful": false, "said_faithful": false,
 "quote": "the 25-year-old is currently working in new york for the second day",
 "quote_is_real": true}
```

## Results

**1. Pooling the domains inflates the score (Simpson's paradox).**

| | n | AUC | faithful | mean ungrounded (unfaithful vs faithful) |
|---|---:|---:|---:|---|
| **Pooled** | 10,066 | **0.820** | 58% | 4.00 vs 0.88 |
| cnndm | 6,231 | 0.669 | 79% | 1.36 vs 0.39 |
| xsum | 3,835 | 0.668 | 23% | 5.17 vs 3.59 |

XSum summaries carry more ungrounded words *and* are more often unfaithful, so pooled, the
checker scores 0.820 by identifying the domain.

**2. F1 on an imbalanced set.** On XSum, threshold 1 flags 97.4% of summaries and scores
**F1 0.868** (precision 0.777, recall 0.984), because 77% of the set is unfaithful. The same
checker's AUC is 0.668.

**3. Invented content is detectable; recombined content is not.**

| XSum spans | n | AUC | mean ungrounded vs faithful |
|---|---:|---:|---|
| extrinsic only (invented) | 928 | **0.655** | 5.21 vs 3.89 |
| **intrinsic only** (recombined) | 177 | **0.327** | **2.68 vs 3.89** |
| both kinds | 568 | 0.563 | 4.36 vs 3.89 |

Intrinsic hallucinations score well below chance: they reuse source vocabulary more tightly
than an honest paraphrase. FRANK agrees from a different taxonomy (single-error summaries
only): out-of-article `OutE` AUC **0.926**, wrong-circumstance `CircE` 0.896, wrong-entity
`EntE` **0.680**.

**4. The 14B judge invents its own evidence in 6.0% of its flags.** Over all 9,979 distinct
summaries:

| | |
|---:|---:|
| called unfaithful | 4,606 |
| **quote not in the summary** | **276 (6.0% of flags)** |
| refused to name any words | 21 (0.5%) |

| Split | n | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| pooled | 9,979 | 0.825 | 0.773 | 0.836 | 0.803 |
| cnndm | 6,154 | 0.842 | **0.598** | 0.789 | 0.680 |
| xsum | 3,825 | 0.799 | **0.878** | 0.857 | 0.868 |

cnndm has the higher accuracy and the worse precision, because 79% of it is faithful.
Invented evidence by benchmark ranges from 1.8% (Goyal21) to 15.2% (Polytope); SummEval is
10.6%, FRANK 9.7%. Full per-benchmark table: [`data/judge_report.txt`](data/judge_report.txt).

The labels are on summaries written by 2018–2021 systems, so this scores the 14B as a
*judge*, not as a summariser. The judge sees the first 6,000 characters of each article.

## Tech stack

- Python 3.11+; the library is standard library only
- PyArrow (once, to build the corpus from the downloaded parquet files)
- Ollama `qwen2.5:14b-instruct` for the judge arm (its 9,979 answers are committed)
- pytest, ruff, GitHub Actions

## Run it

```bash
git clone https://github.com/hammasbuilds/faithful
cd faithful
python -m venv .venv && .venv/Scripts/pip install -e ".[dev,build]"   # .venv/bin on Linux/macOS

bash scripts/fetch_data.sh          # AggreFact, FRANK, XSum annotations, XSum test split
python scripts/build_corpus.py      # -> data/corpus.json
python scripts/measure.py           # Results 1-3
python scripts/judge_report.py      # Result 4, from the committed data/judge_runs.jsonl

python -m pytest -q                 # no corpus or model needed
```

The stemmer is a suffix strip: irregular forms such as `withdrew` / `withdrawn` do not match
and count as ungrounded.

```
src/faithful/corpus.py      the three grains, loaded and typed
src/faithful/grounding.py   the model-free checker
src/faithful/judge.py       the 14B judge, with the quote-is-real check
scripts/                    fetch, build, measure, judge run and report
```

## Licence

Code: MIT, see [LICENSE](LICENSE). The corpus is not committed; `scripts/fetch_data.sh`
downloads AggreFact (Tang et al., 2023), FRANK (Pagnoni et al., 2021) and the XSum
hallucination annotations (Maynez et al., 2020), each under its own licence.
`data/judge_runs.jsonl` and `data/judge_report.txt` are this repository's judge output
(CC BY 4.0); the summaries quoted in them stay under the benchmarks' terms.
