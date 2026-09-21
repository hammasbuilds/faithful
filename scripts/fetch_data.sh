#!/usr/bin/env bash
# The four inputs. Everything here is public and needs no account.
#
# One of them fights back. `aggre_fact_final.csv` is 30 MB and a plain GET
# stalls at zero on a throttled link, while byte-range requests come through
# fine — so it is fetched in 2 MB chunks. That is not belt-and-braces, it is
# the only way it arrives.
set -u
cd "$(dirname "$0")/../data" || exit 1

echo "1/4  AggreFact  (30 MB, byte-range)"
U="https://raw.githubusercontent.com/Liyan06/AggreFact/main/data/aggre_fact_final.csv"
: > aggrefact.csv
for i in $(seq 0 15); do
  s=$((i * 2000000)); e=$((s + 1999999))
  curl -sL -r "${s}-${e}" "$U" >> aggrefact.csv || break
  size=$(stat -c%s aggrefact.csv)
  printf '     chunk %2d -> %s bytes\n' "$i" "$size"
  [ "$size" -lt "$((e + 1))" ] && break
done

echo "2/4  FRANK typed error annotations"
curl -sL "https://raw.githubusercontent.com/artidoro/frank/main/data/human_annotations.json" \
  -o frank.json

echo "3/4  Google XSum annotations"
BASE="https://raw.githubusercontent.com/google-research-datasets/xsum_hallucination_annotations/master"
curl -sL "$BASE/factuality_annotations_xsum_summaries.csv" -o xsum_factuality.csv
curl -sL "$BASE/hallucination_annotations_xsum_summaries.csv" \
  -o hallucination_annotations_xsum_summaries.csv

echo "4/4  XSum test split (source articles, 16 MB)"
curl -sL "https://huggingface.co/api/datasets/EdinburghNLP/xsum/parquet/default/test/0.parquet" \
  -o xsum_test.parquet

echo
echo "done:"
ls -la aggrefact.csv frank.json xsum_factuality.csv \
       hallucination_annotations_xsum_summaries.csv xsum_test.parquet 2>/dev/null
echo
echo "now run: python scripts/build_corpus.py"
