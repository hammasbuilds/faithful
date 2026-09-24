"""The model-free checker against 10,066 human-labelled summaries.

Every figure asserted here was produced by running `scripts/measure.py` over
`data/corpus.json`. Nothing is typed from a paper.

The corpus has three grains and each answers something the others cannot:
binary labels at 10k across nine benchmarks and two domains, FRANK's typed
error taxonomy at 2.6k, and XSum's span marks at 1.9k. The central claim here
is only visible at the second and third.
"""

import statistics

import pytest

from faithful.corpus import CORPUS, graded, load
from faithful.grounding import check

pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built")


def ungrounded_count(item) -> int:
    return check(item.summary, item.document).ungrounded


def auc(positive, negative) -> float:
    """P(a random unfaithful scores above a random faithful). 0.5 is chance."""
    p = [ungrounded_count(i) for i in positive]
    n = [ungrounded_count(i) for i in negative]
    wins = ties = 0
    for a in p:
        for b in n:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(p) * len(n))


@pytest.fixture(scope="module")
def items():
    return load()


@pytest.fixture(scope="module")
def marks():
    return graded()


def test_the_corpus_is_the_size_it_claims(items, marks):
    assert len(items) == 10_066
    assert len(marks) == 1_858
    assert sum(1 for i in items if i.typed) == 2_613
    assert {i.domain for i in items} == {"cnndm", "xsum"}
    assert len({i.benchmark for i in items}) == 9


def test_both_domains_are_present_in_force(items):
    # A finding on one alone is a fact about that one. XSum is single-sentence
    # and aggressively abstractive; CNN/DailyMail is multi-sentence and closer
    # to extractive. They behave completely differently below.
    assert sum(1 for i in items if i.domain == "cnndm") == 6_231
    assert sum(1 for i in items if i.domain == "xsum") == 3_835


def test_pooling_the_domains_inflates_the_score(items):
    # THE FIRST FINDING. Simpson's paradox, measured on a benchmark people
    # publish against. The pooled AUC beats BOTH of the domains it is made of.
    #
    # The mechanism is not subtle once seen: XSum summaries carry far more
    # ungrounded words than CNN/DM ones (5.17 vs 1.36) AND are far more often
    # unfaithful (77% vs 21%). Pool them and "many ungrounded words" becomes a
    # detector for "is XSum", which is a detector for "is unfaithful". The
    # checker scores well by identifying the domain.
    pooled = auc([i for i in items if not i.faithful], [i for i in items if i.faithful])
    assert pooled == pytest.approx(0.820, abs=0.01)

    within = {}
    for domain in ("cnndm", "xsum"):
        sub = [i for i in items if i.domain == domain]
        within[domain] = auc([i for i in sub if not i.faithful], [i for i in sub if i.faithful])

    assert within["cnndm"] == pytest.approx(0.669, abs=0.01)
    assert within["xsum"] == pytest.approx(0.668, abs=0.01)
    assert pooled > max(within.values()) + 0.1


def test_the_domains_differ_in_both_label_rate_and_feature(items):
    # Both conditions are needed for the paradox, so both are pinned.
    for domain, faithful_share, mean_ungrounded in (
        ("cnndm", 0.79, 1.36),
        ("xsum", 0.23, 5.17),
    ):
        sub = [i for i in items if i.domain == domain]
        share = sum(1 for i in sub if i.faithful) / len(sub)
        loose = statistics.mean(ungrounded_count(i) for i in sub if not i.faithful)
        assert share == pytest.approx(faithful_share, abs=0.02), domain
        assert loose == pytest.approx(mean_ungrounded, abs=0.15), domain


def test_f1_flatters_the_checker_on_an_imbalanced_set(items):
    # THE SECOND FINDING. On XSum alone, flagging 97% of everything scores
    # F1 0.87 — because 77% of the set really is unfaithful. The same checker,
    # same data, scores AUC 0.668. One number says "excellent", the other says
    # "barely better than a coin flip", and only the second is about the checker.
    xsum = [i for i in items if i.domain == "xsum"]
    truth = [not i.faithful for i in xsum]
    flagged = [ungrounded_count(i) >= 1 for i in xsum]

    pairs = list(zip(flagged, truth, strict=True))
    tp = sum(1 for f, t in pairs if f and t)
    fp = sum(1 for f, t in pairs if f and not t)
    fn = sum(1 for f, t in pairs if not f and t)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    assert sum(flagged) / len(xsum) > 0.97  # it flags almost everything
    assert f1 == pytest.approx(0.868, abs=0.01)
    assert sum(truth) / len(xsum) == pytest.approx(0.77, abs=0.02)

    honest = auc([i for i in xsum if not i.faithful], [i for i in xsum if i.faithful])
    assert honest == pytest.approx(0.668, abs=0.01)
    assert f1 - honest > 0.15


def test_invented_content_is_detectable_and_recombined_content_is_not(marks):
    # THE CENTRAL FINDING, on XSum's span marks.
    #
    # An *extrinsic* hallucination introduces content the article never had, so
    # a lexical check sees new words and scores above chance. An *intrinsic*
    # one recombines the article's own words into a claim it never made, so
    # every word traces back and the check sees nothing.
    #
    # It does not merely fail there. It scores 0.327 — well BELOW chance — and
    # the reason is mechanical: an intrinsic hallucination reuses source
    # vocabulary more tightly than an honest paraphrase does. The checker rates
    # the lie as safer than the truth.
    clean = [g for g in marks if g.faithful]
    extrinsic = [g for g in marks if g.extrinsic_only and not g.faithful]
    intrinsic = [g for g in marks if g.intrinsic_only and not g.faithful]

    assert len(extrinsic) == 928
    assert len(intrinsic) == 177

    assert auc(extrinsic, clean) == pytest.approx(0.655, abs=0.02)
    assert auc(intrinsic, clean) == pytest.approx(0.327, abs=0.02)
    assert auc(intrinsic, clean) < 0.5

    # And the mechanism, stated as a number: intrinsic hallucinations carry
    # FEWER ungrounded words than faithful summaries do.
    loose_intrinsic = statistics.mean(ungrounded_count(g) for g in intrinsic)
    loose_faithful = statistics.mean(ungrounded_count(g) for g in clean)
    assert loose_intrinsic < loose_faithful


def test_a_second_corpus_and_taxonomy_agree(items):
    # The same claim, arrived at independently. FRANK tags errors by type on
    # CNN/DailyMail and BBC together; `OutE` is content that is not in the
    # article at all, which is what XSum calls extrinsic, and `EntE` is a wrong
    # entity, which is recombination.
    #
    # Restricted to summaries carrying exactly ONE kind of error, so the groups
    # do not overlap: OutE 0.926, EntE 0.680. Different corpus, different
    # annotators, different taxonomy, same conclusion.
    typed_items = [i for i in items if i.typed]
    base = [i for i in typed_items if i.faithful]

    def only(kind):
        return [i for i in typed_items if not i.faithful and i.error_kinds == (kind,)]

    out_of_article = only("OutE")
    wrong_entity = only("EntE")
    assert len(out_of_article) == 118
    assert len(wrong_entity) == 58

    assert auc(out_of_article, base) == pytest.approx(0.926, abs=0.02)
    assert auc(wrong_entity, base) == pytest.approx(0.680, abs=0.03)
    assert auc(out_of_article, base) > auc(wrong_entity, base) + 0.15


def test_every_item_carries_its_article(items):
    assert all(i.document for i in items)
    assert all(i.summary for i in items)
    # Articles are stored once and shared; 10,066 summaries, far fewer articles.
    assert len({i.document for i in items}) < len(items) / 2


def test_a_missing_corpus_is_reported_rather_than_faked():
    from faithful.corpus import CorpusMissingError

    with pytest.raises(CorpusMissingError):
        load(str(CORPUS.parent / "nope.json"))
