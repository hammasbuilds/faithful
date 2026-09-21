"""The model-free grounding check, on constructed cases.

These are unit tests of the rule itself. What the rule is worth on real data is
in test_real_corpus.py, and the answer there is "less than it looks".
"""

import pytest

from faithful.grounding import check, numbers, tokens, ungrounded


def test_a_summary_drawn_entirely_from_the_article_is_grounded():
    article = "Police said the fire began in the kitchen of a house in Bristol."
    summary = "A fire began in a Bristol kitchen."
    assert check(summary, article).ungrounded == 0
    assert not ungrounded(summary, article)


def test_an_invented_detail_is_caught():
    article = "Police said the fire began in the kitchen of a house in Bristol."
    summary = "A fire began in a Bristol kitchen, killing two firefighters."
    loose = check(summary, article)
    # Evidence quotes the summary's own words, not their stems.
    assert "firefighters" in loose.loose_tokens
    assert "2" in loose.loose_numbers
    # "two" is counted once, as a number, never also as a word.
    assert "two" not in loose.loose_tokens
    assert ungrounded(summary, article)


def test_plurals_are_not_hallucinations():
    # Morphology was the main source of false alarms before stemming.
    article = "Fifty pupil places were cancelled by the academy."
    summary = "The academy cancelled pupils' places."
    assert check(summary, article).loose_tokens == set()


def test_an_irregular_verb_defeats_the_stemmer_and_that_is_recorded():
    # A known false positive, kept as a test rather than hidden. "withdrew" and
    # "withdrawn" are the same verb and no suffix rule connects them, so the
    # checker calls a faithful summary ungrounded. A dictionary would fix it;
    # this module is deliberately cheap, and the cost of that choice is here.
    article = "Fifty pupil places were withdrawn by the academy."
    summary = "The academy withdrew pupils' places."
    assert check(summary, article).loose_tokens == {"withdrew"}


def test_a_written_out_number_is_a_number():
    # "three people died" against an article saying two is exactly the failure
    # this exists for, and it never reaches the digit pattern.
    assert numbers("three people died") == {"3"}
    assert numbers("50 pupils and twelve staff") == {"50", "12"}
    article = "Two people died in the crash."
    assert ungrounded("Three people died in the crash.", article)


def test_a_number_the_article_states_is_fine_however_written():
    article = "Two people died in the crash."
    assert not ungrounded("2 people died in the crash.", article)


def test_stopwords_never_count_as_evidence():
    assert tokens("the and of it was") == set()


def test_the_threshold_must_be_at_least_one():
    with pytest.raises(ValueError):
        ungrounded("a", "b", threshold=0)


def test_evidence_names_what_was_missing():
    article = "The council met on Tuesday."
    loose = check("The council met on Tuesday to discuss flooding.", article)
    assert "flooding" in loose.evidence
    assert "words not in the article" in loose.evidence


def test_evidence_says_so_when_nothing_is_missing():
    article = "The council met on Tuesday."
    assert "every content word traces" in check("The council met.", article).evidence
