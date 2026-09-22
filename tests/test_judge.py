"""The parser between the model and the headline number.

`quote_is_real` is the one measurement this half exists to make: the model is
required to quote the words it objects to, and that quote is checkable with no
judgement at all. Everything upstream of it is parsing, and parsing is where
this project has been wrong — twice, both times in the same direction, both
times turning an abstention into "the detector invented its evidence".

So the cases below are not invented. They are the exact strings the 14B
produced during the 600-item run, kept as fixtures.
"""

from __future__ import annotations

import pytest

from faithful import judge


def answer(verdict: str, quote: str) -> str:
    return f"VERDICT: {verdict}\nQUOTE: {quote}"


def test_a_plain_quote_that_is_in_the_summary_is_real():
    summary = "The man died in hospital on Tuesday."
    v = judge.parse(answer("UNFAITHFUL", "the man died in hospital"), summary)
    assert not v.faithful
    assert v.quote_is_real
    assert not v.abstained


def test_a_quote_the_summary_never_contained_is_not_real():
    v = judge.parse(answer("UNFAITHFUL", "he was arrested in Glasgow"), "A fire broke out.")
    assert not v.quote_is_real


def test_punctuation_and_case_are_not_counted_as_invention():
    """The model re-punctuates constantly. Scoring that as a hallucination
    would measure its typography rather than its reasoning."""
    summary = "Ms Ahmed, 41, said the council “acted too late”."
    v = judge.parse(answer("UNFAITHFUL", '"Acted Too Late!"'), summary)
    assert v.quote_is_real


# --- the two parser bugs, as they actually appeared -------------------------


def test_commentary_after_the_quote_is_not_part_of_the_quote():
    """Bug one, from the run: the model quotes correctly, then explains itself.

        QUOTE: "dpp accepts 77 allegations" (the correct name is "Ratcliffe")

    Taking the line whole produces a string that is not in the summary, and the
    row is scored as invented evidence when the model quoted exactly right.
    """
    summary = "The dpp accepts 77 allegations were made against the officer."
    raw = answer(
        "UNFAITHFUL",
        '"dpp accepts 77 allegations" (the correct name is "Ratcliffe" and '
        '"dpp" is not mentioned in the article)',
    )
    v = judge.parse(raw, summary)
    assert v.quote == "dpp accepts 77 allegations"
    assert v.quote_is_real


def test_the_instruction_echoed_back_is_an_abstention_not_a_quote():
    """Bug two, from the run: the model restates the question before answering.

        QUOTE: the exact words from the SUMMARY that are unsupported, or NONE:
               NONE (but the summary is inaccurate as it ...)

    Its structured answer is NONE. Read literally, the prompt's own wording
    becomes the model's quote, is of course absent from the summary, and an
    abstention is recorded as the detector inventing evidence.
    """
    raw = answer(
        "UNFAITHFUL",
        "the exact words from the SUMMARY that are unsupported, or NONE: NONE "
        '(but the summary is inaccurate as it incorrectly states "found guilty '
        'of fraud after he was found guilty of fraud")',
    )
    v = judge.parse(raw, "He was found guilty of fraud.")
    assert v.quote == judge.NONE
    assert v.abstained
    assert v.quote_is_real, "an abstention has no evidence to be false about"


def test_a_summary_may_genuinely_begin_with_the_word_none():
    """The guard on the fix above.

    Collapsing any leading NONE into an abstention would silently discard real
    quotes from summaries that open a sentence with the word.
    """
    summary = "None of the passengers survived the crash."
    v = judge.parse(answer("UNFAITHFUL", "None of the passengers survived"), summary)
    assert v.quote == "None of the passengers survived"
    assert not v.abstained
    assert v.quote_is_real


# --- verdicts ---------------------------------------------------------------


def test_none_with_a_faithful_verdict_is_not_an_abstention():
    v = judge.parse(answer("FAITHFUL", "NONE"), "Anything at all.")
    assert v.faithful
    assert not v.abstained
    assert v.quote_is_real


def test_unfaithful_with_no_words_named_is_an_abstention():
    v = judge.parse(answer("UNFAITHFUL", "NONE"), "Anything at all.")
    assert v.abstained


@pytest.mark.parametrize("raw", ["", "I cannot help with that.", "VERDICT: MAYBE"])
def test_an_unparseable_answer_counts_as_faithful(raw):
    """The conservative default, and it is load-bearing.

    This measures a detector. An answer that produced no usable finding did not
    find anything; treating garbage as a positive would inflate recall with
    noise and make a broken run look like a sensitive one.
    """
    assert judge.parse(raw, "A summary.").faithful


def test_a_test_can_never_reach_a_live_model_by_accident():
    """An hour of GPU is a bad thing to spend on a typo in a fixture."""
    model = judge.Recorded({"known prompt": "VERDICT: FAITHFUL\nQUOTE: NONE"})
    with pytest.raises(judge.ModelUnavailableError):
        model("something it was never told about")
