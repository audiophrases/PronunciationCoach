"""Learner-facing feedback text, built from hand-made scores (no model needed)."""

from pronunciationcoach.feedback import describe, phone_tip, summary, word_feedback
from pronunciationcoach.scoring import PhoneScore, WordScore


def ps(expected, heard, gop, dropped=False):
    return PhoneScore(expected=expected, start_s=0.0, end_s=0.02, gop=gop, posterior=0.5, heard=heard, dropped=dropped)


def test_describe_uses_plain_spelling_and_example():
    assert describe("ð") == "'th' as in *this*"
    assert describe("æ") == "'a' as in *cat*"
    assert describe("ʁ") == "the French 'r'"
    assert describe("a") == "the Catalan/Spanish 'a'"
    assert describe("ə").startswith("the weak 'uh' sound")
    assert describe("ʡ") == "'ʡ'"  # unknown symbol degrades gracefully


def test_phone_tips_read_naturally():
    assert phone_tip(ps("ð", "z", -6.0), "than") == "In *than*, 'th' as in *this* came out as 'z' as in *zoo*."
    assert phone_tip(ps("t", "ʁ", -1.5, dropped=True), "accent").startswith("The 't' as in *ten* sound in *accent* was not heard")
    assert phone_tip(ps("ə", "ɪ", -1.0), "gonna").startswith("The weak 'uh' sound (as in the first syllable of *about*) in *gonna* was not quite clear")
    assert "not quite clear" in phone_tip(ps("aɪ", "a", -1.0), "try")


def test_word_feedback_bands_and_tips():
    clear = word_feedback(WordScore("this", [ps("ð", "ð", 0.0), ps("ɪ", "ɪ", 0.0), ps("s", "s", 0.0)]))
    assert clear.band == "clear" and clear.tips == []
    bad = word_feedback(WordScore("than", [ps("ð", "z", -8.0), ps("ɐ", "ɛ", -4.0), ps("n", "n", 0.0)]))
    assert bad.band == "work on this" and len(bad.tips) == 2


def test_summary_counts_words_and_groups_recurring_sounds():
    words = [
        WordScore("this", [ps("ð", "ð", 0.0)]),
        WordScore("than", [ps("ð", "z", -8.0)]),
        WordScore("other", [ps("ð", "z", -7.0)]),
        WordScore("accent", [ps("t", "ʁ", -1.5, dropped=True)]),
    ]
    text = summary(words)
    assert text.startswith("Understood: 4 of 4 words. Clear: 1")
    assert "Work on:" in text and "Accent notes" in text
    assert "'th' as in *this* came out as 'z' as in *zoo* (2 times)" in text
    assert "'t' as in *ten* was missing" in text
    mixed = summary([WordScore("than", [ps("ð", "z", -8.0)]), WordScore("other", [ps("ð", "d", -7.0)])])
    assert "'th' as in *this* came out as 'z' as in *zoo* or 'd' as in *dog* (2 times)" in mixed
    assert summary([WordScore("hi", [ps("h", "h", 0.0)])]) == "All 1 words were clear. Nice work!"


def test_summary_ranks_th_above_many_unreduced_vowels_and_merges_weak_vowels():
    words = [
        WordScore("an", [ps("ɐ", "a", -3.0)]),
        WordScore("accent", [ps("ə", "ɛ", -4.0)]),
        WordScore("banana", [ps("ə", "a", -4.0)]),
        WordScore("than", [ps("ð", "z", -8.0)]),
        WordScore("other", [ps("ð", "d", -7.0)]),
    ]
    lines = summary(words).split("\n")
    practise = [ln for ln in lines if ln.startswith("•")]
    assert practise[0].startswith("• 'th' as in *this*"), practise
    assert sum("weak 'uh'" in ln for ln in practise) == 1, practise
    assert "(3 times)" in [ln for ln in practise if "weak 'uh'" in ln][0]
