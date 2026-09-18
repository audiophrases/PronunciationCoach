"""G2P checks. These need espeak-ng installed (they are skipped otherwise)."""

import pytest

from pronunciationcoach.g2p import text_to_phones, words_of

try:
    text_to_phones("hello", "en-us")
    HAVE_ESPEAK = True
except RuntimeError:
    HAVE_ESPEAK = False

needs_espeak = pytest.mark.skipif(not HAVE_ESPEAK, reason="espeak-ng not installed")


def test_words_of_keeps_apostrophes_and_drops_punctuation_and_digits():
    assert words_of("Hello, this is a test. Don't stop at 21!") == ["Hello", "this", "is", "a", "test", "Don't", "stop", "at"]


@needs_espeak
def test_every_word_gets_its_own_phones():
    result = text_to_phones("Hello, this is a test. Hi.", "en-us")
    assert [w.word for w in result] == ["Hello", "this", "is", "a", "test", "Hi"]
    assert all(w.phones for w in result)


@needs_espeak
def test_function_words_get_weak_forms_in_context():
    by_word = {w.word: w.phones for w in text_to_phones("this is a test", "en-us")}
    assert by_word["a"] != ["eɪ"], "'a' must not be read as the letter name"
    the = [w.phones for w in text_to_phones("the apple and the pear", "en-us") if w.word == "the"]
    assert the[0] != the[1], "'the' should differ before a vowel and before a consonant"


@needs_espeak
def test_accents_differ_where_expected():
    us = {w.word: w.phones for w in text_to_phones("the water was cold", "en-us")}
    gb = {w.word: w.phones for w in text_to_phones("the water was cold", "en-gb")}
    assert us["water"] == ["w", "ɔː", "ɾ", "ɚ"]  # flap + r-coloured schwa
    assert gb["water"] == ["w", "ɔː", "t", "ə"]  # no r before a consonant


@needs_espeak
def test_doubled_r_is_removed_but_british_linking_r_is_kept():
    us = {w.word: w.phones for w in text_to_phones("curiosity", "en-us")}
    assert us["curiosity"][:4] == ["k", "j", "ʊɹ", "ɪ"], "espeak's ʊɹ ɹ must collapse to one r"
    us = {w.word: w.phones for w in text_to_phones("the water is cold", "en-us")}
    assert us["water"] == ["w", "ɔː", "ɾ", "ɚ"], "ɚ already carries the r"
    gb = {w.word: w.phones for w in text_to_phones("the water is cold", "en-gb")}
    assert gb["water"] == ["w", "ɔː", "t", "ə", "ɹ"], "British linking r before a vowel"


@needs_espeak
def test_fused_pair_does_not_break_word_mapping():
    text = "This sentence is far too long for a three second clip"
    result = text_to_phones(text, "en-us")
    assert [w.word for w in result] == text.split()
    by_word = {w.word: w.phones for w in result}
    assert by_word["a"] in (["ə"], ["ɐ"]), "weak form even when espeak fuses 'for a'"
    assert by_word["three"] == ["θ", "ɹ", "iː"]
