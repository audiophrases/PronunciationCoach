"""Word verdicts from listener confidence and weighted severity (no model)."""

from pronunciationcoach.asr import HeardWord
from pronunciationcoach.listener import match_words
from pronunciationcoach.scoring import PhoneScore, WordScore


def ps(expected, heard, gop, dropped=False):
    return PhoneScore(expected=expected, start_s=0.0, end_s=0.02, gop=gop, posterior=0.5, heard=heard, dropped=dropped)


def hw(text, p):
    return HeardWord(text, p, 0.0, 0.1)


def test_match_words_carries_probabilities_and_marks_misses():
    ref = ["I", "don't", "know", "the", "app's"]
    heard = [hw("I", 0.95), hw("don't", 0.7), hw("no", 0.4), hw("the", 0.9), hw("apps", 0.8)]
    out = match_words(ref, heard)
    assert [o.matched for o in out] == [True, True, False, True, False]
    assert out[1].probability == 0.7 and out[2].probability == 0.0
    assert out[1].confident and not out[2].understood


def test_verdicts_separate_accent_from_problems():
    # o for oʊ, listener confident -> accent colouring, not an error
    so = WordScore("so", [ps("s", "s", 0.0), ps("oʊ", "oː", -2.7)], listener_p=0.95, understood=True)
    assert so.verdict == "accent"
    # same deviation, listener hesitant -> almost
    so.listener_p = 0.5
    assert so.verdict == "almost"
    # th -> z is always work, however confident the listener
    than = WordScore("than", [ps("ð", "z", -8.0), ps("ə", "ə", 0.0), ps("n", "n", 0.0)], listener_p=0.99, understood=True)
    assert than.verdict == "work on this"
    # æ -> a: work when the listener was not confident, almost when it was
    that = WordScore("that", [ps("ð", "ð", 0.0), ps("æ", "a", -4.7), ps("t", "t", 0.0)], listener_p=0.7, understood=True)
    assert that.verdict == "work on this"
    that.listener_p = 0.9
    assert that.verdict == "almost"
    # listener missed a word that also had a deviation
    missed = WordScore("things", [ps("θ", "θ", 0.0), ps("ɪ", "iː", -3.0)], listener_p=0.0, understood=False)
    assert missed.verdict == "work on this"
    # nothing flagged
    assert WordScore("hi", [ps("h", "h", 0.0)], listener_p=0.9, understood=True).verdict == "clear"


def test_listener_alone_cannot_fail_a_word_whose_sounds_were_all_right():
    # "see" said perfectly; Whisper split its confidence over see / sea (free-speech test, 16 %)
    see = WordScore("see", [ps("s", "s", 0.0), ps("iː", "iː", 0.0)], listener_p=0.16, understood=True)
    assert see.verdict == "clear" and see.listener_doubt == "hesitated"
    # every sound right but the listener wrote another word (known-sentence mode): worth a look
    see.listener_p, see.understood = 0.0, False
    assert see.verdict == "almost" and see.listener_doubt == "missed"
    # the same low confidence with a real deviation still counts against the word
    sea = WordScore("see", [ps("s", "s", 0.0), ps("iː", "ɪ", -3.0)], listener_p=0.16, understood=True)
    assert sea.verdict == "work on this" and sea.listener_doubt is None
    # no listener at all: falls back to severity only
    assert WordScore("so", [ps("oʊ", "oː", -2.7)]).verdict == "almost"
