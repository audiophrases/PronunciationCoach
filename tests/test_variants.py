"""Connected-speech acceptances from variants.py (no model, no network)."""

from pronunciationcoach.g2p import WordPhones
from pronunciationcoach.reference import acceptances
from pronunciationcoach.variants import casual_renderings


def wp(word, phones):
    return WordPhones(word, phones.split())


def test_going_to_merges_into_gonna_and_ona():
    canon = [wp("I'm", "aɪ m"), wp("going", "ɡ oʊ ɪ ŋ"), wp("to", "t ə"), wp("go", "ɡ oʊ")]
    acc = acceptances(canon, None, "en-us", casual=True)
    going = acc[1]
    assert "ə" in going[1].also or "ʌ" in going[1].also  # oʊ -> schwa in "gonna"
    assert going[3].optional or "n" in going[3].also  # the ŋ gives way to n / disappears
    to = acc[2]
    assert to[0].optional  # "to" loses its t in "gonna"


def test_casual_gonna_allows_dropping_the_g():
    canon = [wp("I'm", "aɪ m"), wp("gonna", "ɡ ə n ə")]
    acc = acceptances(canon, None, "en-us", casual=True)
    assert acc[1][0].optional  # 'ona


def test_dont_know_accepts_dunno_and_the_dropped_t():
    canon = [wp("I", "aɪ"), wp("don't", "d oʊ n t"), wp("know", "n oʊ")]
    acc = acceptances(canon, None, "en-us", casual=True)
    dont = acc[1]
    assert "ə" in dont[1].also  # oʊ -> ə (dunno)
    assert dont[3].optional  # final t before a consonant


def test_rules_did_you_and_in_the_and_next_day():
    canon = [wp("did", "d ɪ d"), wp("you", "j uː")]
    acc = acceptances(canon, None, "en-us", casual=True)
    assert "dʒ" in acc[0][2].also and acc[1][0].optional
    canon = [wp("in", "ɪ n"), wp("the", "ð ə")]
    acc = acceptances(canon, None, "en-us", casual=True)
    assert "n" in acc[1][0].also
    canon = [wp("next", "n ɛ k s t"), wp("day", "d eɪ")]
    acc = acceptances(canon, None, "en-us", casual=True)
    assert acc[0][4].optional


def test_careful_mode_accepts_nothing_extra():
    canon = [wp("going", "ɡ oʊ ɪ ŋ"), wp("to", "t ə")]
    acc = acceptances(canon, None, "en-us", casual=False)
    assert all(not a.also and not a.optional for word in acc for a in word)


def test_renderings_keep_untouched_words_canonical():
    canon = [wp("I", "aɪ"), wp("and", "æ n d"), wp("you", "j uː")]
    r = casual_renderings(canon)
    assert r["casual:0"][0] == ["aɪ"] and r["casual:0"][1] == ["ə", "n"] and r["casual:0"][2] == ["j", "ə"]
