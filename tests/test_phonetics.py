"""GAPhonetics guidance mapping (uses the sibling checkout or the cached site data)."""

import pytest

from pronunciationcoach.phonetics import get_phonetics

ph = get_phonetics()
needs_data = pytest.mark.skipif(ph is None, reason="GAPhonetics data unavailable")


@needs_data
def test_inventory_and_symbol_mapping():
    assert len(ph.sounds) == 48
    assert ph.sound("iː").key == "i" and ph.sound("ɡ").key == "g" and ph.sound("ɾ").key == "t"
    assert ph.sound("a").key == "ɑ"  # the open 'a' anchor
    assert ph.sound("ʡ") is None


@needs_data
def test_guidance_for_th_as_d_has_instruction_mistake_pair_and_link():
    g = ph.guidance("ð", "d")
    assert g.target.example == "the" and g.heard.key == "d"
    assert any("teeth" in h for h in g.target.how)
    assert "d-like" in g.target.mistake
    assert g.practise == "they / day"
    assert g.link.endswith("#consonants?a=%C3%B0&b=d")


@needs_data
def test_guidance_maps_learner_sounds_to_nearest_chart_sound():
    g = ph.guidance("oʊ", "oː")
    assert g.heard.key == "ɔ" and "closest to" in g.contrast
    assert ph.guidance("ɾ", "t").heard is None  # allophone, nothing to contrast
    assert ph.guidance("æ", "a").link.endswith("#vowels?a=%C3%A6&b=%C9%91")
