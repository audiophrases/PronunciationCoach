"""The "how do I make this sound?" layer, from the GAPhonetics project.

GAPhonetics (a sibling repository, also published on GitHub Pages) holds the
full General American inventory - 24 vowels and 24 consonants - with, for each
sound, where the tongue and lips go, the sensation to feel, the mistake to
avoid, an example word, and human recordings of the sound and the word. The
coach detects that /ð/ came out as [d]; this module supplies the instruction,
the recordings, and a link that opens GAPhonetics on that exact contrast.

Data comes from the local checkout when present (PC_GAPHONETICS_DIR, or the
sibling folder), otherwise from the published site; audio is copied into the
cache on first use. Recordings are CC BY-SA / CC0 / public domain, credited in
GAPhonetics' *-audio-sources.json files.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("pronunciationcoach")

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "tts_cache" / "gaphonetics"
SITE_URL = os.environ.get("PC_GAPHONETICS_URL", "https://audiophrases.github.io/GAPhonetics/").rstrip("/") + "/"

# espeak-ng IPA (as the recogniser labels it) -> GAPhonetics key.
VOWEL_KEYS = {
    "iː": "i", "i": "i", "ɪ": "ɪ", "ɪɹ": "ɪr", "ɪə": "ɪr", "eɪ": "eɪ", "ɛ": "ɛ", "ɛɹ": "ɛr", "eə": "ɛr",
    "ɝ": "ɝ", "ɜː": "ɝ", "ɚ": "ɚ", "uː": "u", "ʊ": "ʊ", "ʊɹ": "ʊr", "ʊə": "ʊr", "ʌ": "ʌ",
    "ə": "ə", "ɐ": "ə", "ᵻ": "ə", "oʊ": "oʊ", "əʊ": "oʊ", "ɔː": "ɔ", "ɔ": "ɔ", "ɔːɹ": "ɔr", "æ": "æ",
    "aʊ": "aʊ", "aʊɚ": "aʊr", "aʊə": "aʊr", "aɪ": "aɪ", "aɪɚ": "aɪr", "aɪə": "aɪr", "ɑː": "ɑ2", "ɒ": "ɑ2",
    "ɑːɹ": "ɑr",
    # sounds learners bring along that GAPhonetics also places on its chart
    "a": "ɑ",  # the open Catalan/Spanish 'a' is GAPhonetics' low-front anchor
}
# Learner sounds with no GA entry: the nearest chart sound, so the contrast can still be shown.
NEAREST = {"oː": "ɔ", "o": "ɔ", "eː": "ɛ", "e": "ɛ", "y": "u", "yː": "u", "ɨ": "ɪ", "ɑ": "ɑ2", "ɛ̃": "ɛ", "ɑ̃": "ɑ2"}
CONSONANT_KEYS = {
    "p": "p", "b": "b", "t": "t", "ɾ": "t", "d": "d", "k": "k", "ɡ": "g", "g": "g", "f": "f", "v": "v",
    "θ": "θ", "ð": "ð", "s": "s", "z": "z", "ʃ": "ʃ", "ʒ": "ʒ", "h": "h", "tʃ": "tʃ", "dʒ": "dʒ",
    "m": "m", "n": "n", "n̩": "n", "ŋ": "ŋ", "l": "l", "ɫ": "l", "əl": "l", "ɹ": "ɹ", "r": "ɹ", "j": "j", "w": "w",
}

# GAPhonetics' preset minimal pairs (vowel keys) and a few consonant ones worth drilling.
MINIMAL_PAIRS = {
    frozenset({"i", "ɪ"}): "sheep / ship",
    frozenset({"ɛ", "æ"}): "bed / bad",
    frozenset({"ʊ", "u"}): "look / Luke",
    frozenset({"ʌ", "ɑ2"}): "cut / cot",
    frozenset({"ɑ2", "ɔ"}): "cot / caught",
    frozenset({"eɪ", "ɛ"}): "pain / pen",
    frozenset({"ð", "d"}): "they / day",
    frozenset({"ð", "z"}): "then / Zen",
    frozenset({"θ", "t"}): "thin / tin",
    frozenset({"θ", "s"}): "think / sink",
    frozenset({"v", "b"}): "very / berry",
    frozenset({"s", "z"}): "Sue / zoo",
    frozenset({"ʃ", "s"}): "ship / sip",
    frozenset({"dʒ", "j"}): "jet / yet",
    frozenset({"h", "ɹ"}): "hat / rat",
}


@dataclass
class Sound:
    key: str
    ipa: str
    kind: str  # "vowel" | "consonant"
    example: str
    how: list[str] = field(default_factory=list)  # instructions, in order
    mistake: str = ""
    phoneme_audio: str | None = None  # relative path within the site
    word_audio: str | None = None


@dataclass
class Guidance:
    target: Sound
    heard: Sound | None
    contrast: str  # one line placing the heard sound against the target, or ""
    practise: str  # minimal pair, or ""
    link: str  # GAPhonetics deep link


class Phonetics:
    def __init__(self, site_dir: Path | None, site_url: str = SITE_URL):
        self.site_dir = site_dir
        self.site_url = site_url
        self.sounds: dict[str, Sound] = {}
        self._load()

    # ------------------------------------------------------------------ loading
    def _read(self, rel: str) -> dict:
        if self.site_dir and (self.site_dir / rel).exists():
            return json.loads((self.site_dir / rel).read_text(encoding="utf-8"))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = CACHE_DIR / rel.replace("/", "_")
        if not cached.exists():
            with urllib.request.urlopen(self.site_url + rel, timeout=10) as r:
                cached.write_bytes(r.read())
        return json.loads(cached.read_text(encoding="utf-8"))

    def _load(self) -> None:
        for v in self._read("data/phonemes.json")["phonemes"]:
            art = v.get("articulatory", {})
            how = [art[k] for k in ("tongue", "jaw", "lips", "cue") if art.get(k)]
            key = v["key"]
            self.sounds[key] = Sound(
                key, v["ipa"], "vowel", (v.get("example") or [""])[0], how, "",
                f"audio/phonemes/{urllib.parse.quote(key)}.mp3",
                f"audio/words/{self._slug((v.get('example') or [''])[0])}.mp3" if v.get("example") else None,
            )
        for c in self._read("data/consonants.json")["consonants"]:
            how = [c["mouth"], *c.get("steps", []), c.get("cue", "")]
            self.sounds[c["key"]] = Sound(
                c["key"], c["ipa"], "consonant", (c.get("example") or [""])[0], [h for h in how if h],
                c.get("mistake", ""), c.get("phonemeAudio"), c.get("audio"),
            )

    @staticmethod
    def _slug(word: str) -> str:
        import re

        return re.sub(r"(^-|-$)", "", re.sub(r"[^a-z0-9]+", "-", word.lower().strip()))

    # ------------------------------------------------------------------ lookup
    def key_for(self, phone: str) -> str | None:
        return VOWEL_KEYS.get(phone) or CONSONANT_KEYS.get(phone) or (phone if phone in self.sounds else None)

    def sound(self, phone: str) -> Sound | None:
        key = self.key_for(phone)
        return self.sounds.get(key) if key else None

    def audio_path(self, rel: str | None) -> str | None:
        """Local file for a site-relative audio path, fetching it into the cache if needed."""
        if not rel:
            return None
        rel = urllib.parse.unquote(rel)
        if self.site_dir and (self.site_dir / rel).exists():
            return str(self.site_dir / rel)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = CACHE_DIR / rel.replace("/", "_")
        if not cached.exists():
            try:
                with urllib.request.urlopen(self.site_url + urllib.parse.quote(rel), timeout=10) as r:
                    cached.write_bytes(r.read())
            except Exception as exc:
                log.warning("GAPhonetics audio %s unavailable (%s)", rel, exc)
                return None
        return str(cached)

    def link(self, target: Sound, heard: Sound | None) -> str:
        mode = "vowels" if target.kind == "vowel" else "consonants"
        q = {"a": target.key}
        if heard and heard.kind == target.kind and heard.key != target.key:
            q["b"] = heard.key
        return f"{self.site_url}#{mode}?{urllib.parse.urlencode(q)}"

    def guidance(self, expected: str, heard: str | None) -> Guidance | None:
        target = self.sound(expected)
        if target is None:
            return None
        other = self.sound(heard) if heard else None
        approximate = False
        if other is None and heard in NEAREST:
            other, approximate = self.sounds.get(NEAREST[heard]), True
        if other is not None and other.key == target.key:
            other = None  # an allophone of the target (flap for t): nothing to contrast
        contrast = ""
        if other and approximate:
            contrast = (f"The sound you made ({heard}) is closest to /{other.ipa}/ (as in *{other.example}*) on the chart; "
                        f"compare it with the target /{target.ipa}/ (as in *{target.example}*).")
        elif other and other.kind == "vowel" == target.kind:
            contrast = (f"You made /{other.ipa}/ (as in *{other.example}*); the target /{target.ipa}/ (as in *{target.example}*) "
                        "is a different tongue position - compare them on the chart.")
        elif other and other.kind == "consonant" == target.kind:
            contrast = f"You made /{other.ipa}/ (as in *{other.example}*) instead of /{target.ipa}/ (as in *{target.example}*)."
        practise = MINIMAL_PAIRS.get(frozenset({target.key, other.key}), "") if other else ""
        return Guidance(target, other, contrast, practise, self.link(target, other))


def _site_dir() -> Path | None:
    for candidate in (os.environ.get("PC_GAPHONETICS_DIR"), ROOT.parent / "GAPhonetics" / "site"):
        if candidate and Path(candidate).joinpath("data", "phonemes.json").exists():
            return Path(candidate)
    return None


@lru_cache(maxsize=1)
def get_phonetics() -> Phonetics | None:
    try:
        return Phonetics(_site_dir())
    except Exception as exc:  # offline and no local checkout
        log.warning("GAPhonetics data unavailable (%s); sound guidance is off", exc)
        return None
