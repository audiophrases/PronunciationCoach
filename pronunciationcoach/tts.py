"""Model pronunciation for the learner to compare against, and for the scorer to learn from.

Same source as the user's DictationApp and password projects: Microsoft Edge's
"Read Aloud" neural voices, reached through the `edge-tts` package (no key, no
account). It needs the internet; when that fails, espeak-ng speaks instead so
the button never dies - it just sounds robotic.

Synthesised audio is cached on disk, keyed by voice, speed and text, so a class
working on one sentence costs one synthesis, not one per student per replay.
Edge also reports where each word starts and ends in the audio; that is kept
next to the mp3 and is what reference.py uses to listen to natives word by word.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("pronunciationcoach")

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "tts_cache"

# One natural voice per target accent for playback, and the voices whose renderings the
# scorer treats as native references (a female and a male voice per accent).
VOICES = {"en-us": "en-US-JennyNeural", "en-gb": "en-GB-SoniaNeural"}
REFERENCE_VOICES = {
    "en-us": ["en-US-JennyNeural", "en-US-GuyNeural"],
    "en-gb": ["en-GB-SoniaNeural", "en-GB-RyanNeural"],
}

# Edge TTS rate strings. "Slow" is what a teacher does when modelling a word.
SPEEDS = {"normal": "+0%", "slow": "-35%"}

_ESPEAK_EXE = "espeak-ng" if sys.platform != "win32" else r"C:\Program Files\eSpeak NG\espeak-ng.exe"


@dataclass
class WordBoundary:
    text: str
    start: float
    end: float


def _key(voice: str, rate: str, text: str) -> str:
    return hashlib.sha1(f"{voice}|{rate}|{text}".encode("utf-8")).hexdigest()[:20]


async def _stream(text: str, voice: str, rate: str, mp3: Path) -> list[WordBoundary]:
    import edge_tts

    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    words: list[WordBoundary] = []
    tmp = mp3.with_suffix(".part")
    with open(tmp, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7
                words.append(WordBoundary(chunk["text"], start, start + chunk["duration"] / 1e7))
    tmp.replace(mp3)
    return words


def synthesize(
    text: str, lang: str = "en-us", speed: str = "normal", voice: str | None = None, with_words: bool = False
) -> Path:
    """Return an audio file (mp3, or wav from the fallback) saying `text`.

    `with_words` insists on Edge's word timings being available next to the file
    (files cached before timings were kept are synthesised again).
    """
    text = " ".join(text.split())
    if not text:
        raise ValueError("nothing to say")
    voice = voice or VOICES.get(lang, VOICES["en-us"])
    rate = SPEEDS.get(speed, SPEEDS["normal"])
    CACHE_DIR.mkdir(exist_ok=True)
    key = _key(voice, rate, text)
    mp3 = CACHE_DIR / f"{key}.mp3"
    wav = CACHE_DIR / f"{key}.wav"
    if mp3.exists() and (not with_words or mp3.with_suffix(".words.json").exists()):
        return mp3
    if wav.exists() and not with_words:
        return wav

    try:
        words = asyncio.run(_stream(text, voice, rate, mp3))
        mp3.with_suffix(".words.json").write_text(
            json.dumps([w.__dict__ for w in words], ensure_ascii=False), encoding="utf-8"
        )
        return mp3
    except Exception as exc:  # offline, endpoint changed, ...
        log.warning("edge-tts failed (%s); falling back to espeak-ng for %r", exc, text)

    speed_wpm = "150" if speed == "normal" else "100"
    subprocess.run([_ESPEAK_EXE, "-v", lang, "-s", speed_wpm, "-w", str(wav), text], check=True, capture_output=True)
    return wav


def word_boundaries(audio_path: Path) -> list[WordBoundary] | None:
    """Edge's word timings for a synthesised file, or None for the espeak fallback."""
    sidecar = audio_path.with_suffix(".words.json")
    if audio_path.suffix != ".mp3" or not sidecar.exists():
        return None
    return [WordBoundary(**w) for w in json.loads(sidecar.read_text(encoding="utf-8"))]
