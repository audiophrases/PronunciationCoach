"""Model pronunciation for the learner to compare against.

Same source as the user's DictationApp and password projects: Microsoft Edge's
"Read Aloud" neural voices, reached through the `edge-tts` package (no key, no
account). It needs the internet; when that fails, espeak-ng speaks instead so
the button never dies - it just sounds robotic.

Synthesised audio is cached on disk, keyed by voice, speed and text, so a class
working on one sentence costs one synthesis, not one per student per replay.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("pronunciationcoach")

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "tts_cache"

# One natural voice per target accent. Others worth trying: en-US-AriaNeural,
# en-US-GuyNeural, en-GB-RyanNeural (male).
VOICES = {"en-us": "en-US-JennyNeural", "en-gb": "en-GB-SoniaNeural"}

# Edge TTS rate strings. "Slow" is what a teacher does when modelling a word.
SPEEDS = {"normal": "+0%", "slow": "-35%"}

_ESPEAK_EXE = "espeak-ng" if sys.platform != "win32" else r"C:\Program Files\eSpeak NG\espeak-ng.exe"


def synthesize(text: str, lang: str = "en-us", speed: str = "normal") -> Path:
    """Return an audio file (mp3, or wav from the fallback) saying `text`."""
    text = " ".join(text.split())
    if not text:
        raise ValueError("nothing to say")
    voice = VOICES.get(lang, VOICES["en-us"])
    rate = SPEEDS.get(speed, SPEEDS["normal"])
    CACHE_DIR.mkdir(exist_ok=True)
    key = hashlib.sha1(f"{voice}|{rate}|{text}".encode("utf-8")).hexdigest()[:20]
    mp3 = CACHE_DIR / f"{key}.mp3"
    wav = CACHE_DIR / f"{key}.wav"
    if mp3.exists():
        return mp3
    if wav.exists():
        return wav

    try:
        import edge_tts

        tmp = mp3.with_suffix(".part")
        edge_tts.Communicate(text, voice, rate=rate).save_sync(str(tmp))
        tmp.replace(mp3)
        return mp3
    except Exception as exc:  # offline, endpoint changed, ...
        log.warning("edge-tts failed (%s); falling back to espeak-ng for %r", exc, text)

    speed_wpm = "150" if speed == "normal" else "100"
    subprocess.run([_ESPEAK_EXE, "-v", lang, "-s", speed_wpm, "-w", str(wav), text], check=True, capture_output=True)
    return wav
