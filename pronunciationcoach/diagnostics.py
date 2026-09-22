"""Small, credential-free setup snapshot for comparing recorded runs."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import shutil
import subprocess


def mfa_snapshot() -> dict:
    """Which aligner models a recording was cropped with, for machine comparisons."""
    try:
        from . import mfa, mfa_worker

        return {"installed": mfa_worker.available(), "model": mfa.MODEL,
                "dictionary": mfa.DICTIONARY, "offset_s": mfa.OFFSET_S,
                "fingerprints": mfa.model_fingerprints()}
    except Exception as exc:
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}


def runtime_snapshot() -> dict:
    from .asr import DEFAULT_ASR
    from .engine import DEFAULT_MODEL
    from .segmenter import MODEL_ID, TOKENIZER_ID
    from .tts import PLAYBACK_VOICES

    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ("torch", "torchaudio", "transformers", "faster-whisper", "ctranslate2",
                 "numpy", "gradio", "edge-tts", "phonemizer", "soundfile"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                  text=True, timeout=3, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=root, capture_output=True, text=True, timeout=3,
                                    check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    defaults = {"PC_ASR_MODEL": DEFAULT_ASR,
                "PC_CROP_RECHECK": "0", "PC_CHUNKS": "1", "PC_LISTENER": "1",
                "PC_NATIVE_REF": "1", "PC_CASUAL": "1", "PC_ACCENT": "American",
                "PC_MFA": "1", "PC_MFA_TIMEOUT": "30",
                "PC_VOICE": "Male", "HF_HUB_OFFLINE": "0"}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages,
            "git_revision": revision, "git_dirty": dirty,
            "settings": {key: os.environ.get(key, default) for key, default in defaults.items()},
            "phoneme_model": DEFAULT_MODEL, "segmenter": MODEL_ID, "segmenter_tokenizer": TOKENIZER_ID,
            "mfa": mfa_snapshot(),
            "playback_voices": PLAYBACK_VOICES, "ffmpeg": shutil.which("ffmpeg")}
