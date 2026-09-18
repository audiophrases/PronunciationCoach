"""Make sure phonemizer can find espeak-ng before anything imports it.

phonemizer finds espeak-ng on Linux/macOS by itself. On Windows it needs the
DLL path in PHONEMIZER_ESPEAK_LIBRARY, and the recogniser's tokenizer creates
a phonemizer backend as soon as it is constructed, so this must run early.
"""

from __future__ import annotations

import os
import sys

_WINDOWS_DLLS = (
    r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
    r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll",
)


def ensure_espeak() -> None:
    if sys.platform == "win32" and "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
        for dll in _WINDOWS_DLLS:
            if os.path.exists(dll):
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = dll
                return
