"""Regenerate pronunciationcoach/gradio_ui_english.json from the installed Gradio.

Gradio translates its own widget labels ("Record", "Submit", ...) to the browser's
language, with no switch to turn that off. The app keeps its UI in English by
registering the English strings as the "translation" for every other locale
(see app.py). This script pulls those strings and the locale list out of the
bundled frontend so they match the installed Gradio version exactly.

    uv run python scripts/extract_gradio_strings.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import gradio

ASSETS = Path(gradio.__file__).parent / "templates" / "frontend" / "assets"
OUT = Path(__file__).resolve().parents[1] / "pronunciationcoach" / "gradio_ui_english.json"

OBJECT = re.compile(r"\{((?:[A-Za-z_]+:`[^`]*`,?)+)\}")
PAIR = re.compile(r"([A-Za-z_]+):`([^`]*)`")
LOCALE_FILE = re.compile(r"^([a-z]{2,3}(?:-[A-Za-z]{2,4})?)-[A-Za-z0-9_-]{6,}\.js$")


def sections_of(lang_js: str) -> dict[str, dict[str, str]]:
    """A language file assigns each section to a variable and exports `var as name`."""
    exports = dict(re.findall(r"\b([A-Za-z]+) as ([A-Za-z_]+)", lang_js.split("export{")[-1]))
    out = {}
    for var, name in exports.items():
        m = re.search(r"(?:var |,)" + re.escape(var) + r"=(\{(?:[A-Za-z_]+:`[^`]*`,?)+\})", lang_js)
        if m:
            out[name] = dict(PAIR.findall(m.group(1)))
    return out


def main() -> None:
    core = next(ASSETS.glob("core-*.js")).read_text(encoding="utf-8")
    english_blocks = [dict(PAIR.findall(b)) for b in OBJECT.findall(core)]

    # The Catalan file is a convenient full key list; any complete language file would do.
    reference = sections_of(next(ASSETS.glob("ca-*.js")).read_text(encoding="utf-8"))

    strings: dict[str, str] = {}
    for section, keys in reference.items():
        want = set(keys)
        candidates = [b for b in english_blocks if want <= set(b)]
        if not candidates:
            print(f"  ! no English block covers section {section!r} ({sorted(want)})", file=sys.stderr)
            continue
        block = min(candidates, key=len)  # tightest superset is the real section
        for key in want:
            strings[f"{section}.{key}"] = block[key]

    # Language files export `_name` (the language's own name); other assets with locale-like names don't.
    locales = sorted(
        {
            m.group(1)
            for f in ASSETS.iterdir()
            if (m := LOCALE_FILE.match(f.name)) and " as _name" in f.read_text(encoding="utf-8", errors="ignore")
        }
        - {"en"}
    )
    OUT.write_text(
        json.dumps({"gradio_version": gradio.__version__, "locales": locales, "strings": strings}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"{len(strings)} strings, {len(locales)} locales, Gradio {gradio.__version__} -> {OUT.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
