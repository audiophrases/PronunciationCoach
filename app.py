"""Gradio front end.

Learner view: a score ring, the sentence as tappable word chips, one player that
speaks whatever was tapped (you, the model, one word, one sound), and a word
panel with the word's sounds as tappable pills and one line of advice each.
Teacher view: everything technical (timeline, IPA, per-phone table, posterior
heatmap) in a collapsed section underneath.
"""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path

import gradio as gr
import numpy as np

from pronunciationcoach import SAMPLE_RATE
from pronunciationcoach.asr import DEFAULT_ASR
from pronunciationcoach.audio import to_mono_16k
from pronunciationcoach.engine import DEFAULT_MODEL
from pronunciationcoach.feedback import BAND_COLOR, phone_tip, sound_pills, summary, word_feedback
from pronunciationcoach.g2p import ACCENTS, DEFAULT_ACCENT
from pronunciationcoach.logs import DEBUG, LOG_FILE, SAVE_RECORDINGS, log_assessment, setup_logging
from pronunciationcoach.pipeline import assess
from pronunciationcoach.scoring import GOP_GOOD, GOP_UNSURE
from pronunciationcoach.tts import synthesize
from pronunciationcoach.viz import posterior_heatmap, timeline_figure

log = setup_logging()
IPA_COLORS = {"good": "#2e8b57", "unsure": "#e0a800", "off": "#c0392b"}
L1_OPTIONS = ["Catalan", "Spanish", "Other / unknown"]  # plumbed through for the next stage; unused today
DEFAULT_L1 = os.environ.get("PC_L1", "Catalan")
if DEFAULT_L1 not in L1_OPTIONS:
    raise ValueError(f"PC_L1 must be one of {L1_OPTIONS}, got {DEFAULT_L1!r}")

CSS = """
#sentence .textfield { line-height: 2.9; font-size: 1.55rem; }
#sentence .token.highlighted { padding: .38rem .75rem; border-radius: .75rem; margin: 0 .12rem;
    cursor: pointer; font-weight: 600; transition: transform .08s; }
#sentence .token.highlighted:hover { transform: scale(1.06); }
#sounds .textfield { line-height: 2.6; font-size: 1.3rem; }
#sounds .token.highlighted { padding: .3rem .65rem; border-radius: .65rem; margin: 0 .1rem;
    cursor: pointer; font-weight: 600; font-family: ui-monospace, Menlo, Consolas, monospace; }
#sounds .token.highlighted:hover { transform: scale(1.06); }
.score-card { display: flex; gap: 1.2rem; align-items: center; padding: .6rem .2rem; }
.score-card .ring { width: 110px; height: 110px; flex: none; }
.score-card .ring .bg { fill: none; stroke: #e6e6e6; stroke-width: 3.2; }
.score-card .ring .fg { fill: none; stroke-width: 3.2; stroke-linecap: round; }
.score-card .ring text { font-size: .55rem; font-weight: 700; text-anchor: middle; fill: currentColor; }
.score-card .ring .sub { font-size: .26rem; font-weight: 500; opacity: .7; }
.score-card ul { margin: .3rem 0 0 1rem; padding: 0; }
.score-card li { margin: .15rem 0; }
.score-card .headline { font-size: 1.05rem; font-weight: 600; margin-bottom: .2rem; }
.hint { opacity: .7; font-size: .9rem; }
"""


def english_ui() -> gr.I18n:
    """Keep Gradio's own widget labels ("Record", "Submit", ...) in English.

    Gradio localises them from the browser's language and has no off switch, but
    custom translations given as flat "section.key" strings take precedence over
    its built-in dictionaries. So the English strings are registered as the
    translation for every other locale. Regenerate the JSON after a Gradio
    upgrade with scripts/extract_gradio_strings.py.
    """
    data = json.loads((Path(__file__).parent / "pronunciationcoach" / "gradio_ui_english.json").read_text(encoding="utf-8"))
    if data["gradio_version"] != gr.__version__:
        log.warning("gradio_ui_english.json was made for Gradio %s, running %s", data["gradio_version"], gr.__version__)
    translations = {locale: data["strings"] for locale in data["locales"]}
    return gr.I18n(**translations)


def model_audio(text: str, lang: str, speed: str) -> str | None:
    try:
        return str(synthesize(text, lang, speed))
    except Exception:
        log.exception("could not synthesise %r", text)
        return None


def clip(audio: np.ndarray, start: float, end: float, pad: float = 0.0) -> tuple[int, np.ndarray]:
    a = max(0, int((start - pad) * SAMPLE_RATE))
    b = min(len(audio), int((end + pad) * SAMPLE_RATE))
    return SAMPLE_RATE, audio[a:b]


def score_card_html(words, summary_text: str) -> str:
    n = len(words)
    clear = sum(w.category == "good" for w in words)
    pct = 100.0 * clear / n if n else 0.0
    color = "#2e8b57" if pct >= 80 else ("#e0a800" if pct >= 50 else "#c0392b")
    lines = [ln[2:].strip() for ln in summary_text.splitlines() if ln.startswith("•")]
    items = "".join(f"<li>{escape(ln).replace('*', '')}</li>" for ln in lines)
    headline = "All clear - nice work!" if clear == n and n else f"{clear} of {n} words clear"
    practise = f"<div class='headline'>Sounds to practise</div><ul>{items}</ul>" if items else ""
    ring = (
        "<svg class='ring' viewBox='0 0 36 36'>"
        "<path class='bg' d='M18 2.0845a15.9155 15.9155 0 0 1 0 31.831a15.9155 15.9155 0 0 1 0-31.831'/>"
        f"<path class='fg' stroke='{color}' stroke-dasharray='{pct:.0f}, 100' d='M18 2.0845a15.9155 15.9155 0 0 1 0 31.831a15.9155 15.9155 0 0 1 0-31.831'/>"
        f"<text x='18' y='19.5'>{clear}/{n}</text><text class='sub' x='18' y='24'>words clear</text></svg>"
    )
    return f"<div class='score-card'>{ring}<div><div class='headline'>{escape(headline)}</div>{practise}</div></div>"


def run(audio, text, accent_name, l1):
    if audio is None:
        log.warning("assess called without audio")
        raise gr.Error("Record or upload some audio first.")
    sr, samples = audio
    lang = ACCENTS[accent_name]
    log.debug("request: sr=%d samples=%s text=%r accent=%s L1=%s", sr, getattr(samples, "shape", None), text, accent_name, l1)
    try:
        result = assess(samples, sr, text, lang)
    except ValueError as exc:  # e.g. audio far too short for the sentence
        log.warning("assessment rejected: %s", exc)
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        log.exception("assessment failed")
        raise gr.Error(f"Something went wrong: {exc!r}. Details are in {LOG_FILE}") from exc
    audio16k = result.audio
    log_assessment(log, result, l1, audio16k)

    # --- learner view ---------------------------------------------------------
    feedback = [word_feedback(w) for w in result.words]
    words_hl: list[tuple[str, str | None]] = []
    word_index: list[int | None] = []
    for i, f in enumerate(feedback):
        words_hl.append((f.word, f.band))
        word_index.append(i)
        words_hl.append((" ", None))
        word_index.append(None)

    summary_text = summary(result.words)
    card = score_card_html(result.words, summary_text)
    note = ""
    if result.transcribed:
        note += f"I heard: **{result.text}**  "
    if result.extra_text():
        note += "<span class='hint'>Some repeated or hesitated parts were ignored.</span>"

    w_spans = result.word_spans()
    p_spans = result.phone_spans()
    state = {
        "lang": lang,
        "text": result.text,
        "audio": audio16k,
        "word_index": word_index,
        "words": [
            {
                "word": w.word,
                "band": f.band,
                "span": (sp.start, sp.end),
                "pills": sound_pills(w),
                "phone_spans": [None if s is None else (s.start, s.end) for s in ps],
                "tips": [phone_tip(p, w.word) if p.category != "good" else "" for p in w.phones],
                "expected": [p.expected for p in w.phones],
                "heard": [p.heard_label for p in w.phones],
            }
            for w, f, sp, ps in zip(result.words, feedback, w_spans, p_spans)
        ],
        "current": None,
    }

    # --- teacher view ---------------------------------------------------------
    ipa_hl = []
    for word in result.words:
        for p in word.phones:
            ipa_hl.append((p.expected, p.category))
        ipa_hl.append(("  ", None))
    rows = []
    for word, ps in zip(result.words, p_spans):
        for i, (p, s) in enumerate(zip(word.phones, ps)):
            rows.append(
                [
                    word.word if i == 0 else "",
                    p.expected,
                    f"{p.start_s:.2f}",
                    "" if s is None else f"{s.start:.2f}-{s.end:.2f}",
                    f"{p.gop:.2f}",
                    f"{p.posterior:.2f}",
                    p.heard_label,
                    "  ".join(f"{ph} {pr:.2f}" for ph, pr in p.candidates),
                ]
            )
    timing = "  ".join(f"{k} {v:.1f}s" for k, v in result.timings.items())
    tech = (
        f"Reference{' (Whisper)' if result.transcribed else ''}: {result.text}\n"
        f"Heard, text-independent: {result.heard_text}\n"
        + (f"Heard but not in the sentence: {result.extra_text()}\n" if result.extra_text() else "")
        + (f"No model label for: {' '.join(result.unknown_phones)}\n" if result.unknown_phones else "")
        + f"{result.duration_s:.1f} s of audio · {timing}"
    )

    return (
        card,
        note,
        words_hl,
        state,
        None,  # player: nothing playing yet
        gr.update(visible=False),  # word panel
        tech,
        timeline_figure(result),
        ipa_hl,
        rows,
        posterior_heatmap(result.emissions, result.segments),
    )


# --- playback --------------------------------------------------------------------


def play_you(state):
    if not state:
        return None
    return SAMPLE_RATE, state["audio"]


def play_model(state, speed="normal"):
    if not state:
        return None
    return model_audio(state["text"], state["lang"], speed)


def play_slow(state):
    return play_model(state, "slow")


def _current(state):
    if state and state.get("current") is not None:
        return state["words"][state["current"]]
    return None


def play_word_you(state):
    w = _current(state)
    return clip(state["audio"], *w["span"]) if w else None


def play_word_model(state):
    w = _current(state)
    return model_audio(w["word"], state["lang"], "slow") if w else None


# --- taps ------------------------------------------------------------------------


def _index(evt: gr.SelectData) -> int | None:
    if evt.index is None:
        return None
    return evt.index if isinstance(evt.index, int) else evt.index[0]


def pick_word(evt: gr.SelectData, state):
    """Tap a word: open its panel and immediately play how it was said."""
    idx = _index(evt)
    nothing = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), state)
    if not state or idx is None or idx >= len(state["word_index"]) or state["word_index"][idx] is None:
        return nothing
    i = state["word_index"][idx]
    w = state["words"][i]
    state["current"] = i
    pills = []
    for label, band in w["pills"]:
        pills.append((label, band))
        pills.append((" ", None))
    title = f"### {w['word']} — {w['band']}"
    tips = [t for t in w["tips"] if t]
    body = "\n".join(f"- {t}" for t in tips) if tips else "This word sounded clear. Tap a sound to hear just that part."
    body += f"\n\n<span class='hint'>expected /{' '.join(w['expected'])}/ · heard /{' '.join(w['heard'])}/</span>"
    return gr.update(visible=True), title, pills, body, clip(state["audio"], *w["span"]), state


def pick_sound(evt: gr.SelectData, state):
    """Tap a sound pill: play just that sound (with a little context) and show its advice."""
    idx = _index(evt)
    w = _current(state)
    if not w or idx is None:
        return gr.update(), gr.update()
    j = idx // 2  # pills alternate with separators
    if j >= len(w["pills"]):
        return gr.update(), gr.update()
    span = w["phone_spans"][j]
    tip = w["tips"][j] or f"The {w['pills'][j][0]!r} sound in *{w['word']}* was clear."
    if span is None:
        return clip(state["audio"], *w["span"]), f"- {tip}"
    return clip(state["audio"], span[0], span[1], pad=0.04), f"- {tip}"


with gr.Blocks(title="Pronunciation Coach") as demo:
    gr.Markdown("# Pronunciation Coach")
    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="1. Record yourself")
        with gr.Column(scale=3):
            text = gr.Textbox(label="2. The sentence you are reading (leave empty to just talk)", lines=2)
            with gr.Row():
                accent = gr.Dropdown(list(ACCENTS), value=DEFAULT_ACCENT, label="Target accent", scale=1)
                l1 = gr.Dropdown(L1_OPTIONS, value=DEFAULT_L1, label="Your first language", scale=1)
                button = gr.Button("3. Check my pronunciation", variant="primary", scale=2)

    with gr.Row():
        with gr.Column(scale=2):
            card = gr.HTML()
            note = gr.Markdown()
        with gr.Column(scale=3):
            words_hl = gr.HighlightedText(
                label="Tap a word to hear it",
                color_map=BAND_COLOR,
                show_legend=True,
                show_inline_category=False,
                elem_id="sentence",
            )
            with gr.Row():
                btn_you = gr.Button("▶ You")
                btn_model = gr.Button("▶ Model")
                btn_slow = gr.Button("▶ Model, slowly")
            player = gr.Audio(label="Now playing", autoplay=True, interactive=False, elem_id="player")

    with gr.Group(visible=False) as word_panel:
        word_title = gr.Markdown()
        sounds_hl = gr.HighlightedText(
            label="Tap a sound to hear just that part",
            color_map=BAND_COLOR,
            show_legend=False,
            show_inline_category=False,
            elem_id="sounds",
        )
        with gr.Row():
            btn_word_you = gr.Button("▶ You said this word")
            btn_word_model = gr.Button("▶ Model says it slowly")
        word_tips = gr.Markdown()

    with gr.Accordion("Technical details (for teachers)", open=False):
        timeline = gr.Plot(label="Timeline: waveform, energy, spikes, word and sound crops")
        tech_text = gr.Textbox(label="What the recogniser saw", lines=5)
        ipa_hl = gr.HighlightedText(
            label=f"Expected phones, IPA (green ≥ {GOP_GOOD}, amber ≥ {GOP_UNSURE}, red below)",
            color_map=IPA_COLORS,
            show_legend=True,
        )
        table = gr.Dataframe(
            headers=["word", "phone", "spike (s)", "crop (s)", "GOP", "posterior", "heard", "top-3 candidates"],
            label="Per-phone detail",
            wrap=True,
        )
        plot = gr.Plot(label="Phone posteriors over time")

    state = gr.State()
    button.click(
        run,
        [audio, text, accent, l1],
        [card, note, words_hl, state, player, word_panel, tech_text, timeline, ipa_hl, table, plot],
    )
    words_hl.select(pick_word, [state], [word_panel, word_title, sounds_hl, word_tips, player, state])
    sounds_hl.select(pick_sound, [state], [player, word_tips])
    btn_you.click(play_you, [state], [player])
    btn_model.click(play_model, [state], [player])
    btn_slow.click(play_slow, [state], [player])
    btn_word_you.click(play_word_you, [state], [player])
    btn_word_model.click(play_word_model, [state], [player])

if __name__ == "__main__":
    log.info(
        "starting: phoneme model=%s asr=%s accent=%s debug=%s save_recordings=%s log=%s",
        DEFAULT_MODEL, DEFAULT_ASR, DEFAULT_ACCENT, DEBUG, SAVE_RECORDINGS, LOG_FILE,
    )
    demo.queue(default_concurrency_limit=1).launch(
        i18n=english_ui(),
        css=CSS,
        theme=gr.themes.Soft(),
        show_error=True,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        inbrowser=os.environ.get("PC_OPEN_BROWSER") == "1",  # the .bat launcher sets this
    )
