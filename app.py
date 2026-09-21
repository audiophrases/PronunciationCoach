"""Gradio front end.

Learner view: a score ring, the sentence as tappable word chips, one player that
speaks whatever was tapped (you, the model, one word) at the pace set by one speed
slider, and a word panel with one line of advice per sound that needs work.
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
from pronunciationcoach.feedback import BAND_COLOR, phone_tip, summary, word_feedback
from pronunciationcoach.g2p import ACCENTS, DEFAULT_ACCENT
from pronunciationcoach.phonetics import get_phonetics
from pronunciationcoach.logs import DEBUG, LOG_FILE, SAVE_RECORDINGS, log_assessment, setup_logging
from pronunciationcoach.pipeline import assess
from pronunciationcoach.scoring import GOP_GOOD, GOP_UNSURE
from pronunciationcoach.tts import synthesize
from pronunciationcoach.viz import posterior_heatmap, timeline_figure

log = setup_logging()
PHONETICS = get_phonetics()  # GAPhonetics: how to make each sound (None when unavailable)
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


def model_audio(text: str, lang: str, speed: float) -> str | None:
    try:
        return str(synthesize(text, lang, speed))
    except Exception:
        log.exception("could not synthesise %r", text)
        return None


def practise_pair(expected: str, heard: str) -> str:
    if PHONETICS is None:
        return ""
    g = PHONETICS.guidance(expected, heard)
    return g.practise if g else ""


def sound_guides(word) -> list[dict]:
    """For each sound of the word that needs work: label, instructions, clips, chart link."""
    guides, seen = [], set()
    for p in word.phones:
        if p.category == "good" or p.expected in seen or PHONETICS is None:
            continue
        g = PHONETICS.guidance(p.expected, None if p.dropped else p.heard)
        if g is None:
            continue
        seen.add(p.expected)
        t = g.target
        lines = [f"**/{t.ipa}/ as in *{t.example}***"]
        if g.contrast:
            lines.append(g.contrast)
        lines += [f"- {h}" for h in t.how]
        if t.mistake:
            lines.append(f"- *{t.mistake}*")
        if g.practise:
            lines.append(f"Practise the pair: **{g.practise}**")
        lines.append(f"<a href='{g.link}' target='_blank'>Open this contrast in the vowel & consonant charts ↗</a>")
        guides.append({
            "label": f"/{t.ipa}/ ({t.example})" + (" - not heard" if p.dropped else f", you said {p.heard}"),
            "md": "\n\n".join(lines),
            "sound": PHONETICS.audio_path(t.phoneme_audio),
            "word": PHONETICS.audio_path(t.word_audio),
        })
    return guides


# What a learner hears when a word is played back: the crop from the original recording, a
# little room on both sides but only through quiet audio (silence, a closure, breath - never
# a neighbouring word's vowel, which is what "it plays the previous word" was), short fades
# against clicks, and a level boost - class recordings are usually quiet.
PAD_BEFORE_S = 0.06
PAD_AFTER_S = 0.05
PAD_QUIET_DB = 18.0  # a pad frame must be this far below the word's own loudest 20 ms
FADE_S = 0.01
TARGET_PEAK = 0.7
MAX_GAIN = 8.0


def clip(audio: np.ndarray, start: float, end: float, sr: int = SAMPLE_RATE) -> tuple[int, np.ndarray]:
    a = max(0, int(start * sr))
    b = min(len(audio), int(end * sr))
    win = max(1, int(0.005 * sr))

    def rms(x) -> float:
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float32)))) if len(x) else 0.0

    core = np.asarray(audio[a:b], dtype=np.float32)
    n = len(core) // win
    if n >= 4:
        frames = np.sqrt(np.mean(np.square(core[: n * win]).reshape(n, win), axis=1))
        loud = float(np.max(np.convolve(frames, np.ones(4) / 4, mode="valid")))  # loudest 20 ms
    else:
        loud = rms(core)
    thr = loud * 10 ** (-PAD_QUIET_DB / 20)
    lim = max(0, a - int(PAD_BEFORE_S * sr))
    while a - win >= lim and rms(audio[a - win : a]) < thr:
        a -= win
    lim = min(len(audio), b + int(PAD_AFTER_S * sr))
    while b + win <= lim and rms(audio[b : b + win]) < thr:
        b += win
    out = np.array(audio[a:b], dtype=np.float32)
    n = min(int(FADE_S * sr), len(out) // 2)
    if n > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n, dtype=np.float32))
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if peak > 0:
        out *= min(TARGET_PEAK / peak, MAX_GAIN)
    return sr, out


# Playback speed: one slider for everything that can be played, so you and the model are
# always compared at the same pace. 1 = as spoken; ffmpeg's atempo takes 0.5-100 in one pass.
SPEED_MIN, SPEED_MAX, SPEED_STEP, SPEED_DEFAULT = 0.5, 1.5, 0.05, 1.0


def _speed(value) -> float:
    try:
        return round(min(max(float(value), SPEED_MIN), SPEED_MAX), 2)
    except (TypeError, ValueError):
        return SPEED_DEFAULT


def retimed(sr: int, samples: np.ndarray, speed: float):
    """The same audio at another pace with the pitch preserved (ffmpeg's atempo).
    Files are named by content and speed, so replaying is free and two listeners never
    share a file; at speed 1 the samples are returned as they are."""
    if abs(speed - 1.0) < 1e-6:
        return sr, samples
    import hashlib
    import subprocess
    import tempfile

    import soundfile as sf

    digest = hashlib.sha1(np.ascontiguousarray(samples, dtype=np.float32).tobytes()).hexdigest()[:16]
    dst = Path(tempfile.gettempdir()) / f"coach_clip_{digest}_{speed:.2f}.wav"
    if dst.exists():
        return str(dst)
    src = dst.with_name(f"coach_clip_{digest}.wav")
    if not src.exists():
        sf.write(src, samples, sr)
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-filter:a", f"atempo={speed}", str(dst)], check=True, capture_output=True)
        return str(dst)
    except Exception as exc:
        log.warning("could not change the clip's speed (%s); playing it as recorded", exc)
        return sr, samples


def score_card_html(words, summary_text: str) -> str:
    """Ring = words a listener caught; below it, the counts per verdict and the sound lists."""
    n = len(words)
    has_listener = any(w.listener_p is not None for w in words)
    understood = sum(1 for w in words if w.listener_p is None or w.understood)
    clear = sum(w.verdict == "clear" for w in words)
    ring_n, ring_label = (understood, "understood") if has_listener else (clear, "words clear")
    pct = 100.0 * ring_n / n if n else 0.0
    color = "#2e8b57" if pct >= 80 else ("#e0a800" if pct >= 50 else "#c0392b")
    counts = {v: sum(w.verdict == v for w in words) for v in ("clear", "accent", "almost", "work on this")}
    headline = "All clear - nice work!" if clear == n and n else " · ".join(
        f"<span style='color:{BAND_COLOR[v]}'>{c} {v}</span>" for v, c in counts.items() if c
    )
    sections = ""
    block, items = None, []
    for ln in summary_text.splitlines()[1:]:
        if ln.endswith(":") and not ln.startswith("•"):
            if block:
                sections += f"<div class='headline'>{escape(block)}</div><ul>{''.join(items)}</ul>"
            block, items = ln[:-1], []
        elif ln.startswith("•"):
            items.append(f"<li>{escape(ln[2:].strip()).replace('*', '')}</li>")
    if block:
        sections += f"<div class='headline'>{escape(block)}</div><ul>{''.join(items)}</ul>"
    ring = (
        "<svg class='ring' viewBox='0 0 36 36'>"
        "<path class='bg' d='M18 2.0845a15.9155 15.9155 0 0 1 0 31.831a15.9155 15.9155 0 0 1 0-31.831'/>"
        f"<path class='fg' stroke='{color}' stroke-dasharray='{pct:.0f}, 100' d='M18 2.0845a15.9155 15.9155 0 0 1 0 31.831a15.9155 15.9155 0 0 1 0-31.831'/>"
        f"<text x='18' y='19.5'>{ring_n}/{n}</text><text class='sub' x='18' y='24'>{ring_label}</text></svg>"
    )
    return f"<div class='score-card'>{ring}<div><div class='headline'>{headline}</div>{sections}</div></div>"


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

    summary_text = summary(result.words, practise=practise_pair)
    card = score_card_html(result.words, summary_text)
    note = ""
    if result.transcribed:
        note += f"I heard: **{result.text}**  "
    if result.extra_text():
        note += "<span class='hint'>Some repeated or hesitated parts were ignored.</span>"

    w_spans = result.word_spans()
    orig = np.asarray(samples, dtype=np.float32)
    if orig.ndim == 2:
        orig = orig.mean(axis=1)
    if np.issubdtype(np.asarray(samples).dtype, np.integer):
        orig = orig / np.iinfo(np.asarray(samples).dtype).max
    state = {
        "lang": lang,
        "text": result.text,
        "audio": audio16k,
        "orig": (int(sr), orig),  # the recording as it came in, for playback
        "word_index": word_index,
        "words": [
            {
                "word": w.word,
                "band": f.band,
                "span": (sp.start, sp.end),
                "listener": w.listener_p,
                "tips": f.tips,
                "guides": sound_guides(w),
                "expected": [p.expected for p in w.phones],
                "heard": [p.heard_label for p in w.phones],
            }
            for w, f, sp in zip(result.words, feedback, w_spans)
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
    for word, sp in zip(result.words, w_spans):
        for i, p in enumerate(word.phones):
            rows.append(
                [
                    word.word if i == 0 else "",
                    f"{sp.start:.2f}-{sp.end:.2f}" if i == 0 else "",
                    p.expected,
                    f"{p.start_s:.2f}",
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
        + f"Word crops: {result.span_source} · native reference: {', '.join(result.reference_voices) or 'none'}\n"
        + ("Sound guidance and clips: GAPhonetics (human US recordings; Wiktionary/Wikimedia Commons contributors, CC BY-SA 3.0 / CC0 - credits in its *-audio-sources.json)\n" if PHONETICS else "")
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


def play_you(state, speed):
    if not state:
        return None
    sr, orig = state.get("orig", (SAMPLE_RATE, state["audio"]))
    return retimed(sr, orig, _speed(speed))


def play_model(state, speed):
    if not state:
        return None
    return model_audio(state["text"], state["lang"], _speed(speed))


def _current(state):
    if state and state.get("current") is not None:
        return state["words"][state["current"]]
    return None


def _word_clip(state, w, speed):
    sr, orig = state.get("orig", (SAMPLE_RATE, state["audio"]))
    return retimed(*clip(orig, *w["span"], sr=sr), _speed(speed))


def play_word_you(state, speed):
    w = _current(state)
    return _word_clip(state, w, speed) if w else None


def play_word_model(state, speed):
    w = _current(state)
    return model_audio(w["word"], state["lang"], _speed(speed)) if w else None


# --- taps ------------------------------------------------------------------------


def _index(evt: gr.SelectData) -> int | None:
    if evt.index is None:
        return None
    return evt.index if isinstance(evt.index, int) else evt.index[0]


def pick_word(evt: gr.SelectData, state, speed):
    """Tap a word: open its panel and immediately play how it was said."""
    idx = _index(evt)
    nothing = (gr.update(), gr.update(), gr.update(), gr.update(), state, gr.update(), gr.update(), gr.update())
    if not state or idx is None or idx >= len(state["word_index"]) or state["word_index"][idx] is None:
        return nothing
    i = state["word_index"][idx]
    w = state["words"][i]
    state["current"] = i
    title = f"### {w['word']} — {w['band']}" + (f"  <span class='hint'>listener confidence {w['listener']:.0%}</span>" if w.get("listener") is not None else "")
    body = "\n".join(f"- {t}" for t in w["tips"]) if w["tips"] else "This word sounded clear."
    body += f"\n\n<span class='hint'>expected /{' '.join(w['expected'])}/ · heard /{' '.join(w['heard'])}/</span>"
    guides = w.get("guides") or []
    labels = [g["label"] for g in guides]
    return (
        gr.update(visible=True),
        title,
        body,
        _word_clip(state, w, speed),
        state,
        gr.update(visible=bool(guides)),
        gr.update(choices=labels, value=labels[0] if labels else None),
        guides[0]["md"] if guides else "",
    )


def _guide(state, label):
    w = _current(state)
    if not w or not label:
        return None
    return next((g for g in w.get("guides", []) if g["label"] == label), None)


def pick_sound(label, state):
    g = _guide(state, label)
    return g["md"] if g else ""


def play_guide_sound(label, state):
    g = _guide(state, label)
    return g["sound"] if g else None


def play_guide_word(label, state):
    g = _guide(state, label)
    return g["word"] if g else None


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
            speed = gr.Slider(
                SPEED_MIN, SPEED_MAX, value=SPEED_DEFAULT, step=SPEED_STEP,
                label="Playback speed (you and the model)",
                info="1 = as spoken. Around 0.7 is good for hearing the sounds in a word.",
            )
            with gr.Row():
                btn_you = gr.Button("▶ You")
                btn_model = gr.Button("▶ Model")
            player = gr.Audio(label="Now playing", autoplay=True, interactive=False, elem_id="player")

    with gr.Group(visible=False) as word_panel:
        word_title = gr.Markdown()
        with gr.Row():
            btn_word_you = gr.Button("▶ You said this word")
            btn_word_model = gr.Button("▶ Model says this word")
        word_tips = gr.Markdown()
        with gr.Group(visible=False) as guide_panel:
            gr.Markdown("#### How to make it")
            sound_pick = gr.Radio(choices=[], label="Sound to work on")
            with gr.Row():
                btn_guide_sound = gr.Button("▶ Hear the sound")
                btn_guide_word = gr.Button("▶ Hear it in a word")
            guide_md = gr.Markdown()

    with gr.Accordion("Technical details (for teachers)", open=False):
        timeline = gr.Plot(label="Timeline: waveform, energy, spikes, word and sound crops")
        tech_text = gr.Textbox(label="What the recogniser saw", lines=5)
        ipa_hl = gr.HighlightedText(
            label=f"Expected phones, IPA (green ≥ {GOP_GOOD}, amber ≥ {GOP_UNSURE}, red below)",
            color_map=IPA_COLORS,
            show_legend=True,
        )
        table = gr.Dataframe(
            headers=["word", "word crop (s)", "phone", "spike (s)", "GOP", "posterior", "heard", "top-3 candidates"],
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
    words_hl.select(pick_word, [state, speed], [word_panel, word_title, word_tips, player, state, guide_panel, sound_pick, guide_md])
    sound_pick.change(pick_sound, [sound_pick, state], [guide_md])
    btn_guide_sound.click(play_guide_sound, [sound_pick, state], [player])
    btn_guide_word.click(play_guide_word, [sound_pick, state], [player])
    btn_you.click(play_you, [state, speed], [player])
    btn_model.click(play_model, [state, speed], [player])
    btn_word_you.click(play_word_you, [state, speed], [player])
    btn_word_model.click(play_word_model, [state, speed], [player])

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
