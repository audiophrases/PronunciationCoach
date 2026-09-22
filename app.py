"""Gradio front end.

Learner view: a score ring, the sentence as tappable word chips (tap: you; tap again:
the model; again: you...), one player at the pace set by one speed slider, and a word
panel with one line of advice per sound that needs work.
Teacher view: the settings (target accent, female or male model voice) and everything
technical (timeline, IPA, per-phone table, posterior heatmap) in a collapsed section
underneath.
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
from pronunciationcoach.playback import FADE_S, MAX_GAIN, PAD_AFTER_S, PAD_BEFORE_S, PAD_QUIET_DB, TARGET_PEAK, clip
from pronunciationcoach.scoring import GOP_GOOD, GOP_UNSURE
from pronunciationcoach.tts import REFERENCE_VOICES, synthesize
from pronunciationcoach.viz import posterior_heatmap, timeline_figure

log = setup_logging()
PHONETICS = get_phonetics()  # GAPhonetics: how to make each sound (None when unavailable)
IPA_COLORS = {"good": "#2e8b57", "unsure": "#e0a800", "off": "#c0392b"}

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
#review-columns { align-items: flex-start; }
#word-column { position: sticky; top: 1rem; }
#word-panel { padding: 1rem; border: 1px solid var(--border-color-primary); border-radius: 1rem;
    background: var(--block-background-fill); }
#word-column:has(#word-panel) #word-placeholder { display: none; }
#sentence .token.highlighted:focus-visible { outline: 3px solid var(--color-accent); }
#sentence .token.coach-selected { outline: 3px solid var(--body-text-color); outline-offset: 2px; }
#review-toolbar { align-items: center; }
@media (max-width: 767px) {
    #review-columns { flex-direction: column; }
    #review-columns > .column { width: 100%; }
    #word-column { position: static; min-width: 0 !important; }
    #word-placeholder { display: none; }
    #review:has(#word-panel) { padding-bottom: 55dvh; }
    #word-panel { position: fixed; bottom: 0; left: 0; right: 0; z-index: 50;
        max-height: 55dvh; overflow-y: auto; overscroll-behavior: contain;
        border-radius: 1rem 1rem 0 0; box-shadow: 0 -8px 30px #0002;
        padding-bottom: max(1rem, env(safe-area-inset-bottom)); }
    #sentence .textfield { font-size: 1.2rem; line-height: 2.7; }
    .score-card .ring { width: 75px; height: 75px; }
}
"""

# Delegation survives Gradio replacing the sentence after another assessment.
REVIEW_JS = """() => {
    if (window.coachSelectionInstalled) return;
    window.coachSelectionInstalled = true;
    const selectWord = (event) => {
        if (event.type === 'keydown' && !['Enter', ' '].includes(event.key)) return;
        const token = event.target.closest('#sentence .token.highlighted');
        if (!token) return;
        document.querySelectorAll('#sentence .coach-selected').forEach(el => el.classList.remove('coach-selected'));
        token.classList.add('coach-selected');
    };
    document.addEventListener('click', selectWord);
    document.addEventListener('keydown', selectWord);
}"""


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


# The model voice: one natural female and one male voice per accent (the same two the
# scorer uses as native references), chosen in the teachers' section.
VOICE_CHOICES = ["Female", "Male"]
DEFAULT_VOICE = os.environ.get("PC_VOICE", "Male")
if DEFAULT_VOICE not in VOICE_CHOICES:
    raise ValueError(f"PC_VOICE must be one of {VOICE_CHOICES}, got {DEFAULT_VOICE!r}")


def model_voice(lang: str, choice: str) -> str:
    female, male = REFERENCE_VOICES.get(lang, REFERENCE_VOICES["en-us"])[:2]
    return male if choice == "Male" else female


def model_audio(text: str, lang: str, speed: float, voice: str = DEFAULT_VOICE) -> str | None:
    try:
        return str(synthesize(text, lang, speed, voice=model_voice(lang, voice)))
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


def run(audio, text, accent_name):
    if audio is None:
        log.warning("assess called without audio")
        raise gr.Error("Record or upload some audio first.")
    sr, samples = audio
    lang = ACCENTS[accent_name]
    log.debug("request: sr=%d samples=%s text=%r accent=%s", sr, getattr(samples, "shape", None), text, accent_name)
    try:
        result = assess(samples, sr, text, lang)
    except ValueError as exc:  # e.g. audio far too short for the sentence
        log.warning("assessment rejected: %s", exc)
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        log.exception("assessment failed")
        raise gr.Error(f"Something went wrong: {exc!r}. Details are in {LOG_FILE}") from exc
    audio16k = result.audio
    log_assessment(log, result, audio16k)

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
    chunk_ids = {i: cid for cid, chunk in enumerate(result.chunks) for i in chunk.members}
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
        "chunks": [{"text": c.text, "members": c.members,
                    "span": (c.span.start, c.span.end) if c.span else None,
                    "pad_before": c.pad_before, "pad_after": c.pad_after} for c in result.chunks],
        "words": [
            {
                "word": w.word,
                "band": f.band,
                "span": (sp.start, sp.end),
                "chunk": chunk_ids[i],
                "listener": w.listener_p,
                "tips": f.tips,
                "guides": sound_guides(w),
                "expected": [p.expected for p in w.phones],
                "heard": [p.heard_label if not p.inserted else f"+{p.heard}" for p in w.all_phones],
            }
            for i, (w, f, sp) in enumerate(zip(result.words, feedback, w_spans))
        ],
        "current": None,
        "turn": "you",  # what the next tap on the current word plays: "you" or "model"
    }

    # --- teacher view ---------------------------------------------------------
    ipa_hl = []
    for word in result.words:
        for p in word.phones:
            ipa_hl.append((p.expected, p.category))
        ipa_hl.append(("  ", None))
    rows = []
    for wi, (word, sp) in enumerate(zip(result.words, w_spans)):
        for i, p in enumerate(word.phones):
            rows.append(
                [
                    word.word if i == 0 else "",
                    result.chunks[chunk_ids[wi]].text if i == 0 else "",
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
        + "Playback: " + " | ".join(f"[{c.text}] ({c.reason})" for c in result.chunks) + "\n"
        + f"Crop recheck: {result.crop_recheck_mode}\n"
        + "".join(c.summary() + "\n" for c in result.crop_checks)
        + "".join("Playback verification: " + c.summary() + "\n" for c in result.transcript_checks)
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


def play_model(state, speed, voice):
    if not state:
        return None
    return model_audio(state["text"], state["lang"], _speed(speed), voice)


def _current(state):
    if state and state.get("current") is not None:
        return state["words"][state["current"]]
    return None


def _word_clip(state, w, speed):
    chunk = _chunk_for(state, w)
    if chunk["span"] is None or chunk["span"][1] <= chunk["span"][0]:
        return None
    sr, orig = state.get("orig", (SAMPLE_RATE, state["audio"]))
    return retimed(*clip(orig, *chunk["span"], sr=sr,
                         pad_before=chunk.get("pad_before", True),
                         pad_after=chunk.get("pad_after", True)), _speed(speed))


def _chunk_for(state, w):
    if "chunk" in w and state.get("chunks"):
        return state["chunks"][w["chunk"]]
    return {"text": w["word"], "span": w["span"], "members": [state.get("current")]}


def _word_title(state, turn):
    w = _current(state)
    chunk = _chunk_for(state, w)
    phrase = " ".join(
        f"<strong>{escape(state['words'][i]['word'])}</strong>" if i == state["current"] else escape(state["words"][i]["word"])
        for i in chunk["members"]
    )
    title = word_title(w, turn, phrase)
    if chunk["span"] is None:
        title += "\nThis part was not heard. You can still play the model."
    return title


def play_word_you(state, speed):
    w = _current(state)
    if not w:
        return None, state, gr.update()
    state["turn"] = "you"  # the next tap on the word plays the model
    return _word_clip(state, w, speed), state, _word_title(state, "you")


def play_word_model(state, speed, voice):
    w = _current(state)
    if not w:
        return None, state, gr.update()
    state["turn"] = "model"
    return model_audio(_chunk_for(state, w)["text"], state["lang"], _speed(speed), voice), state, _word_title(state, "model")


def word_title(w, turn: str, phrase: str | None = None) -> str:
    listener = f"  <span class='hint'>listener confidence {w['listener']:.0%}</span>" if w.get("listener") is not None else ""
    now = "▶ you" if turn == "you" else "▶ the model"
    nxt = "the model" if turn == "you" else "yourself"
    return f"### {phrase or escape(w['word'])}\n**{w['word']}** — {w['band']} · {now}{listener}\n<span class='hint'>Tap again to hear {nxt}.</span>"


# --- taps ------------------------------------------------------------------------


def _index(evt: gr.SelectData) -> int | None:
    if evt.index is None:
        return None
    return evt.index if isinstance(evt.index, int) else evt.index[0]


def pick_word(evt: gr.SelectData, state, speed, voice):
    """Select word-specific feedback, alternating learner/model audio within its group."""
    idx = _index(evt)
    nothing = (gr.update(), gr.update(), gr.update(), gr.update(), state, gr.update(), gr.update(), gr.update())
    if not state or idx is None or idx < 0 or idx >= len(state["word_index"]) or state["word_index"][idx] is None:
        return nothing
    i = state["word_index"][idx]
    w = state["words"][i]
    previous = _current(state)
    same_chunk = previous is not None and previous.get("chunk", state["current"]) == w.get("chunk", i)
    if same_chunk and state.get("turn") == "you":
        turn = "model"
    else:
        turn = "you"  # a new playback group starts with what you said
    state["current"], state["turn"] = i, turn
    body = "\n".join(f"- {t}" for t in w["tips"]) if w["tips"] else "This word sounded clear."
    body += f"\n\n<span class='hint'>expected /{' '.join(w['expected'])}/ · heard /{' '.join(w['heard'])}/</span>"
    guides = w.get("guides") or []
    labels = [g["label"] for g in guides]
    playing = _word_clip(state, w, speed) if turn == "you" else model_audio(_chunk_for(state, w)["text"], state["lang"], _speed(speed), voice)
    return (
        gr.update(visible=True),
        _word_title(state, turn),
        body,
        playing,
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
    with gr.Accordion("Recording & sentence", open=True, elem_id="setup") as setup:
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record yourself")
            with gr.Column(scale=3):
                text = gr.Textbox(label="Sentence you are reading (optional)",
                                  placeholder="Leave empty to just talk", lines=2)
                button = gr.Button("Check my pronunciation", variant="primary", size="md")

    with gr.Column(visible=False, elem_id="review") as review:
        with gr.Row(elem_id="review-toolbar"):
            gr.Markdown("## Your assessment")
            retry = gr.Button("Try again", size="sm", scale=0, min_width=100)
        card = gr.HTML()
        note = gr.Markdown()
        with gr.Row(elem_id="review-columns"):
            with gr.Column(scale=3, min_width=280):
                words_hl = gr.HighlightedText(
                    label="Your sentence",
                    color_map=BAND_COLOR,
                    show_legend=True,
                    show_inline_category=False,
                    elem_id="sentence",
                )
                gr.Markdown("Tap a word for feedback and your recording. Tap again to hear the model.")
                with gr.Row():
                    btn_you = gr.Button("▶ Whole recording", size="sm")
                    btn_model = gr.Button("▶ Whole model", size="sm")
                player = gr.Audio(label="Now playing", autoplay=True, interactive=False, elem_id="player")
                with gr.Accordion("Playback speed", open=False):
                    speed = gr.Slider(
                        SPEED_MIN, SPEED_MAX, value=SPEED_DEFAULT, step=SPEED_STEP,
                        label="Speed for you and the model",
                        info="1 = as spoken. Try 0.7 to hear individual sounds.",
                    )
            with gr.Column(scale=2, min_width=280, elem_id="word-column"):
                gr.Markdown("### Word feedback\nSelect a word to compare and practise it.", elem_id="word-placeholder")
                with gr.Column(visible=False, elem_id="word-panel", min_width=0) as word_panel:
                    with gr.Row():
                        gr.Markdown("### Word feedback")
                        close_word = gr.Button("Close word feedback", size="sm", scale=0, min_width=130)
                    word_head = gr.Markdown()
                    with gr.Row():
                        btn_word_you = gr.Button("▶ Hear yourself", size="sm")
                        btn_word_model = gr.Button("▶ Hear model", size="sm")
                    word_tips = gr.Markdown()
                    with gr.Accordion("How to make this sound", open=False, visible=False) as guide_panel:
                        sound_pick = gr.Radio(choices=[], label="Sound to work on")
                        with gr.Row():
                            btn_guide_sound = gr.Button("▶ Hear the sound", size="sm")
                            btn_guide_word = gr.Button("▶ Hear it in a word", size="sm")
                        guide_md = gr.Markdown()

    with gr.Accordion("Technical details (for teachers)", open=False):
        with gr.Row():
            accent = gr.Dropdown(
                list(ACCENTS), value=DEFAULT_ACCENT, label="Target accent",
                info="Reference pronunciation and model voice. Applies from the next check.",
            )
            voice = gr.Radio(VOICE_CHOICES, value=DEFAULT_VOICE, label="Model voice", info="Applies to the next thing played.")
        timeline = gr.Plot(label="Timeline: waveform, energy, spikes, word and sound crops")
        tech_text = gr.Textbox(label="What the recogniser saw", lines=5)
        ipa_hl = gr.HighlightedText(
            label=f"Expected phones, IPA (green ≥ {GOP_GOOD}, amber ≥ {GOP_UNSURE}, red below)",
            color_map=IPA_COLORS,
            show_legend=True,
        )
        table = gr.Dataframe(
            headers=["word", "playback", "word crop (s)", "phone", "spike (s)", "GOP", "posterior", "heard", "top-3 candidates"],
            label="Per-phone detail",
            wrap=True,
        )
        plot = gr.Plot(label="Phone posteriors over time")

    state = gr.State()
    check = button.click(
        run,
        [audio, text, accent],
        [card, note, words_hl, state, player, word_panel, tech_text, timeline, ipa_hl, table, plot],
    )
    check.success(
        lambda: (gr.update(open=False, label="Recording & sentence · Edit"), gr.update(visible=True)),
        outputs=[setup, review],
    )
    retry.click(
        lambda: (gr.update(open=True, label="Recording & sentence"), gr.update(visible=False), None),
        outputs=[setup, word_panel, player], queue=False,
    ).then(fn=None, js="() => document.getElementById('setup')?.scrollIntoView({block: 'start', behavior: 'smooth'})")
    close_word.click(lambda: (gr.update(visible=False), None), outputs=[word_panel, player], queue=False)
    demo.load(fn=None, js=REVIEW_JS)
    words_hl.select(pick_word, [state, speed, voice], [word_panel, word_head, word_tips, player, state, guide_panel, sound_pick, guide_md])
    sound_pick.change(pick_sound, [sound_pick, state], [guide_md])
    btn_guide_sound.click(play_guide_sound, [sound_pick, state], [player])
    btn_guide_word.click(play_guide_word, [sound_pick, state], [player])
    btn_you.click(play_you, [state, speed], [player])
    btn_model.click(play_model, [state, speed, voice], [player])
    btn_word_you.click(play_word_you, [state, speed], [player, state, word_head])
    btn_word_model.click(play_word_model, [state, speed, voice], [player, state, word_head])

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
