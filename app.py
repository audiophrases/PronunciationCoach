"""Gradio front end.

Learner view: the sentence with each word coloured, a plain-language summary,
"you vs model" playback for the whole sentence and for any word you tap.
Teacher view: everything technical (IPA, per-phone table, posterior heatmap)
in a collapsed section underneath.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from pronunciationcoach import SAMPLE_RATE
from pronunciationcoach.asr import DEFAULT_ASR
from pronunciationcoach.audio import to_mono_16k
from pronunciationcoach.engine import DEFAULT_MODEL
from pronunciationcoach.feedback import BAND_COLOR, summary, word_feedback
from pronunciationcoach.g2p import ACCENTS, DEFAULT_ACCENT
from pronunciationcoach.logs import DEBUG, LOG_FILE, SAVE_RECORDINGS, log_assessment, setup_logging
from pronunciationcoach.pipeline import assess
from pronunciationcoach.scoring import GOP_GOOD, GOP_UNSURE
from pronunciationcoach.tts import synthesize
from pronunciationcoach.viz import posterior_heatmap

log = setup_logging()
IPA_COLORS = {"good": "#2e8b57", "unsure": "#e0a800", "off": "#c0392b"}
L1_OPTIONS = ["Catalan", "Spanish", "Other / unknown"]  # plumbed through for the next stage; unused today
DEFAULT_L1 = os.environ.get("PC_L1", "Catalan")
if DEFAULT_L1 not in L1_OPTIONS:
    raise ValueError(f"PC_L1 must be one of {L1_OPTIONS}, got {DEFAULT_L1!r}")


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
    audio16k = to_mono_16k(samples, sr)
    log_assessment(log, result, l1, audio16k)

    # --- learner view ---------------------------------------------------------
    feedback = [word_feedback(w) for w in result.words]
    words_hl: list[tuple[str, str | None]] = []
    index_map: list[int | None] = []
    for i, f in enumerate(feedback):
        words_hl.append((f.word, f.band))
        index_map.append(i)
        words_hl.append((" ", None))
        index_map.append(None)

    text_md = summary(result.words)
    if result.transcribed:
        text_md = f"I heard: **{result.text}**\n\n" + text_md
    if result.extra_text():
        text_md += "\n\n_Some repeated or hesitated parts were ignored._"

    state = {
        "lang": lang,
        "audio": audio16k,
        "spans": result.word_spans(),
        "feedback": feedback,
        "index_map": index_map,
        "expected": [[p.expected for p in w.phones] for w in result.words],
        "heard": [[p.heard_label for p in w.phones] for w in result.words],
    }

    # --- teacher view ---------------------------------------------------------
    ipa_hl = []
    for word in result.words:
        for p in word.phones:
            ipa_hl.append((p.expected, p.category))
        ipa_hl.append(("  ", None))
    rows = [
        [
            word.word if i == 0 else "",
            p.expected,
            f"{p.start_s:.2f}",
            f"{p.gop:.2f}",
            f"{p.posterior:.2f}",
            p.heard_label,
            "  ".join(f"{ph} {pr:.2f}" for ph, pr in p.candidates),
        ]
        for word in result.words
        for i, p in enumerate(word.phones)
    ]
    timing = "  ".join(f"{k} {v:.1f}s" for k, v in result.timings.items())
    tech = (
        f"Reference{' (Whisper)' if result.transcribed else ''}: {result.text}\n"
        f"Heard, text-independent: {result.heard_text}\n"
        + (f"Heard but not in the sentence: {result.extra_text()}\n" if result.extra_text() else "")
        + (f"No model label for: {' '.join(result.unknown_phones)}\n" if result.unknown_phones else "")
        + f"{result.duration_s:.1f} s of audio · {timing}"
    )
    fig = posterior_heatmap(result.emissions, result.segments)

    return (
        words_hl,
        text_md,
        (SAMPLE_RATE, audio16k),
        model_audio(result.text, lang, "normal"),
        model_audio(result.text, lang, "slow"),
        state,
        gr.update(visible=False),  # word panel hidden until a word is tapped
        tech,
        ipa_hl,
        rows,
        fig,
    )


def pick_word(evt: gr.SelectData, state):
    """A tap on a word: show its advice and let the learner hear both versions."""
    nothing = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    if not state or evt.index is None:
        return nothing
    index = evt.index if isinstance(evt.index, int) else evt.index[0]
    i = state["index_map"][index] if index < len(state["index_map"]) else None
    if i is None:
        return nothing
    f = state["feedback"][i]
    lo, hi = state["spans"][i]
    you = state["audio"][int(lo * SAMPLE_RATE) : int(hi * SAMPLE_RATE)]
    title = f"### {f.word} — {f.band}"
    tips = "\n".join(f"- {t}" for t in f.tips) if f.tips else "This word sounded clear."
    tips += f"\n\n<small>expected /{' '.join(state['expected'][i])}/ · heard /{' '.join(state['heard'][i])}/</small>"
    return gr.update(visible=True), title, tips, (SAMPLE_RATE, you), model_audio(f.word, state["lang"], "slow")


with gr.Blocks(title="Pronunciation Coach") as demo:
    gr.Markdown(
        "# Pronunciation Coach\n"
        "Record yourself reading the sentence (or just speak, and leave the box empty). "
        "Then **tap any word** to hear how you said it and how it should sound."
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your recording")
            text = gr.Textbox(label="The sentence you are reading (leave empty for free speech)", lines=2)
            with gr.Row():
                accent = gr.Dropdown(list(ACCENTS), value=DEFAULT_ACCENT, label="Target accent")
                l1 = gr.Dropdown(L1_OPTIONS, value=DEFAULT_L1, label="Your first language")
            button = gr.Button("Check my pronunciation", variant="primary")
        with gr.Column(scale=2):
            words_hl = gr.HighlightedText(label="Your sentence — tap a word", color_map=BAND_COLOR, show_legend=True)
            summary_md = gr.Markdown()
            with gr.Row():
                you_audio = gr.Audio(label="You", interactive=False)
                model_normal = gr.Audio(label="Model", interactive=False)
                model_slow = gr.Audio(label="Model, slowly", interactive=False)
            with gr.Group(visible=False) as word_panel:
                word_title = gr.Markdown()
                word_tips = gr.Markdown()
                with gr.Row():
                    word_you = gr.Audio(label="You said", interactive=False)
                    word_model = gr.Audio(label="Model says", interactive=False)

    with gr.Accordion("Technical details (for teachers)", open=False):
        tech_text = gr.Textbox(label="What the recogniser saw", lines=5)
        ipa_hl = gr.HighlightedText(
            label=f"Expected phones, IPA (green ≥ {GOP_GOOD}, amber ≥ {GOP_UNSURE}, red below)",
            color_map=IPA_COLORS,
            show_legend=True,
        )
        table = gr.Dataframe(
            headers=["word", "phone", "start (s)", "GOP", "posterior", "heard", "top-3 candidates"],
            label="Per-phone detail",
            wrap=True,
        )
        plot = gr.Plot(label="Phone posteriors over time")

    state = gr.State()
    button.click(
        run,
        [audio, text, accent, l1],
        [words_hl, summary_md, you_audio, model_normal, model_slow, state, word_panel, tech_text, ipa_hl, table, plot],
    )
    words_hl.select(pick_word, [state], [word_panel, word_title, word_tips, word_you, word_model])

if __name__ == "__main__":
    log.info(
        "starting: phoneme model=%s asr=%s accent=%s debug=%s save_recordings=%s log=%s",
        DEFAULT_MODEL, DEFAULT_ASR, DEFAULT_ACCENT, DEBUG, SAVE_RECORDINGS, LOG_FILE,
    )
    demo.queue(default_concurrency_limit=1).launch(
        i18n=english_ui(),
        show_error=True,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        inbrowser=os.environ.get("PC_OPEN_BROWSER") == "1",  # the .bat launcher sets this
    )
