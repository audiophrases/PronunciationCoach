"""Warm MFA alignment worker; runs inside .cache/mfa-env, not the app's venv.

`mfa align_one` costs ~16 s per call, almost all of it loading the acoustic model
and lexicon. This loads them once and then serves alignments in ~25-95 ms, with
output identical to the CLI (verified to 0.0000 ms on every archived recording).

It also removes a correctness hazard: AcousticModel unpacks the 50 MB model into a
shared directory on every construction (models/classes.py), so two concurrent `mfa`
processes corrupt each other - one observed leaving final.mdl missing. Pointing
PC_MFA_ACOUSTIC_DIR at an already-extracted directory skips the unpack entirely.

Protocol: newline-delimited JSON, one response object per request line.
  ready     {"ready": true, "pid": int, "setup_s": float}
  align     {"wav": "<path>", "text": "<transcript>"} ->
            {"ok": true, "duration": float, "tiers": {...}, "elapsed_s": float}
            {"ok": false, "error": str}
  control   {"cmd": "ping"} | {"cmd": "shutdown"}
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import traceback

SETUP_START = time.perf_counter()

# Kalpy and its Kaldi bindings print to stdout. Hand the protocol a private
# duplicate of fd 1 and point fd 1 at stderr so that chatter cannot corrupt it.
_RESPONSE = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
os.dup2(2, 1)


def emit(payload: dict) -> None:
    _RESPONSE.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _RESPONSE.flush()


import warnings  # noqa: E402

warnings.simplefilter("ignore")

import pywrapfst  # noqa: E402
from kalpy.aligner import KalpyAligner  # noqa: E402
from kalpy.feat.cmvn import CmvnComputer  # noqa: E402
from kalpy.fstext.lexicon import HierarchicalCtm, LexiconCompiler  # noqa: E402
from kalpy.utterance import Segment, Utterance as KalpyUtterance  # noqa: E402

from montreal_forced_aligner import config  # noqa: E402
from montreal_forced_aligner.alignment import PretrainedAligner  # noqa: E402
from montreal_forced_aligner.corpus.classes import FileData  # noqa: E402
from montreal_forced_aligner.data import (  # noqa: E402
    BRACKETED_WORD, CUTOFF_WORD, LAUGHTER_WORD, OOV_WORD, Language,
)
from montreal_forced_aligner.dictionary.mixins import (  # noqa: E402
    DEFAULT_BRACKETS, DEFAULT_CLITIC_MARKERS, DEFAULT_COMPOUND_MARKERS,
    DEFAULT_PUNCTUATION, DEFAULT_WORD_BREAK_MARKERS,
)
from montreal_forced_aligner.models import AcousticModel, DictionaryModel  # noqa: E402
from montreal_forced_aligner.online.alignment import tokenize_utterance_text  # noqa: E402
from montreal_forced_aligner.tokenization.simple import SimpleTokenizer  # noqa: E402
from montreal_forced_aligner.tokenization.spacy import generate_language_tokenizer  # noqa: E402

ACOUSTIC = os.environ.get("PC_MFA_ACOUSTIC", "english_mfa")
DICTIONARY = os.environ.get("PC_MFA_DICTIONARY", "english_us_mfa")


class Aligner:
    """Everything align_one does once, hoisted out of the per-request path."""

    def __init__(self) -> None:
        config.load_configuration()
        config.CLEAN, config.QUIET, config.NUM_JOBS = False, True, 1

        extracted = os.environ.get("PC_MFA_ACOUSTIC_DIR")
        if extracted and pathlib.Path(extracted).is_dir():
            self.acoustic = AcousticModel(pathlib.Path(extracted))  # no unpack, no shared state
        else:
            self.acoustic = AcousticModel(AcousticModel.get_pretrained_path(ACOUSTIC))
        dictionary = DictionaryModel(DictionaryModel.get_pretrained_path(DICTIONARY))

        c = PretrainedAligner.parse_parameters(None, None, None)
        p = self.acoustic.parameters
        self.lexicon = LexiconCompiler(
            disambiguation=False,
            silence_probability=p["silence_probability"],
            initial_silence_probability=p["initial_silence_probability"],
            final_silence_correction=p["final_silence_correction"],
            final_non_silence_correction=p["final_non_silence_correction"],
            silence_phone=p["optional_silence_phone"], oov_phone=p["oov_phone"],
            position_dependent_phones=p["position_dependent_phones"],
            phones=p["non_silence_phones"], ignore_case=c.get("ignore_case", True),
        )
        # The compiled lexicon FSTs are cached on disk by MFA itself; reuse them.
        directory = config.TEMPORARY_DIRECTORY / "extracted_models" / "dictionary" / dictionary.path.stem
        directory.mkdir(parents=True, exist_ok=True)
        l_fst, l_align = directory / "L.fst", directory / "L_align.fst"
        words, phones = directory / "words.txt", directory / "phones.txt"
        if l_fst.exists():
            self.lexicon.load_l_from_file(l_fst)
            self.lexicon.load_l_align_from_file(l_align)
            self.lexicon.word_table = pywrapfst.SymbolTable.read_text(words)
            self.lexicon.phone_table = pywrapfst.SymbolTable.read_text(phones)
        else:
            self.lexicon.load_pronunciations(dictionary.path)
            self.lexicon.create_fsts()
            self.lexicon.clear()
            self.lexicon.fst.write(str(l_fst))
            self.lexicon.align_fst.write(str(l_align))
            self.lexicon.word_table.write_text(words)
            self.lexicon.phone_table.write_text(phones)

        if self.acoustic.language is Language.unknown:
            self.tokenizer = SimpleTokenizer(
                word_table=self.lexicon.word_table,
                word_break_markers=c.get("word_break_markers", DEFAULT_WORD_BREAK_MARKERS),
                punctuation=c.get("punctuation", DEFAULT_PUNCTUATION),
                clitic_markers=c.get("clitic_markers", DEFAULT_CLITIC_MARKERS),
                compound_markers=c.get("compound_markers", DEFAULT_COMPOUND_MARKERS),
                brackets=c.get("brackets", DEFAULT_BRACKETS),
                laughter_word=c.get("laughter_word", LAUGHTER_WORD),
                oov_word=c.get("oov_word", OOV_WORD), cutoff_word=c.get("cutoff_word", CUTOFF_WORD),
                bracketed_word=c.get("bracketed_word", BRACKETED_WORD),
                ignore_case=c.get("ignore_case", True),
            )
        else:
            self.tokenizer = generate_language_tokenizer(self.acoustic.language)

        self.cmvn = CmvnComputer()
        options = {k: v for k, v in c.items() if k in (
            "beam", "retry_beam", "acoustic_scale", "transition_scale", "self_loop_scale", "boost_silence")}
        self.aligner = KalpyAligner(self.acoustic, self.lexicon, **options)

    def align(self, wav_path: str, text: str, scratch_dir: str) -> dict:
        wav = pathlib.Path(wav_path).resolve()
        scratch = pathlib.Path(scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
        lab = scratch / (wav.stem + ".lab")
        lab.write_text(text, encoding="utf-8")

        file = FileData.parse_file(wav.stem, wav, lab, "", 0)
        utterances = []
        for utterance in file.utterances:
            segment = Segment(wav, utterance.begin, utterance.end, utterance.channel)
            spoken = tokenize_utterance_text(utterance.text, self.lexicon, self.tokenizer, None)
            utt = KalpyUtterance(segment, spoken)
            utt.generate_mfccs(self.acoustic.mfcc_computer)
            utterances.append(utt)
        # align_one computes CMVN across the whole file before aligning; keep that.
        cmvn = self.cmvn.compute_cmvn_from_features([u.mfccs for u in utterances])
        ctm = HierarchicalCtm([])
        for utt in utterances:
            utt.apply_cmvn(cmvn)
            ctm.word_intervals.extend(self.aligner.align_utterance(utt).word_intervals)

        out = scratch / (wav.stem + ".json")
        ctm.export_textgrid(out, file_duration=file.wav_info.duration, output_format="json")
        return {"duration": file.wav_info.duration,
                "tiers": json.loads(out.read_text(encoding="utf-8"))["tiers"]}


def main() -> None:
    scratch = os.environ.get("PC_MFA_SCRATCH") or str(config.TEMPORARY_DIRECTORY / "worker")
    try:
        aligner = Aligner()
    except Exception as exc:
        emit({"ready": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        return
    emit({"ready": True, "pid": os.getpid(), "setup_s": time.perf_counter() - SETUP_START})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        started = time.perf_counter()
        try:
            request = json.loads(line)
        except Exception as exc:
            emit({"ok": False, "error": f"bad request: {exc}"})
            continue
        if request.get("cmd") == "shutdown":
            emit({"ok": True, "shutdown": True})
            break
        if request.get("cmd") == "ping":
            emit({"ok": True, "pong": True})
            continue
        try:
            result = aligner.align(request["wav"], request["text"], request.get("scratch", scratch))
            emit({"ok": True, "elapsed_s": time.perf_counter() - started, **result})
        except Exception as exc:  # a bad request must not end the session
            emit({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc(), "elapsed_s": time.perf_counter() - started})


if __name__ == "__main__":
    main()
