"""Host side of the warm MFA aligner: one long-lived process, serialized requests.

Why a resident process rather than `mfa align_one` per assessment:
  * speed - the CLI costs ~16 s per call, almost all of it loading the acoustic
    model and lexicon; warm, an alignment takes ~25-95 ms with identical output;
  * safety - AcousticModel re-unpacks the 50 MB model into a shared directory on
    every construction, so two concurrent CLI runs corrupt each other (observed:
    one process left final.mdl missing). The worker unpacks once, privately.

The environment's interpreter is launched directly rather than through
`micromamba run`, which builds a micromamba -> cmd.exe -> python tree whose
python survives terminate() as an orphan. On Windows the process is also placed
in a job object that dies with the app, so closing the console cannot leak it.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

import numpy as np

from . import SAMPLE_RATE

log = logging.getLogger("pronunciationcoach")

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / ".cache/mfa-env"
MODELS = ROOT / ".cache/mfa-models"
SERVER = ROOT / "scripts/mfa_align_server.py"
WORK = MODELS / "pc-worker"  # private to the app; the CLI never writes here
ACOUSTIC_ZIP = MODELS / "pretrained_models/acoustic/english_mfa.zip"

START_TIMEOUT_S = 180.0  # the first run compiles the lexicon FSTs; later starts are ~5 s
REQUEST_TIMEOUT_S = float(os.environ.get("PC_MFA_TIMEOUT", "30"))


def available() -> bool:
    """True when the warm worker could be started without downloading anything."""
    return interpreter() is not None and SERVER.exists() and ACOUSTIC_ZIP.exists()


def interpreter() -> Path | None:
    for candidate in (ENVIRONMENT / "python.exe", ENVIRONMENT / "bin/python"):
        if candidate.exists():
            return candidate
    return None


def _environ(python: Path) -> dict:
    """The conda prefix must be on PATH or the DLLs (libsndfile) do not load."""
    prefix = python.parent
    conda_path = os.pathsep.join(str(prefix / part) for part in
                                 ("", "Library/mingw-w64/bin", "Library/usr/bin", "Library/bin", "Scripts", "bin"))
    return dict(os.environ,
                PATH=conda_path + os.pathsep + os.environ.get("PATH", ""),
                MFA_ROOT_DIR=str(MODELS),
                MAMBA_ROOT_PREFIX=str(ROOT / ".cache/mamba"),
                PC_MFA_ACOUSTIC_DIR=str(WORK / "english_mfa"),
                PC_MFA_SCRATCH=str(WORK / "scratch"),
                PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")


def _extract_acoustic_model() -> None:
    """Unpack the acoustic model once into a private directory.

    Handing MFA an already-extracted directory skips its rmtree-and-unpack step,
    which is what makes concurrent runs unsafe and costs most of a cold start.
    """
    target = WORK / "english_mfa"
    if (target / "final.mdl").exists():
        return
    # Stage under this process's own name: a second app instance starting at the same
    # moment must not be able to delete the directory our worker is already reading.
    staging = WORK / f"english_mfa.partial-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(ACOUSTIC_ZIP) as archive:
            archive.extractall(staging)
        try:
            (staging / "english_mfa").rename(target)  # only now is the directory complete
        except OSError:
            if not (target / "final.mdl").exists():
                raise  # someone else won the race and finished first, which is fine
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _confine_to_job(process: subprocess.Popen) -> None:
    """Tie the worker's lifetime to this process, even on an abrupt exit."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return

        class _Limits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.POINTER(ctypes.c_ulong)), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _Limits), ("IoInfo", ctypes.c_byte * 48),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        info = _Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        handle = kernel32.OpenProcess(0x0400 | 0x0800 | 0x1000, False, process.pid)  # QUERY|SET_QUOTA|TERMINATE
        if handle:
            kernel32.AssignProcessToJobObject(job, handle)
            kernel32.CloseHandle(handle)
        process._pc_job = job  # keep the handle alive for as long as the worker
    except Exception as exc:  # a missing job object only risks an orphan, never a wrong answer
        log.debug("MFA worker job object unavailable (%s)", exc)


class Worker:
    """One resident aligner. Requests are serialized; a dead worker is replaced."""

    def __init__(self) -> None:
        python = interpreter()
        if python is None:
            raise RuntimeError("MFA environment not installed; run launchers/setup.bat")
        _extract_acoustic_model()
        (WORK / "scratch").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._replies: queue.Queue = queue.Queue()
        self._log = open(WORK / "worker.log", "w", encoding="utf-8", errors="replace")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.process = subprocess.Popen(
            [str(python), "-u", str(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log, env=_environ(python), cwd=str(ROOT), text=True,
            encoding="utf-8", errors="replace", bufsize=1, creationflags=flags,
            start_new_session=(sys.platform != "win32"))
        _confine_to_job(self.process)
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

        started = time.perf_counter()
        hello = self._await(START_TIMEOUT_S)
        if not hello.get("ready"):
            self.close()
            raise RuntimeError(f"MFA worker failed to start: {hello.get('error', hello)}")
        self.startup_s = time.perf_counter() - started
        log.info("MFA worker ready in %.1f s (pid %s)", self.startup_s, hello.get("pid"))

    def _pump(self) -> None:
        try:
            for line in self.process.stdout:
                if line.strip():
                    self._replies.put(line)
        except Exception:
            pass
        finally:
            self._replies.put(None)  # the worker died; unblock whoever is waiting

    def _await(self, timeout: float) -> dict:
        try:
            line = self._replies.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"MFA worker did not answer within {timeout:.0f} s") from None
        if line is None:
            raise RuntimeError("MFA worker exited; see " + str(WORK / "worker.log"))
        return json.loads(line)

    def alive(self) -> bool:
        return self.process.poll() is None

    def request(self, payload: dict, timeout: float = REQUEST_TIMEOUT_S) -> dict:
        with self._lock:
            if not self.alive():
                raise RuntimeError("MFA worker is not running")
            try:
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
                return self._await(timeout)
            except (TimeoutError, RuntimeError, OSError, ValueError):
                self.close()  # a wedged worker must not be reused
                raise

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"cmd": "shutdown"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except Exception:
                pass
        for finish in (self.process.terminate, self.process.kill):
            if self.process.poll() is None:
                try:
                    finish()
                    self.process.wait(timeout=3)
                except Exception:
                    pass
        try:
            self._log.close()
        except Exception:
            pass


_worker: Worker | None = None
_start_lock = threading.Lock()


def get_worker() -> Worker:
    """The shared worker, started on first use and replaced if it has died."""
    global _worker
    with _start_lock:
        if _worker is None or not _worker.alive():
            _worker = Worker()
        return _worker


def shutdown() -> None:
    global _worker
    with _start_lock:
        if _worker is not None:
            _worker.close()
            _worker = None


atexit.register(shutdown)


def align(audio: np.ndarray, text: str, timeout: float = REQUEST_TIMEOUT_S) -> tuple[dict, float]:
    """Align one utterance. Returns MFA's raw tiers (as mfa.word_spans parses them)."""
    import soundfile as sf

    scratch = WORK / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    # A unique name per request: the app serves several students at once, and a shared
    # file would let one learner's audio be aligned against another's sentence.
    handle, path = tempfile.mkstemp(prefix="utt-", suffix=".wav", dir=scratch)
    os.close(handle)
    wav = Path(path)
    try:
        sf.write(wav, audio, SAMPLE_RATE, subtype="PCM_16")
        started = time.perf_counter()
        reply = get_worker().request({"wav": str(wav), "text": text}, timeout=timeout)
        if not reply.get("ok"):
            raise RuntimeError(f"MFA alignment failed: {reply.get('error', reply)}")
        return reply, time.perf_counter() - started
    finally:
        for leftover in (wav, wav.with_suffix(".lab"), wav.with_suffix(".json")):
            try:
                leftover.unlink()
            except OSError:
                pass
