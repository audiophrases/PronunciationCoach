"""Share the coach with students: run it here, expose it through a Cloudflare quick
tunnel, and publish the tunnel's address behind a fixed GitHub Pages front door.

    uv run python scripts/share.py            (launchers\\share.bat does this)

What happens:
  1. app.py starts on this machine (127.0.0.1:7860).
  2. `cloudflared tunnel --url http://127.0.0.1:7860` opens an outbound tunnel and prints
     a public https://<random>.trycloudflare.com address. No account needed. The tunnel
     runs over HTTP/2 (TCP): the default QUIC transport needs outbound UDP 7844, which
     school and guest networks often block - and cloudflared then never connects.
  3. Once the tunnel has actually registered, that address is written to coach.json on the
     repository's gh-pages branch (via the GitHub API, through the logged-in `gh`), next to
     share/index.html. The page at https://<owner>.github.io/<repo>/ reads it and forwards
     students to the coach.
  4. While sharing: the tunnel is probed end to end every minute and re-opened (with a new
     address, republished) if it dies; coach.json is re-stamped every few minutes so the
     front door can tell a live session from a machine that went away; the machine is kept
     from sleeping on idle.
  5. On Ctrl+C, stop_sharing.bat, the window being closed, or the app stopping, coach.json
     is marked closed.

Every session's address is different; the front door is what students bookmark.

The session is logged to logs/share.log: machine, app version, addresses, when it opened
and closed, what cloudflared reported, and at the end how many assessments the app log
recorded meanwhile. The app itself keeps writing logs/app.log (and, with
PC_SAVE_RECORDINGS=1 as share.bat sets, archives every recording in recordings/).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
BRANCH = "gh-pages"
CLOUDFLARED = shutil.which("cloudflared") or r"C:\Program Files (x86)\cloudflared\cloudflared.exe"
TUNNEL_PROTOCOL = os.environ.get("PC_TUNNEL_PROTOCOL", "http2")  # "quic" needs outbound UDP 7844
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
REGISTERED = "Registered tunnel connection"
DEAD_MARKERS = ("Tunnel not found", "Register tunnel error")  # Cloudflare dropped the quick tunnel
STOP_FILE = ROOT / "tmp" / "share.stop"  # launchers\stop_sharing.bat creates it; Ctrl+C works too
APP_SCRIPT = Path(os.environ.get("PC_SHARE_APP", ROOT / "app.py"))  # a stand-in server, to test the sharing machinery without the models
LOG_DIR = ROOT / "logs"
SHARE_LOG = LOG_DIR / "share.log"
APP_LOG = LOG_DIR / "app.log"
PROBE_EVERY_S = 60  # end-to-end check of the tunnel
PROBE_FAILURES = 3  # consecutive failures before the tunnel is re-opened
HEARTBEAT_EVERY_S = 240  # re-stamp coach.json; the front door treats older than ~15 min as lost
RETRY_TUNNEL_S = 30  # when a tunnel cannot be opened (network down)

log = logging.getLogger("share")


def setup_logging() -> None:
    """Everything to logs/share.log (with timestamps, cloudflared's chatter included);
    the messages meant for the teacher on the console as well, as plain lines."""
    LOG_DIR.mkdir(exist_ok=True)
    to_file = RotatingFileHandler(SHARE_LOG, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    to_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
    to_console = logging.StreamHandler(sys.stdout)
    to_console.setLevel(logging.INFO)
    to_console.setFormatter(logging.Formatter("%(message)s"))
    log.handlers.clear()
    log.addHandler(to_file)
    log.addHandler(to_console)
    log.setLevel(logging.DEBUG)


def repo_slug() -> str:
    """owner/repo from the git remote, e.g. audiophrases/PronunciationCoach."""
    url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    if not m:
        raise SystemExit(f"origin is not a GitHub repository: {url}")
    return m.group(1)


def app_version() -> str:
    r = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.strip() or "unknown"


def gh(*args: str, input_json: dict | None = None) -> dict:
    cmd = ["gh", "api", *args]
    data = json.dumps(input_json).encode("utf-8") if input_json is not None else None
    if data is not None:
        cmd += ["--input", "-"]
    r = subprocess.run(cmd, input=data, capture_output=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace").strip() or r.stdout.decode("utf-8", "replace").strip())
    return json.loads(r.stdout or b"{}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def publish(slug: str, url: str | None, open_: bool, opened_at: str | None = None) -> None:
    """Write coach.json on the gh-pages branch (create or update). `seen` is the heartbeat."""
    body = {"url": url, "open": open_, "opened_at": opened_at or now_iso(), "seen": now_iso()}
    content = base64.b64encode((json.dumps(body, indent=1) + "\n").encode("utf-8")).decode("ascii")
    payload = {"message": f"coach {'open' if open_ else 'closed'}", "content": content, "branch": BRANCH}
    try:
        current = gh(f"repos/{slug}/contents/coach.json?ref={BRANCH}")
        payload["sha"] = current["sha"]
    except RuntimeError:
        pass  # first publish: no file yet
    gh("-X", "PUT", f"repos/{slug}/contents/coach.json", input_json=payload)


def sync_front_door(slug: str) -> None:
    """Keep share/index.html on the gh-pages branch identical to the copy in this repo."""
    local = (ROOT / "share" / "index.html").read_bytes().replace(b"\r\n", b"\n")  # same bytes from every machine
    blob_sha = hashlib.sha1(b"blob %d" % len(local) + bytes([0]) + local).hexdigest()  # what GitHub reports for a file
    payload = {"message": "front door page", "content": base64.b64encode(local).decode("ascii"), "branch": BRANCH}
    try:
        current = gh(f"repos/{slug}/contents/index.html?ref={BRANCH}")
        if current["sha"] == blob_sha:
            return
        payload["sha"] = current["sha"]
    except RuntimeError:
        pass
    gh("-X", "PUT", f"repos/{slug}/contents/index.html", input_json=payload)
    log.info("Front door page updated on GitHub Pages.")


def wait_for_port(port: int, timeout: float = 240.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1.0)
    return False


class Tunnel:
    """One cloudflared quick tunnel: its process, its address, and whether Cloudflare
    has since reported it dead (seen in its own output, which is drained to the log)."""

    def __init__(self, proc: subprocess.Popen, url: str) -> None:
        self.proc, self.url, self.dead = proc, url, False
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.rstrip()
            if line:
                log.debug("cloudflared: %s", line)
            if any(m in line for m in DEAD_MARKERS):
                self.dead = True

    def alive(self) -> bool:
        return self.proc.poll() is None and not self.dead

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def open_tunnel() -> Tunnel | None:
    """Start cloudflared and wait until it has both an address and a registered connection.
    An address alone means nothing: on a network that blocks the transport, cloudflared
    prints one and then never connects."""
    proc = subprocess.Popen(
        [CLOUDFLARED, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--protocol", TUNNEL_PROTOCOL, "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdout is not None
    url: str | None = None
    registered = False
    deadline = time.time() + 90
    while time.time() < deadline and not (url and registered):
        line = proc.stdout.readline()
        if not line:
            break
        line = line.rstrip()
        log.debug("cloudflared: %s", line)
        m = TUNNEL_RE.search(line)
        if m:
            url = m.group(0)
        if REGISTERED in line:
            registered = True
        if any(mk in line for mk in DEAD_MARKERS):
            break
    if url and registered:
        return Tunnel(proc, url)
    proc.terminate()
    log.warning("The tunnel did not connect (%s).", "no address" if not url else "address given but never registered - is the internet up?")
    return None


def tunnel_ok(url: str) -> bool:
    """Is the coach reachable through the tunnel, end to end?"""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "coach-share-probe"}), timeout=20) as r:
            return 200 <= r.status < 400
    except Exception as exc:
        log.debug("probe failed: %s", exc)
        return False


def keep_awake(on: bool) -> None:
    """Stop Windows from sleeping on idle while sharing (the display may still turn off).
    Closing the lid is a separate power setting."""
    if sys.platform != "win32":
        return
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def on_console_close(callback) -> None:
    """Run `callback` when the console window is closed (or the user logs off / shuts down):
    Windows allows a few seconds, enough to mark the front door closed."""
    if sys.platform != "win32":
        return
    import ctypes

    HANDLER = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

    def handler(event: int) -> bool:
        if event in (2, 5, 6):  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
            callback()
            return True
        return False

    on_console_close.keep = HANDLER(handler)  # type: ignore[attr-defined]  # must outlive the call
    ctypes.windll.kernel32.SetConsoleCtrlHandler(on_console_close.keep, True)  # type: ignore[attr-defined]


def session_summary(since: datetime) -> str:
    """What the app logged while the session was open: assessments and archived recordings."""
    assessed = archived = 0
    try:
        lines = APP_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "no app log found"
    for line in lines:
        try:
            when = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if when < since:
            continue
        if " assess mode=" in line:
            assessed += 1
        elif "recording archived:" in line:
            archived += 1
    return f"{assessed} assessment(s), {archived} recording(s) archived"


def main() -> None:
    setup_logging()
    if not Path(CLOUDFLARED).exists() and shutil.which("cloudflared") is None:
        raise SystemExit("cloudflared is not installed: winget install Cloudflare.cloudflared")
    slug = repo_slug()
    owner, repo = slug.split("/", 1)
    front_door = f"https://{owner}.github.io/{repo}/"
    started = datetime.now()
    opened_at = now_iso()
    archiving = os.environ.get("PC_SAVE_RECORDINGS") == "1"
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        local_ip = "?"
    log.info("=== Sharing session started on %s (%s; coach %s, debug=%s, recordings archived=%s, tunnel %s) ===",
             socket.gethostname(), local_ip, app_version(), os.environ.get("PC_DEBUG") == "1", archiving, TUNNEL_PROTOCOL)

    env = {**os.environ, "GRADIO_SERVER_NAME": "127.0.0.1", "GRADIO_SERVER_PORT": str(PORT), "PC_OPEN_BROWSER": "0"}
    log.info("Starting the coach…")
    app = subprocess.Popen([sys.executable, str(APP_SCRIPT)], cwd=ROOT, env=env)
    if not wait_for_port(PORT):
        app.terminate()
        log.error("The coach did not start (see logs/app.log).")
        raise SystemExit(1)
    log.info("The coach is up on port %d (pid %d).", PORT, app.pid)

    log.info("Opening the tunnel…")
    tunnel = open_tunnel()
    if tunnel is None:
        app.terminate()
        log.error("Could not open a tunnel. Check the internet connection and run share.bat again.")
        raise SystemExit(1)

    print()
    log.info("Coach address for this session: %s", tunnel.url)
    published = False
    try:
        sync_front_door(slug)
        publish(slug, tunnel.url, True, opened_at)
        published = True
        log.info("Published. Students use the fixed address:  %s", front_door)
        log.info("(GitHub Pages takes about a minute to pick up a change; the address above works right away.)")
    except Exception as exc:
        log.warning("Could not publish to GitHub Pages (%s). Share the session address above directly.", exc)

    print()
    log.info("Leave this window open while students use the coach. Press Ctrl+C (or run stop_sharing.bat) to stop.")
    log.info("Session log: %s   app log: %s%s", SHARE_LOG.relative_to(ROOT), APP_LOG.relative_to(ROOT),
             "   recordings: recordings\\" if archiving else "")
    print()
    STOP_FILE.parent.mkdir(exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)
    keep_awake(True)

    closing = threading.Event()

    def close_front_door() -> None:
        if closing.is_set():
            return
        closing.set()
        try:
            publish(slug, None, False, opened_at)
            log.info("Front door marked closed.")
        except Exception as exc:
            log.warning("Could not mark the front door closed (%s).", exc)

    on_console_close(lambda: (log.info("Window closed."), close_front_door(), tunnel.close()))

    restarts = 0
    failures = 0
    checked = False  # first successful end-to-end probe is worth telling the teacher
    next_probe = time.time() + PROBE_EVERY_S
    next_beat = time.time() + HEARTBEAT_EVERY_S
    try:
        while app.poll() is None and not STOP_FILE.exists():
            time.sleep(1.0)
            now = time.time()
            if tunnel.alive() and now >= next_probe:
                next_probe = now + PROBE_EVERY_S
                failures = 0 if tunnel_ok(tunnel.url) else failures + 1
                if failures:
                    log.debug("tunnel probe failed (%d/%d)", failures, PROBE_FAILURES)
                elif not checked:
                    checked = True
                    log.info("Checked: the coach answers through the tunnel.")
            if not tunnel.alive() or failures >= PROBE_FAILURES:
                log.warning("The tunnel %s. Opening a new one…", "was dropped by Cloudflare" if tunnel.dead else
                            "stopped" if tunnel.proc.poll() is not None else "is not reachable")
                tunnel.close()
                while app.poll() is None and not STOP_FILE.exists():
                    fresh = open_tunnel()
                    if fresh is not None:
                        break
                    log.info("Retrying in %d s…", RETRY_TUNNEL_S)
                    time.sleep(RETRY_TUNNEL_S)
                else:
                    break
                tunnel, failures, restarts, checked = fresh, 0, restarts + 1, False
                next_probe = time.time() + PROBE_EVERY_S
                log.info("New coach address: %s", tunnel.url)
                try:
                    publish(slug, tunnel.url, True, opened_at)
                    published = True
                    log.info("Republished; the fixed address follows within a minute.")
                except Exception as exc:
                    log.warning("Could not republish (%s).", exc)
            elif published and now >= next_beat:
                next_beat = now + HEARTBEAT_EVERY_S
                try:
                    publish(slug, tunnel.url, True, opened_at)
                    log.debug("heartbeat published")
                except Exception as exc:
                    log.debug("heartbeat failed: %s", exc)
        if STOP_FILE.exists():
            log.info("Stop requested (stop_sharing.bat).")
        elif app.poll() is not None:
            log.warning("The coach stopped on its own (exit code %s) - see logs/app.log.", app.returncode)
        STOP_FILE.unlink(missing_ok=True)
    except KeyboardInterrupt:
        print()
        log.info("Stopping (Ctrl+C)…")
    finally:
        keep_awake(False)
        close_front_door()
        tunnel.close()
        if app.poll() is None:
            app.terminate()
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
        minutes = int((datetime.now() - started).total_seconds() // 60)
        log.info("=== Session ended after %dh%02d: %s, tunnel re-opened %d time(s) ===",
                 minutes // 60, minutes % 60, session_summary(started), restarts)


if __name__ == "__main__":
    main()
