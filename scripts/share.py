"""Share the coach with students: run it here, expose it through a Cloudflare quick
tunnel, and publish the tunnel's address behind a fixed GitHub Pages front door.

    uv run python scripts/share.py            (launchers\\share.bat does this)

What happens:
  1. app.py starts on this machine (127.0.0.1:7860).
  2. `cloudflared tunnel --url http://127.0.0.1:7860` opens an outbound tunnel and prints
     a public https://<random>.trycloudflare.com address. No account needed.
  3. That address is written to coach.json on the repository's gh-pages branch (via the
     GitHub API, through the logged-in `gh`), next to share/index.html. The page at
     https://<owner>.github.io/<repo>/ reads it and forwards students to the coach.
  4. On Ctrl+C (or when the app stops) coach.json is marked closed.

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
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
BRANCH = "gh-pages"
CLOUDFLARED = shutil.which("cloudflared") or r"C:\Program Files (x86)\cloudflared\cloudflared.exe"
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
STOP_FILE = ROOT / "tmp" / "share.stop"  # launchers\stop_sharing.bat creates it; Ctrl+C works too
LOG_DIR = ROOT / "logs"
SHARE_LOG = LOG_DIR / "share.log"
APP_LOG = LOG_DIR / "app.log"

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


def publish(slug: str, url: str | None, open_: bool) -> None:
    """Write coach.json on the gh-pages branch (create or update)."""
    body = {"url": url, "open": open_, "opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
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


def drain(stream, name: str) -> None:
    """Copy a child's output into the session log (file only) so it survives the window."""
    for line in stream:
        line = line.rstrip()
        if line:
            log.debug("%s: %s", name, line)


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
    archiving = os.environ.get("PC_SAVE_RECORDINGS") == "1"
    log.info("=== Sharing session started on %s (coach %s, debug=%s, recordings archived=%s) ===",
             socket.gethostname(), app_version(), os.environ.get("PC_DEBUG") == "1", archiving)

    env = {**os.environ, "GRADIO_SERVER_NAME": "127.0.0.1", "GRADIO_SERVER_PORT": str(PORT), "PC_OPEN_BROWSER": "0"}
    log.info("Starting the coach…")
    app = subprocess.Popen([sys.executable, str(ROOT / "app.py")], cwd=ROOT, env=env)
    if not wait_for_port(PORT):
        app.terminate()
        log.error("The coach did not start (see logs/app.log).")
        raise SystemExit(1)
    log.info("The coach is up on port %d (pid %d).", PORT, app.pid)

    log.info("Opening the tunnel…")
    tunnel = subprocess.Popen(
        [CLOUDFLARED, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    url: str | None = None
    deadline = time.time() + 60
    assert tunnel.stdout is not None
    while time.time() < deadline and url is None:
        line = tunnel.stdout.readline()
        if not line:
            break
        log.debug("cloudflared: %s", line.rstrip())
        m = TUNNEL_RE.search(line)
        if m:
            url = m.group(0)
    if url is None:
        tunnel.terminate()
        app.terminate()
        log.error("cloudflared did not report a tunnel address (is the internet up?).")
        raise SystemExit(1)
    threading.Thread(target=drain, args=(tunnel.stdout, "cloudflared"), daemon=True).start()

    print()
    log.info("Coach address for this session: %s", url)
    try:
        sync_front_door(slug)
        publish(slug, url, True)
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
    try:
        while app.poll() is None and tunnel.poll() is None and not STOP_FILE.exists():
            time.sleep(1.0)
        if STOP_FILE.exists():
            log.info("Stop requested (stop_sharing.bat).")
        elif app.poll() is not None:
            log.warning("The coach stopped on its own (exit code %s) - see logs/app.log.", app.returncode)
        else:
            log.warning("The tunnel dropped (cloudflared exit code %s).", tunnel.returncode)
        STOP_FILE.unlink(missing_ok=True)
    except KeyboardInterrupt:
        print()
        log.info("Stopping (Ctrl+C)…")
    finally:
        try:
            publish(slug, None, False)
            log.info("Front door marked closed.")
        except Exception as exc:
            log.warning("Could not mark the front door closed (%s).", exc)
        for proc in (tunnel, app):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        minutes = int((datetime.now() - started).total_seconds() // 60)
        log.info("=== Session ended after %dh%02d: %s ===", minutes // 60, minutes % 60, session_summary(started))


if __name__ == "__main__":
    main()
