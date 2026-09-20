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
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
BRANCH = "gh-pages"
CLOUDFLARED = shutil.which("cloudflared") or r"C:\Program Files (x86)\cloudflared\cloudflared.exe"
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
STOP_FILE = ROOT / "tmp" / "share.stop"  # launchers\stop_sharing.bat creates it; Ctrl+C works too


def repo_slug() -> str:
    """owner/repo from the git remote, e.g. audiophrases/PronunciationCoach."""
    url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    if not m:
        raise SystemExit(f"origin is not a GitHub repository: {url}")
    return m.group(1)


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
    local = (ROOT / "share" / "index.html").read_bytes()
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
    print("Front door page updated on GitHub Pages.", flush=True)


def wait_for_port(port: int, timeout: float = 240.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1.0)
    return False


def main() -> None:
    if not Path(CLOUDFLARED).exists() and shutil.which("cloudflared") is None:
        raise SystemExit("cloudflared is not installed: winget install Cloudflare.cloudflared")
    slug = repo_slug()
    owner, repo = slug.split("/", 1)
    front_door = f"https://{owner}.github.io/{repo}/"

    env = {**os.environ, "GRADIO_SERVER_NAME": "127.0.0.1", "GRADIO_SERVER_PORT": str(PORT), "PC_OPEN_BROWSER": "0"}
    print("Starting the coach…", flush=True)
    app = subprocess.Popen([sys.executable, str(ROOT / "app.py")], cwd=ROOT, env=env)
    if not wait_for_port(PORT):
        app.terminate()
        raise SystemExit("The coach did not start (see logs/app.log).")

    print("Opening the tunnel…", flush=True)
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
        m = TUNNEL_RE.search(line)
        if m:
            url = m.group(0)
    if url is None:
        tunnel.terminate()
        app.terminate()
        raise SystemExit("cloudflared did not report a tunnel address (is the internet up?).")
    threading.Thread(target=lambda: [None for _ in tunnel.stdout], daemon=True).start()  # keep draining its output

    print(f"\nCoach address for this session: {url}", flush=True)
    try:
        sync_front_door(slug)
        publish(slug, url, True)
        print(f"Published. Students use the fixed address:  {front_door}", flush=True)
        print("(GitHub Pages takes about a minute to pick up a change; the address above works right away.)", flush=True)
    except Exception as exc:
        print(f"Could not publish to GitHub Pages ({exc}). Share the session address above directly.", flush=True)

    print("\nLeave this window open while students use the coach. Press Ctrl+C (or run stop_sharing.bat) to stop.\n", flush=True)
    STOP_FILE.parent.mkdir(exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)
    try:
        while app.poll() is None and tunnel.poll() is None and not STOP_FILE.exists():
            time.sleep(1.0)
        print("Stop requested." if STOP_FILE.exists() else "The coach or the tunnel stopped.", flush=True)
        STOP_FILE.unlink(missing_ok=True)
    except KeyboardInterrupt:
        print("\nStopping…", flush=True)
    finally:
        try:
            publish(slug, None, False)
            print("Front door marked closed.", flush=True)
        except Exception as exc:
            print(f"Could not mark the front door closed ({exc}).", flush=True)
        for proc in (tunnel, app):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()
