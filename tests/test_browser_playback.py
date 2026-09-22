"""Optional real-browser checks: requires Playwright and Edge (Windows) or Chromium."""
import base64
import io
from pathlib import Path
import sys
import wave

import numpy as np
import pytest


def test_native_player_replays_cached_audio_and_reports_blocked_playback():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, **({"channel": "msedge"} if sys.platform == "win32" else {}))
        except playwright.Error as exc:
            pytest.skip(f"Browser not installed: {exc}")
        try:
            page = browser.new_page()
            page.set_content('<audio id="coach-audio" controls></audio>'
                             '<p id="playback-message"></p><button id="replay">Play</button>')
            script = (Path(__file__).resolve().parents[1] / "pronunciationcoach/playback.js").read_text(encoding="utf-8")
            page.evaluate(f"() => {{ window.replay = ({script}); }}")
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes((3000 * np.sin(2*np.pi*440*np.arange(8000)/16000)).astype('<i2').tobytes())
            url = 'data:audio/wav;base64,' + base64.b64encode(buffer.getvalue()).decode()
            page.evaluate("url => { window.source = url; document.getElementById('replay').onclick = () => window.replay({url}); }", url)
            for _ in range(3):
                page.click('#replay')
                page.wait_for_function("document.getElementById('coach-audio').currentTime > 0.05")
                page.wait_for_function("document.getElementById('coach-audio').ended")
                assert page.locator('#playback-message').inner_text() == ''
            page.evaluate("() => { HTMLMediaElement.prototype.play = () => Promise.reject(new DOMException('blocked', 'NotAllowedError')); }")
            page.click('#replay')
            page.get_by_text('Audio could not start automatically.', exact=False).wait_for()
            page.evaluate("window.replay(null)")
            assert page.locator('#coach-audio').get_attribute('src') is None
            assert page.locator('#playback-message').inner_text() == ''
        finally:
            browser.close()
