"""Optional headless-Chromium fallback for JavaScript-built pages (Playwright).

``LazyRenderer`` launches Chromium only the first time a thin page actually needs it, so the common
case pays no browser start-up cost. If Playwright or its browser is missing it quietly disables
itself and the agent works without it.
"""

from __future__ import annotations

import contextlib
import logging

log = logging.getLogger(__name__)

SETTLE_MS = 1500  # extra wait for late XHR content, after DOMContentLoaded


class PlaywrightRenderer:
    def __init__(self) -> None:
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(args=["--no-sandbox"])

    def render(self, url: str, timeout: float) -> str | None:
        page = None
        try:
            page = self._browser.new_page()
            page.goto(url, timeout=int(timeout * 1000), wait_until="domcontentloaded")
            # analytics / websockets may never go idle: keep whatever has loaded by then
            with contextlib.suppress(Exception):
                page.wait_for_load_state("networkidle", timeout=SETTLE_MS)
            return page.content()
        except Exception as exc:  # noqa: BLE001 - playwright has its own error hierarchy
            log.info("render failed for %s: %s", url, exc)
            return None
        finally:
            if page is not None:
                with contextlib.suppress(Exception):  # browser may already be gone
                    page.close()

    def close(self) -> None:
        with contextlib.suppress(Exception):  # shutting down; nothing useful to do
            self._browser.close()
            self._pw.stop()


class LazyRenderer:
    def __init__(self) -> None:
        self._inner: PlaywrightRenderer | None = None
        self._unavailable = False

    def render(self, url: str, timeout: float) -> str | None:
        if self._unavailable:
            return None
        if self._inner is None:
            try:
                self._inner = PlaywrightRenderer()
            except Exception as exc:  # noqa: BLE001 - browser is optional
                log.info("headless browser unavailable: %s", exc)
                self._unavailable = True
                return None
        return self._inner.render(url, timeout)

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()


def make_renderer() -> LazyRenderer:
    return LazyRenderer()
