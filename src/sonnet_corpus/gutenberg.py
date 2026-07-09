"""Project Gutenberg fetching and cleanup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic, sleep

import requests


GUTENBERG_USER_AGENT = "portfolio-transformer-poetry-gutenberg-probe/0.1"


@dataclass(frozen=True)
class FetchedGutenbergText:
    ebook_id: str
    url: str
    text: str


def candidate_plain_text_urls(ebook_id: str) -> list[str]:
    """Return likely UTF-8/plain-text URLs for a Project Gutenberg eBook."""

    return [
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-8.txt",
    ]


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the standard Project Gutenberg header and footer."""

    lines = text.splitlines()
    start_index = _find_marker_index(lines, marker="START", default=-1)
    end_index = _find_marker_index(lines, marker="END", default=len(lines))

    content_start = start_index + 1 if start_index >= 0 else 0
    content_end = end_index if end_index >= content_start else len(lines)
    body = lines[content_start:content_end]
    body = _strip_legacy_footer(body)

    return "\n".join(body).strip() + "\n"


def fetch_gutenberg_text(
    ebook_id: str,
    *,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> FetchedGutenbergText:
    """Fetch the first available plain-text file for a Project Gutenberg eBook."""

    http = session or requests.Session()
    http.headers.update({"User-Agent": GUTENBERG_USER_AGENT})
    errors: list[str] = []

    for url in candidate_plain_text_urls(ebook_id):
        response = http.get(url, timeout=timeout)
        if response.status_code == 404:
            errors.append(f"{url}: 404")
            continue
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            errors.append(f"{url}: {exc}")
            continue
        return FetchedGutenbergText(
            ebook_id=ebook_id,
            url=url,
            text=response.text,
        )

    raise FileNotFoundError(
        f"could not fetch Project Gutenberg plain text for {ebook_id}: "
        + "; ".join(errors)
    )


class GutenbergRateLimiter:
    """Small request-delay helper for polite sequential probes."""

    def __init__(self, request_delay: float) -> None:
        self.request_delay = request_delay
        self._last_request_at = 0.0

    def wait(self) -> None:
        elapsed = monotonic() - self._last_request_at
        if elapsed < self.request_delay:
            sleep(self.request_delay - elapsed)
        self._last_request_at = monotonic()


def _find_marker_index(lines: list[str], *, marker: str, default: int) -> int:
    marker_text = f" {marker} "
    for index, line in enumerate(lines):
        normalized = " ".join(line.upper().split())
        if (
            normalized.startswith("***")
            and marker_text in normalized
            and "PROJECT GUTENBERG" in normalized
        ):
            return index
    return default


def _strip_legacy_footer(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        normalized = " ".join(line.casefold().split())
        if normalized.startswith("end of project gutenberg"):
            return lines[:index]
    return lines
