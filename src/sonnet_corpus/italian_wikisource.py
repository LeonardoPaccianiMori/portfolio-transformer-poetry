"""Reproducible Italian Wikisource work-collection fetching helpers."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic, sleep

import requests
from bs4 import BeautifulSoup


ITALIAN_WIKISOURCE_API_URL = "https://it.wikisource.org/w/api.php"
USER_AGENT = "portfolio-transformer-poetry-corpus-builder/0.1"


@dataclass(frozen=True)
class WikisourcePageRevision:
    """One parsed Wikisource page and the revision used to render it."""

    title: str
    revision_id: int
    revision_timestamp: str


@dataclass(frozen=True)
class FetchedItalianWikisourceWork:
    """A root work page plus its ordered primary-text subpages."""

    landing_page_url: str
    title: str
    root_revision: WikisourcePageRevision
    page_revisions: list[WikisourcePageRevision]
    text: str
    raw_html_character_count: int


ProgressCallback = Callable[[str], None]


class WikisourceRateLimiter:
    """Enforce a minimum delay between MediaWiki API requests."""

    def __init__(self, request_delay: float = 1.0) -> None:
        if request_delay < 0:
            raise ValueError("request_delay must be greater than or equal to zero")
        self.request_delay = request_delay
        self._last_request_at = 0.0

    def wait(self) -> None:
        elapsed = monotonic() - self._last_request_at
        if elapsed < self.request_delay:
            sleep(self.request_delay - elapsed)
        self._last_request_at = monotonic()


def fetch_italian_wikisource_work(
    landing_page_url: str,
    *,
    expected_title: str,
    expected_first_subpage: str,
    expected_last_subpage: str,
    request_delay: float = 1.0,
    retries: int = 5,
    session: requests.Session | None = None,
    progress: ProgressCallback | None = None,
) -> FetchedItalianWikisourceWork:
    """Fetch an indexed work, pinning every included page to a revision.

    The root page supplies the ordered subpage list. Each subpage is then
    resolved to its current revision and rendered by that immutable revision.
    The returned text is suitable for inspection only; activation and corpus
    building remain separate decisions.
    """

    http = session or requests.Session()
    if session is None:
        http.headers.update({"User-Agent": USER_AGENT})
    if retries < 1:
        raise ValueError("retries must be at least one")
    limiter = WikisourceRateLimiter(request_delay)

    _write_progress(progress, f"resolving root page: {expected_title}")
    root_revision = _fetch_page_revision(
        expected_title,
        expected_title=expected_title,
        http=http,
        limiter=limiter,
        retries=retries,
        progress=progress,
    )
    root_html = _fetch_rendered_revision(
        root_revision,
        http=http,
        limiter=limiter,
        retries=retries,
        progress=progress,
    )
    subpage_titles = extract_ordered_subpage_titles(root_html, expected_title)
    validate_work_boundaries(
        subpage_titles,
        expected_first_subpage=expected_first_subpage,
        expected_last_subpage=expected_last_subpage,
    )

    _write_progress(progress, f"resolving revisions for {len(subpage_titles)} primary pages")
    page_revisions = _fetch_page_revisions(
        subpage_titles,
        http=http,
        limiter=limiter,
        retries=retries,
        progress=progress,
    )
    text_parts: list[str] = []
    raw_html_character_count = len(root_html)
    total_pages = len(subpage_titles)
    for index, page_revision in enumerate(page_revisions, start=1):
        _write_progress(
            progress,
            f"fetching page {index}/{total_pages}: {page_revision.title}",
        )
        page_html = _fetch_rendered_revision(
            page_revision,
            http=http,
            limiter=limiter,
            retries=retries,
            progress=progress,
        )
        page_text = extract_wikisource_prose_text(page_html)
        if page_text == "":
            raise ValueError(f"empty primary text after cleaning: {page_revision.title}")

        text_parts.append(f"## {page_revision.title}\n\n{page_text}")
        raw_html_character_count += len(page_html)

    return FetchedItalianWikisourceWork(
        landing_page_url=landing_page_url,
        title=expected_title,
        root_revision=root_revision,
        page_revisions=page_revisions,
        text="\n\n".join(text_parts).strip() + "\n",
        raw_html_character_count=raw_html_character_count,
    )


def extract_ordered_subpage_titles(root_html: str, root_title: str) -> list[str]:
    """Return unique primary-work subpages in the root page's visible order."""

    soup = BeautifulSoup(root_html, "html.parser")
    prefix = f"{root_title}/"
    titles: list[str] = []
    seen: set[str] = set()
    for link in soup.select(".mw-parser-output a[title]"):
        if link.find_parent(class_="ws-noexport") is not None:
            continue
        title = _normalize_title(link["title"])
        if not title.startswith(prefix) or title in seen:
            continue
        seen.add(title)
        titles.append(title)
    return titles


def validate_work_boundaries(
    subpage_titles: list[str],
    *,
    expected_first_subpage: str,
    expected_last_subpage: str,
) -> None:
    """Reject an index that does not match its recorded work boundaries."""

    if not subpage_titles:
        raise ValueError("Wikisource root page does not contain primary-text subpages")
    if subpage_titles[0] != expected_first_subpage:
        raise ValueError(
            "unexpected first Wikisource subpage: "
            f"expected {expected_first_subpage!r}, got {subpage_titles[0]!r}"
        )
    if subpage_titles[-1] != expected_last_subpage:
        raise ValueError(
            "unexpected last Wikisource subpage: "
            f"expected {expected_last_subpage!r}, got {subpage_titles[-1]!r}"
        )


def extract_wikisource_prose_text(html: str) -> str:
    """Remove common Wikisource wrappers and return readable primary text."""

    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(".mw-parser-output") or soup
    for unwanted in root.select(
        "style, script, table, .ws-noexport, .noprint, .metadata, "
        ".licenseContainer, .mw-editsection, .mw-references-wrap, .references, "
        ".ricerca-container, .quality-ns0, .quality-msg, #textquality, "
        "#catlinks, sup.reference"
    ):
        unwanted.decompose()

    lines = [_normalize_text_line(line) for line in root.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if line)


def _fetch_page_revision(
    requested_title: str,
    *,
    expected_title: str,
    http: requests.Session,
    limiter: WikisourceRateLimiter,
    retries: int,
    progress: ProgressCallback | None,
) -> WikisourcePageRevision:
    payload = _api_get(
        http,
        limiter,
        retries,
        progress,
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "revisions",
            "titles": requested_title,
            "rvprop": "ids|timestamp",
            "rvslots": "main",
            "rvlimit": "1",
        },
    )
    pages = payload.get("query", {}).get("pages", [])
    if len(pages) != 1 or pages[0].get("missing"):
        raise ValueError(f"Wikisource page is missing: {requested_title}")

    page = pages[0]
    actual_title = _normalize_title(page.get("title", ""))
    if actual_title != _normalize_title(expected_title):
        raise ValueError(
            "unexpected Wikisource page title: "
            f"expected {expected_title!r}, got {actual_title!r}"
        )
    revisions = page.get("revisions", [])
    if len(revisions) != 1:
        raise ValueError(f"Wikisource page has no current revision: {requested_title}")
    revision = revisions[0]
    return WikisourcePageRevision(
        title=actual_title,
        revision_id=int(revision["revid"]),
        revision_timestamp=str(revision["timestamp"]),
    )


def _fetch_page_revisions(
    titles: list[str],
    *,
    http: requests.Session,
    limiter: WikisourceRateLimiter,
    retries: int,
    progress: ProgressCallback | None,
) -> list[WikisourcePageRevision]:
    """Resolve up to fifty title revisions per API request and preserve order."""

    revisions_by_title: dict[str, WikisourcePageRevision] = {}
    for batch_index, title_batch in enumerate(_batches(titles, size=50), start=1):
        _write_progress(progress, f"resolving revision batch {batch_index}")
        payload = _api_get(
            http,
            limiter,
            retries,
            progress,
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "prop": "revisions",
                "titles": "|".join(title_batch),
                "rvprop": "ids|timestamp",
                "rvslots": "main",
            },
        )
        pages = payload.get("query", {}).get("pages", [])
        if len(pages) != len(title_batch):
            raise ValueError("Wikisource revision batch returned an unexpected page count")

        expected_titles = {_normalize_title(title) for title in title_batch}
        for page in pages:
            if page.get("missing"):
                raise ValueError(f"Wikisource page is missing: {page.get('title', '')}")
            actual_title = _normalize_title(page.get("title", ""))
            if actual_title not in expected_titles:
                raise ValueError(f"unexpected Wikisource page title: {actual_title!r}")
            revisions = page.get("revisions", [])
            if len(revisions) != 1:
                raise ValueError(f"Wikisource page has no current revision: {actual_title}")
            revision = revisions[0]
            revisions_by_title[actual_title] = WikisourcePageRevision(
                title=actual_title,
                revision_id=int(revision["revid"]),
                revision_timestamp=str(revision["timestamp"]),
            )

    return [revisions_by_title[_normalize_title(title)] for title in titles]


def _fetch_rendered_revision(
    revision: WikisourcePageRevision,
    *,
    http: requests.Session,
    limiter: WikisourceRateLimiter,
    retries: int,
    progress: ProgressCallback | None,
) -> str:
    payload = _api_get(
        http,
        limiter,
        retries,
        progress,
        {
            "action": "parse",
            "format": "json",
            "oldid": str(revision.revision_id),
            "prop": "text|revid",
            "disableeditsection": "1",
        },
    )
    parsed = payload.get("parse", {})
    parsed_revision_id = int(parsed.get("revid", -1))
    if parsed_revision_id != revision.revision_id:
        raise ValueError(
            "rendered revision does not match requested revision: "
            f"expected {revision.revision_id}, got {parsed_revision_id}"
        )
    text = parsed.get("text", {})
    html = text.get("*") if isinstance(text, dict) else ""
    if not isinstance(html, str) or html == "":
        raise ValueError(f"Wikisource revision has no rendered text: {revision.title}")
    return html


def _api_get(
    http: requests.Session,
    limiter: WikisourceRateLimiter,
    retries: int,
    progress: ProgressCallback | None,
    params: dict[str, str],
) -> dict[str, object]:
    response = None
    for attempt in range(retries):
        limiter.wait()
        response = http.get(ITALIAN_WIKISOURCE_API_URL, params=params, timeout=30)
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After", "")
        delay_seconds = float(retry_after) if retry_after.isdigit() else request_backoff_seconds(
            limiter.request_delay,
            attempt,
        )
        _write_progress(
            progress,
            f"rate limited; waiting {delay_seconds:g} seconds before retry {attempt + 1}/{retries}",
        )
        sleep(delay_seconds)
    if response is None:
        raise RuntimeError("Wikisource API request was not attempted")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Wikisource API returned a non-object payload")
    if "error" in payload:
        raise ValueError(f"Wikisource API error: {payload['error']}")
    return payload


def _normalize_title(title: str) -> str:
    return " ".join(title.replace("_", " ").split())


def request_backoff_seconds(request_delay: float, attempt: int) -> float:
    """Return a bounded increasing wait after a MediaWiki rate-limit response."""

    return max(request_delay, 1.0) * (attempt + 2)


def _batches(items: list[str], *, size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _normalize_text_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _write_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
