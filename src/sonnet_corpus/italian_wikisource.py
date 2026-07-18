"""Reproducible Italian Wikisource work-collection fetching helpers."""

from __future__ import annotations

import re
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class FetchedItalianWikisourcePage:
    """One revision-pinned page and its rendered HTML, retained only in memory."""

    revision: WikisourcePageRevision
    html: str


@dataclass(frozen=True)
class FetchedItalianWikisourcePageCollection:
    """A root collection page and its ordered rendered primary-text pages."""

    landing_page_url: str
    title: str
    root_revision: WikisourcePageRevision
    root_html: str
    pages: list[FetchedItalianWikisourcePage]


@dataclass(frozen=True)
class WikisourceWorkSnapshot:
    """Committed revision list for one audited Wikisource work."""

    source_id: str
    landing_page_url: str
    title: str
    scope: str
    root_revision: WikisourcePageRevision
    page_revisions: list[WikisourcePageRevision]


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
    selected_subpage_titles: list[str] | None = None,
    excluded_subpage_prefixes: tuple[str, ...] = (),
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

    collection = fetch_italian_wikisource_page_collection(
        landing_page_url,
        expected_title=expected_title,
        expected_first_subpage=expected_first_subpage,
        expected_last_subpage=expected_last_subpage,
        selected_subpage_titles=selected_subpage_titles,
        excluded_subpage_prefixes=excluded_subpage_prefixes,
        request_delay=request_delay,
        retries=retries,
        session=session,
        progress=progress,
    )
    text_parts: list[str] = []
    raw_html_character_count = len(collection.root_html)
    for page in collection.pages:
        page_text = extract_wikisource_prose_text(page.html)
        if page_text == "":
            raise ValueError(f"empty primary text after cleaning: {page.revision.title}")

        text_parts.append(f"## {page.revision.title}\n\n{page_text}")
        raw_html_character_count += len(page.html)

    return FetchedItalianWikisourceWork(
        landing_page_url=landing_page_url,
        title=expected_title,
        root_revision=collection.root_revision,
        page_revisions=[page.revision for page in collection.pages],
        text="\n\n".join(text_parts).strip() + "\n",
        raw_html_character_count=raw_html_character_count,
    )


def fetch_italian_wikisource_page_collection(
    landing_page_url: str,
    *,
    expected_title: str,
    expected_first_subpage: str = "",
    expected_last_subpage: str = "",
    selected_subpage_titles: list[str] | None = None,
    explicit_page_titles: list[str] | None = None,
    excluded_subpage_prefixes: tuple[str, ...] = (),
    request_delay: float = 1.0,
    retries: int = 5,
    session: requests.Session | None = None,
    progress: ProgressCallback | None = None,
) -> FetchedItalianWikisourcePageCollection:
    """Fetch an indexed collection as individually revision-pinned pages.

    This is deliberately lower level than :func:`fetch_italian_wikisource_work`.
    It keeps each rendered page separate in memory so callers can apply a
    source-appropriate extraction rule, such as poem line counting. It never
    writes raw HTML to disk.
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
    if explicit_page_titles is not None:
        selected_titles = select_explicit_page_titles(root_html, explicit_page_titles)
    else:
        selected_titles = select_work_subpage_titles(
            extract_ordered_subpage_titles(root_html, expected_title),
            selected_subpage_titles,
            excluded_subpage_prefixes,
        )
        validate_work_boundaries(
            selected_titles,
            expected_first_subpage=expected_first_subpage,
            expected_last_subpage=expected_last_subpage,
        )

    _write_progress(progress, f"resolving revisions for {len(selected_titles)} primary pages")
    page_revisions = _fetch_page_revisions(
        selected_titles,
        http=http,
        limiter=limiter,
        retries=retries,
        progress=progress,
    )
    pages: list[FetchedItalianWikisourcePage] = []
    total_pages = len(page_revisions)
    for index, page_revision in enumerate(page_revisions, start=1):
        _write_progress(
            progress,
            f"fetching page {index}/{total_pages}: {page_revision.title}",
        )
        pages.append(
            FetchedItalianWikisourcePage(
                revision=page_revision,
                html=_fetch_rendered_revision(
                    page_revision,
                    http=http,
                    limiter=limiter,
                    retries=retries,
                    progress=progress,
                ),
            )
        )

    return FetchedItalianWikisourcePageCollection(
        landing_page_url=landing_page_url,
        title=expected_title,
        root_revision=root_revision,
        root_html=root_html,
        pages=pages,
    )


def read_wikisource_work_snapshot(path: Path) -> WikisourceWorkSnapshot:
    """Read and validate a committed revision snapshot."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    root = payload["root_revision"]
    pages = payload["page_revisions"]
    snapshot = WikisourceWorkSnapshot(
        source_id=str(payload["source_id"]),
        landing_page_url=str(payload["landing_page_url"]),
        title=str(payload["title"]),
        scope=str(payload.get("scope", "all_root_subpages")),
        root_revision=WikisourcePageRevision(**root),
        page_revisions=[WikisourcePageRevision(**page) for page in pages],
    )
    if not snapshot.page_revisions:
        raise ValueError(f"Wikisource snapshot has no pages: {snapshot.source_id}")
    if snapshot.scope not in {"all_root_subpages", "explicit_subpages"}:
        raise ValueError(f"invalid Wikisource snapshot scope: {snapshot.scope}")
    return snapshot


def fetch_pinned_italian_wikisource_work(
    snapshot: WikisourceWorkSnapshot,
    *,
    request_delay: float = 6.0,
    session: requests.Session | None = None,
    progress: ProgressCallback | None = None,
) -> FetchedItalianWikisourceWork:
    """Render exactly the revisions declared by a committed work snapshot."""

    collection = fetch_pinned_italian_wikisource_page_collection(
        snapshot,
        request_delay=request_delay,
        session=session,
        progress=progress,
    )
    parts: list[str] = []
    raw_html_character_count = len(collection.root_html)
    for page in collection.pages:
        page_text = extract_wikisource_prose_text(page.html)
        if not page_text:
            raise ValueError(f"empty primary text after cleaning: {page.revision.title}")
        parts.append(f"## {page.revision.title}\n\n{page_text}")
        raw_html_character_count += len(page.html)

    return FetchedItalianWikisourceWork(
        landing_page_url=snapshot.landing_page_url,
        title=snapshot.title,
        root_revision=snapshot.root_revision,
        page_revisions=snapshot.page_revisions,
        text="\n\n".join(parts).strip() + "\n",
        raw_html_character_count=raw_html_character_count,
    )


def fetch_pinned_italian_wikisource_page_collection(
    snapshot: WikisourceWorkSnapshot,
    *,
    request_delay: float = 6.0,
    session: requests.Session | None = None,
    progress: ProgressCallback | None = None,
) -> FetchedItalianWikisourcePageCollection:
    """Render a committed page snapshot while retaining each page separately."""

    http = session or requests.Session()
    if session is None:
        http.headers.update({"User-Agent": USER_AGENT})
    limiter = WikisourceRateLimiter(request_delay)
    _write_progress(progress, f"rendering pinned root revision: {snapshot.title}")
    root_html = _fetch_rendered_revision(
        snapshot.root_revision,
        http=http,
        limiter=limiter,
        retries=5,
        progress=progress,
    )
    actual_titles = extract_ordered_subpage_titles(root_html, snapshot.title)
    expected_titles = [revision.title for revision in snapshot.page_revisions]
    if snapshot.scope == "all_root_subpages" and actual_titles != expected_titles:
        raise ValueError("Wikisource root page no longer matches committed page snapshot")
    if snapshot.scope == "explicit_subpages":
        missing_titles = [title for title in expected_titles if title not in actual_titles]
        if missing_titles:
            raise ValueError(f"Wikisource root page is missing committed subpages: {missing_titles}")

    pages: list[FetchedItalianWikisourcePage] = []
    for index, revision in enumerate(snapshot.page_revisions, start=1):
        _write_progress(
            progress,
            f"rendering pinned page {index}/{len(snapshot.page_revisions)}: {revision.title}",
        )
        page_html = _fetch_rendered_revision(
            revision,
            http=http,
            limiter=limiter,
            retries=5,
            progress=progress,
        )
        pages.append(FetchedItalianWikisourcePage(revision=revision, html=page_html))

    return FetchedItalianWikisourcePageCollection(
        landing_page_url=snapshot.landing_page_url,
        title=snapshot.title,
        root_revision=snapshot.root_revision,
        root_html=root_html,
        pages=pages,
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


def select_explicit_page_titles(root_html: str, explicit_page_titles: list[str]) -> list[str]:
    """Validate an ordered source-approved page list linked from a root index."""

    if not explicit_page_titles:
        raise ValueError("explicit Wikisource page selection is empty")
    soup = BeautifulSoup(root_html, "html.parser")
    root = soup.select_one(".mw-parser-output") or soup
    linked_titles = {
        _normalize_title(link["title"])
        for link in root.select("a[title]")
        if link.find_parent(class_="ws-noexport") is None
    }
    normalized_titles = [_normalize_title(title) for title in explicit_page_titles]
    if len(set(normalized_titles)) != len(normalized_titles):
        raise ValueError("explicit Wikisource page selection contains duplicates")
    missing_titles = [title for title in normalized_titles if title not in linked_titles]
    if missing_titles:
        raise ValueError(f"Wikisource root page is missing explicit pages: {missing_titles}")
    return normalized_titles


def validate_work_boundaries(
    subpage_titles: list[str],
    *,
    expected_first_subpage: str,
    expected_last_subpage: str = "",
) -> None:
    """Reject an index that does not match its recorded stable boundaries.

    A blank final boundary is intentional for an audited recursive work whose
    complete primary-text tree is selected through explicit exclusion rules.
    """

    if not subpage_titles:
        raise ValueError("Wikisource root page does not contain primary-text subpages")
    if expected_first_subpage and subpage_titles[0] != expected_first_subpage:
        raise ValueError(
            "unexpected first Wikisource subpage: "
            f"expected {expected_first_subpage!r}, got {subpage_titles[0]!r}"
        )
    if expected_last_subpage and subpage_titles[-1] != expected_last_subpage:
        raise ValueError(
            "unexpected last Wikisource subpage: "
            f"expected {expected_last_subpage!r}, got {subpage_titles[-1]!r}"
        )


def select_work_subpage_titles(
    discovered_titles: list[str],
    selected_titles: list[str] | None,
    excluded_prefixes: tuple[str, ...] = (),
) -> list[str]:
    """Select all non-excluded pages or an explicit approved ordered subset."""

    if selected_titles is None:
        selected_titles = [
            title
            for title in discovered_titles
            if not any(title.startswith(prefix) for prefix in excluded_prefixes)
        ]
        if not selected_titles:
            raise ValueError("Wikisource scope selected no primary-text subpages")
        return selected_titles
    missing_titles = [title for title in selected_titles if title not in discovered_titles]
    if missing_titles:
        raise ValueError(f"approved Wikisource subpages are missing: {missing_titles}")
    return selected_titles


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
        _sleep_with_progress(delay_seconds, progress)
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


def _sleep_with_progress(delay_seconds: float, progress: ProgressCallback | None) -> None:
    """Sleep through an API cooldown while keeping long-running scripts observable."""

    remaining = delay_seconds
    while remaining > 0:
        interval = min(10.0, remaining)
        sleep(interval)
        remaining -= interval
        if remaining > 0:
            _write_progress(progress, f"rate-limit cooldown remaining: {remaining:g} seconds")


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
