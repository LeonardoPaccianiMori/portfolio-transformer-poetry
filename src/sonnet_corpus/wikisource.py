"""Wikisource fetching and HTML parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from urllib.parse import quote, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag


WIKISOURCE_BASE = "https://it.wikisource.org"


@dataclass(frozen=True)
class FetchedPage:
    url: str
    html: str
    downloaded_at_utc: str
    revision_id: str = ""
    revision_timestamp: str = ""


@dataclass(frozen=True)
class PageLink:
    url: str
    title: str
    section: str


class WikisourceClient:
    """Small HTTP client with local raw HTML caching."""

    def __init__(
        self,
        raw_dir: Path,
        timeout: int = 30,
        request_delay: float = 1.0,
        retries: int = 5,
    ) -> None:
        self.raw_dir = raw_dir
        self.timeout = timeout
        self.request_delay = request_delay
        self.retries = retries
        self._last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "portfolio-transformer-poetry-corpus-builder/0.1"}
        )

    def fetch(self, url: str) -> FetchedPage:
        cache_path = self._cache_path(url)
        if cache_path.is_file():
            downloaded_at = datetime.fromtimestamp(
                cache_path.stat().st_mtime, tz=UTC
            ).replace(microsecond=0)
            return FetchedPage(
                url=url,
                html=cache_path.read_text(encoding="utf-8"),
                downloaded_at_utc=downloaded_at.isoformat(),
            )

        response = self._get(url)
        downloaded_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        page = FetchedPage(url=url, html=response.text, downloaded_at_utc=downloaded_at)
        self._cache_raw_html(page)
        return page

    def category_members(self, category_title: str) -> list[str]:
        """Return page titles from a Wikisource category."""

        titles: list[str] = []
        cmcontinue = ""
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category_title,
                "cmlimit": "500",
                "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            response = self._get("https://it.wikisource.org/w/api.php", params=params)
            payload = response.json()
            for member in payload.get("query", {}).get("categorymembers", []):
                if member.get("ns") == 0 and "title" in member:
                    titles.append(member["title"])
            cmcontinue = payload.get("continue", {}).get("cmcontinue", "")
            if not cmcontinue:
                return titles

    def _wait_for_slot(self) -> None:
        elapsed = monotonic() - self._last_request_at
        if elapsed < self.request_delay:
            sleep(self.request_delay - elapsed)
        self._last_request_at = monotonic()

    def _cache_raw_html(self, page: FetchedPage) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(page.url).write_text(page.html, encoding="utf-8")

    def _cache_path(self, url: str) -> Path:
        return self.raw_dir / f"{slug_from_url(url)}.html"

    def _get(self, url: str, params: dict[str, str] | None = None) -> requests.Response:
        response = None
        for attempt in range(self.retries):
            self._wait_for_slot()
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code != 429:
                break
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait_seconds = float(retry_after)
            else:
                wait_seconds = self.request_delay * (attempt + 2)
            sleep(wait_seconds)
        if response is None:
            raise RuntimeError(f"request was not attempted: {url}")
        response.raise_for_status()
        return response


def url_from_title(title: str) -> str:
    encoded_title = quote(title.replace(" ", "_"), safe="()_',:")
    return f"{WIKISOURCE_BASE}/wiki/{encoded_title}"


def slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    title = parsed.path.removeprefix("/wiki/")
    title = unquote(title).strip("/")
    slug = []
    for char in title.lower():
        if char.isalnum():
            slug.append(char)
        elif char in {"_", "-", "/"}:
            slug.append("_")
    compact = "".join(slug).strip("_")
    return compact or "index"


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def content_root(soup: BeautifulSoup) -> Tag:
    root = soup.select_one("div.mw-parser-output")
    if root is None:
        raise ValueError("could not find Wikisource content root")
    return root


def clean_link_title(title: str) -> str:
    return " ".join(title.split()).strip()


def iter_index_links(html: str, index_url: str) -> list[PageLink]:
    """Return content links with nearest heading text as section."""

    soup = soup_from_html(html)
    root = content_root(soup)
    links: list[PageLink] = []
    current_section = ""

    for element in root.descendants:
        if not isinstance(element, Tag):
            continue
        if element.name in {"h2", "h3", "h4"}:
            headline = element.get_text(" ", strip=True)
            current_section = headline.replace("[modifica]", "").strip()
            continue
        if element.name != "a":
            continue

        href = element.get("href", "")
        if not href.startswith("/wiki/"):
            continue
        if ":" in href.removeprefix("/wiki/").split("/", 1)[0]:
            continue
        title = clean_link_title(element.get_text(" ", strip=True))
        if not title:
            continue
        links.append(
            PageLink(
                url=urljoin(index_url, href),
                title=title,
                section=current_section,
            )
        )

    return dedupe_links(links)


def dedupe_links(links: list[PageLink]) -> list[PageLink]:
    seen: set[str] = set()
    unique: list[PageLink] = []
    for link in links:
        if link.url in seen:
            continue
        seen.add(link.url)
        unique.append(link)
    return unique


def extract_poem_text(html: str) -> str:
    """Extract likely poem lines from a Wikisource poem page."""

    soup = soup_from_html(html)
    root = content_root(soup)

    for unwanted in root.select(
        "table, .metadata, .noprint, .printfooter, .mw-editsection, style, script"
    ):
        unwanted.decompose()

    poem_nodes = root.select(".poem")
    if poem_nodes:
        poem_texts: list[str] = []
        for node in poem_nodes:
            # Wikisource often wraps initials, links, and line numbers in spans.
            # Only <br> marks a poetic line break; inline markup must stay joined.
            for line_number in node.select(".numeroriga"):
                line_number.decompose()
            for line_break in node.select("br"):
                line_break.replace_with("\n")
            lines = [line.strip() for line in node.get_text().splitlines() if line.strip()]
            poem_texts.append("\n".join(lines))
        return "\n".join(poem_texts).strip()

    lines: list[str] = []
    for element in root.find_all(["p", "div"], recursive=True):
        classes = set(element.get("class", []))
        if classes.intersection({"references", "licenseContainer"}):
            continue
        text = element.get_text("\n", strip=True)
        if text:
            lines.append(text)
    return "\n".join(lines).strip()
