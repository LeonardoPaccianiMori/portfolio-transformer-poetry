"""Liber Liber download discovery and text extraction helpers."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from time import monotonic, sleep
from urllib.parse import parse_qs, urljoin, urlparse
from xml.etree import ElementTree

import requests


LIBER_LIBER_USER_AGENT = "portfolio-transformer-poetry-liber-liber-probe/0.1"
TEXT_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
OFFICE_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"


@dataclass(frozen=True)
class LiberLiberDownloadCandidate:
    archive_format: str
    url: str


@dataclass(frozen=True)
class FetchedLiberLiberText:
    landing_page_url: str
    download_page_url: str
    archive_url: str
    archive_format: str
    raw_byte_count: int
    text: str


class _HtmlUrlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "a" and attributes.get("href"):
            self.urls.append(attributes["href"])
        if tag == "input" and attributes.get("value"):
            self.urls.append(attributes["value"])


def discover_download_candidates(
    html: str,
    *,
    base_url: str,
) -> list[LiberLiberDownloadCandidate]:
    """Find supported download-page links in preferred format order."""

    parser = _HtmlUrlParser()
    parser.feed(html)
    candidates: list[LiberLiberDownloadCandidate] = []

    for url in parser.urls:
        absolute_url = urljoin(base_url, url)
        download_type = parse_qs(urlparse(absolute_url).query).get("type", [""])[0]
        archive_format = {
            "opera_url_txt": "txt_zip",
            "opera_url_odt": "odt",
        }.get(download_type)
        if archive_format is not None:
            candidates.append(
                LiberLiberDownloadCandidate(
                    archive_format=archive_format,
                    url=absolute_url,
                )
            )

    format_priority = {"txt_zip": 0, "odt": 1}
    return sorted(candidates, key=lambda item: format_priority[item.archive_format])


def discover_archive_url(
    html: str,
    *,
    base_url: str,
    archive_format: str,
) -> str:
    """Find the actual media archive URL on a Liber Liber download page."""

    parser = _HtmlUrlParser()
    parser.feed(html)
    expected_suffix = ".zip" if archive_format == "txt_zip" else ".odt"

    for url in parser.urls:
        absolute_url = urljoin(base_url, url)
        if urlparse(absolute_url).path.lower().endswith(expected_suffix):
            return absolute_url

    raise ValueError(f"download page does not contain a {expected_suffix} archive URL")


def extract_txt_zip_text(archive_bytes: bytes) -> str:
    """Decode the largest text member from a Liber Liber TXT ZIP archive."""

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        text_members = [
            info
            for info in archive.infolist()
            if not info.is_dir()
            and info.filename.lower().endswith(".txt")
            and "__macosx" not in info.filename.lower()
        ]
        if not text_members:
            raise ValueError("TXT ZIP archive does not contain a .txt file")

        member = max(text_members, key=lambda info: info.file_size)
        return _decode_text_bytes(archive.read(member))


def extract_odt_text(archive_bytes: bytes) -> str:
    """Extract paragraph text after the generated ODT table of contents."""

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        try:
            content_xml = archive.read("content.xml")
        except KeyError as exc:
            raise ValueError("ODT archive does not contain content.xml") from exc

    root = ElementTree.fromstring(content_xml)
    office_text = root.find(f".//{{{OFFICE_NAMESPACE}}}text")
    if office_text is None:
        raise ValueError("ODT content.xml does not contain office:text")

    has_table_of_contents = office_text.find(
        f".//{{{TEXT_NAMESPACE}}}table-of-content"
    ) is not None
    paragraphs: list[str] = []
    state = {"started": not has_table_of_contents}
    _collect_odt_paragraphs(office_text, paragraphs, state)
    return _normalize_text("\n".join(paragraphs))


def strip_liber_liber_boilerplate(text: str, *, title: str) -> str:
    """Remove the standard leading Liber Liber metadata and donation wrapper."""

    normalized = _normalize_text(text)
    lines = normalized.splitlines()
    title_candidates = _title_candidates(title)
    matching_indexes = [
        index
        for index, line in enumerate(lines)
        if line.strip().casefold() in title_candidates
    ]

    wrapper_end = _last_wrapper_line(lines[:250])
    after_wrapper = [index for index in matching_indexes if index > wrapper_end]
    if after_wrapper:
        return _normalize_text("\n".join(lines[after_wrapper[0] :]))
    if len(matching_indexes) >= 2:
        return _normalize_text("\n".join(lines[matching_indexes[1] :]))
    if matching_indexes and matching_indexes[0] > 5:
        return _normalize_text("\n".join(lines[matching_indexes[0] :]))

    return normalized


def fetch_liber_liber_text(
    landing_page_url: str,
    *,
    title: str,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> FetchedLiberLiberText:
    """Discover, download, extract, and clean one Liber Liber work."""

    http = session or requests.Session()
    http.headers.update({"User-Agent": LIBER_LIBER_USER_AGENT})
    landing_response = http.get(landing_page_url, timeout=timeout)
    landing_response.raise_for_status()
    candidates = discover_download_candidates(
        landing_response.text,
        base_url=landing_page_url,
    )
    if not candidates:
        raise FileNotFoundError("Liber Liber landing page has no supported TXT or ODT link")

    errors: list[str] = []
    for candidate in candidates:
        try:
            download_response = http.get(candidate.url, timeout=timeout)
            download_response.raise_for_status()
            archive_url = discover_archive_url(
                download_response.text,
                base_url=candidate.url,
                archive_format=candidate.archive_format,
            )
            archive_response = http.get(archive_url, timeout=timeout)
            archive_response.raise_for_status()
            extracted = _extract_archive_text(
                archive_response.content,
                candidate.archive_format,
            )
            cleaned = strip_liber_liber_boilerplate(extracted, title=title)
        except (OSError, ValueError, requests.RequestException) as exc:
            errors.append(f"{candidate.archive_format}: {exc}")
            continue

        return FetchedLiberLiberText(
            landing_page_url=landing_page_url,
            download_page_url=candidate.url,
            archive_url=archive_url,
            archive_format=candidate.archive_format,
            raw_byte_count=len(archive_response.content),
            text=cleaned,
        )

    raise FileNotFoundError(
        "could not fetch a supported Liber Liber text: " + "; ".join(errors)
    )


class LiberLiberRateLimiter:
    """Small request-delay helper for polite sequential probes."""

    def __init__(self, request_delay: float) -> None:
        self.request_delay = request_delay
        self._last_request_at = 0.0

    def wait(self) -> None:
        elapsed = monotonic() - self._last_request_at
        if elapsed < self.request_delay:
            sleep(self.request_delay - elapsed)
        self._last_request_at = monotonic()


def _extract_archive_text(archive_bytes: bytes, archive_format: str) -> str:
    if archive_format == "txt_zip":
        return extract_txt_zip_text(archive_bytes)
    if archive_format == "odt":
        return extract_odt_text(archive_bytes)
    raise ValueError(f"unsupported Liber Liber archive format: {archive_format}")


def _decode_text_bytes(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1")


def _collect_odt_paragraphs(
    element: ElementTree.Element,
    paragraphs: list[str],
    state: dict[str, bool],
) -> None:
    table_of_contents_tag = f"{{{TEXT_NAMESPACE}}}table-of-content"
    paragraph_tags = {
        f"{{{TEXT_NAMESPACE}}}h",
        f"{{{TEXT_NAMESPACE}}}p",
    }

    for child in element:
        if child.tag == table_of_contents_tag:
            state["started"] = True
            continue
        if child.tag in paragraph_tags:
            if state["started"]:
                paragraph = _extract_odt_inline_text(child).strip()
                if paragraph:
                    paragraphs.append(paragraph)
            continue
        _collect_odt_paragraphs(child, paragraphs, state)


def _extract_odt_inline_text(element: ElementTree.Element) -> str:
    note_tag = f"{{{TEXT_NAMESPACE}}}note"
    space_tag = f"{{{TEXT_NAMESPACE}}}s"
    tab_tag = f"{{{TEXT_NAMESPACE}}}tab"
    line_break_tag = f"{{{TEXT_NAMESPACE}}}line-break"
    parts = [element.text or ""]

    for child in element:
        if child.tag == note_tag:
            pass
        elif child.tag == space_tag:
            count = int(child.attrib.get(f"{{{TEXT_NAMESPACE}}}c", "1"))
            parts.append(" " * count)
        elif child.tag == tab_tag:
            parts.append("\t")
        elif child.tag == line_break_tag:
            parts.append("\n")
        else:
            parts.append(_extract_odt_inline_text(child))
        parts.append(child.tail or "")

    return "".join(parts)


def _title_candidates(title: str) -> set[str]:
    base_title = re.sub(r"\s*\[[^]]+\]\s*$", "", title).strip()
    return {title.casefold(), base_title.casefold()}


def _last_wrapper_line(lines: list[str]) -> int:
    wrapper_markers = (
        "progetto manuzio",
        "liberliber.it",
        "liber liber",
        "questo e-book",
    )
    indexes = [
        index
        for index, line in enumerate(lines)
        if any(marker in line.casefold() for marker in wrapper_markers)
    ]
    return max(indexes, default=-1)


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"
