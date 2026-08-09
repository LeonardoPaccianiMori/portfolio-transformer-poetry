"""Biblioteca Italiana catalog access and structure-aware TEI parsing."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from html import entities as html_entities
from dataclasses import asdict, dataclass
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import quote

import requests

from .bibit_legacy_entities import BIBIT_LEGACY_ENTITIES


BIBIT_ARCHIVE_NAME = "Biblioteca Italiana"
BIBIT_CATALOG_URL = (
    "http://backend.bibliotecaitaliana.it/wp-json/muruca/v1/solr/select"
)
BIBIT_API_URL = "http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1"
BIBIT_LANDING_URL = "http://www.bibliotecaitaliana.it/"
BIBIT_FAQ_URL = "http://backend.bibliotecaitaliana.it/faq/"
BIBIT_PROJECT_URL = "https://bibliodlcm.web.uniroma1.it/it/biblioteca-italiana"

_OBJECT_ID_PATTERN = re.compile(r"bibit[0-9]{5,6}\Z")
_WHITESPACE = re.compile(r"[\t\f\v ]+")
_SONNET_TYPE = re.compile(r"^sonett", re.IGNORECASE)
_EXPLICIT_LINE_BREAK = "\ue000"
_NAMED_ENTITY = re.compile(r"&([A-Za-z][A-Za-z0-9._:-]*);")
_XML_ENTITIES = {"amp", "apos", "gt", "lt", "quot"}
_INVALID_XML_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_OMITTED_ELEMENTS = {
    "argument",
    "back",
    "figdesc",
    "figure",
    "front",
    "fw",
    "index",
    "milestone",
    "note",
    "pb",
    "rdg",
}
_BLOCK_ELEMENTS = {
    "byline",
    "closer",
    "dateline",
    "head",
    "item",
    "l",
    "label",
    "opener",
    "p",
    "salute",
    "signed",
    "speaker",
    "stage",
    "trailer",
}
_BOUNDARY_ELEMENTS = {
    "body",
    "castlist",
    "div",
    "div0",
    "div1",
    "div2",
    "div3",
    "div4",
    "div5",
    "lg",
    "list",
    "sp",
    "text",
}


@dataclass(frozen=True)
class BibItCatalogRecord:
    """Normalized metadata for one BibIt text record."""

    object_id: str
    wordpress_id: str
    title: str
    authors: tuple[str, ...]
    genres: tuple[str, ...]
    periods: tuple[str, ...]
    languages: tuple[str, ...]
    source_authors: tuple[str, ...]
    source_publisher: str
    source_publication_place: str
    source_publication_date: str
    source_identifier: str
    source_modified_utc: str

    @property
    def landing_page_url(self) -> str:
        return f"{BIBIT_LANDING_URL}scheda/{self.object_id}"

    @property
    def xml_url(self) -> str:
        return f"{BIBIT_API_URL}/xml/{self.object_id}"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["landing_page_url"] = self.landing_page_url
        payload["xml_url"] = self.xml_url
        return payload


@dataclass(frozen=True)
class BibItProvenance:
    """Edition and digitization provenance extracted from one TEI header."""

    object_id: str
    digital_title: str
    digital_authors: tuple[str, ...]
    digital_publisher: str
    digital_publication_place: str
    digital_publication_date: str
    availability: str
    source_titles: tuple[str, ...]
    source_authors: tuple[str, ...]
    source_editors: tuple[str, ...]
    source_publisher: str
    source_publication_place: str
    source_publication_date: str
    source_identifier: str
    creation_date: str
    languages: tuple[str, ...]
    genres: tuple[str, ...]
    editorial_notes: tuple[str, ...]
    revisions: tuple[str, ...]


@dataclass(frozen=True)
class BibItSonnetUnit:
    """One line-preserving explicit TEI sonnet unit."""

    unit_id: str
    sonnet_type: str
    heading_path: tuple[str, ...]
    text: str
    line_count: int


@dataclass(frozen=True)
class BibItVerseUnit:
    """One maximal line-preserving TEI verse unit that is not a sonnet."""

    unit_id: str
    verse_type: str
    heading_path: tuple[str, ...]
    text: str
    line_count: int


@dataclass(frozen=True)
class ParsedBibItTEI:
    """A BibIt TEI document split into non-overlapping structural views."""

    provenance: BibItProvenance
    body_text: str
    non_sonnet_text: str
    sonnet_candidate_safe_text: str
    residual_text: str
    sonnets: tuple[BibItSonnetUnit, ...]
    non_sonnet_verse: tuple[BibItVerseUnit, ...]
    structural_sonnet_candidates: tuple[BibItVerseUnit, ...]


CatalogProgress = Callable[[str], None]


def fetch_bibit_catalog(
    *,
    periods: Iterable[str] = ("Origini", "200", "300", "400", "500", "600", "700", "800"),
    session: requests.Session | None = None,
    timeout: float = 120.0,
    progress: CatalogProgress | None = None,
) -> list[BibItCatalogRecord]:
    """Fetch normalized Italian BibIt metadata for the selected period facets."""

    selected_periods = tuple(dict.fromkeys(periods))
    if not selected_periods:
        raise ValueError("at least one BibIt period is required")
    if any(not re.fullmatch(r"Origini|[0-9]{3}", period) for period in selected_periods):
        raise ValueError("invalid BibIt period facet")

    query_periods = " OR ".join(selected_periods)
    params = {
        "q": "*:* AND post_type:muruca_resource",
        "fl": (
            "obj_id_s,post_title,author_str,resource_genre_str,"
            "resource_period_str,resource_language_str,source_desc_pub_date_str,"
            "source_desc_pub_place_str,source_desc_publisher_str,"
            "source_desc_author_str,source_desc_identifier_str,"
            "post_modified_gmt,collection_str,ID"
        ),
        "fq": [
            "collection_str:bibit",
            "resource_language_str:ita",
            f"resource_period_str:({query_periods})",
        ],
        "sort": "author_sort asc,post_title asc",
        "rows": "3000",
        "wt": "json",
    }
    _report(progress, "requesting Italian text catalog metadata")
    client = session or requests.Session()
    response = client.get(
        BIBIT_CATALOG_URL,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "portfolio-transformer-poetry/0.1 corpus audit"},
    )
    response.raise_for_status()
    payload = response.json()
    docs = _solr_docs(payload)
    records = [catalog_record_from_solr(doc) for doc in docs]
    if len(records) != payload["response"].get("numFound"):
        raise ValueError("BibIt catalog response was truncated")
    _report(progress, f"received {len(records):,} catalog records")
    return records


def fetch_bibit_rendered_texts(
    object_ids: Iterable[str],
    *,
    session: requests.Session | None = None,
    timeout: float = 300.0,
    batch_size: int = 100,
    progress: CatalogProgress | None = None,
) -> dict[str, str]:
    """Fetch rendered HTML for a bounded metadata-audit sample."""

    selected_ids = tuple(dict.fromkeys(object_ids))
    if not selected_ids:
        return {}
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    for object_id in selected_ids:
        _validate_object_id(object_id)

    client = session or requests.Session()
    rendered: dict[str, str] = {}
    batches = [
        selected_ids[index : index + batch_size]
        for index in range(0, len(selected_ids), batch_size)
    ]
    for index, batch in enumerate(batches, start=1):
        _report(progress, f"fetching rendered sample batch {index}/{len(batches)}")
        params = {
            "q": "obj_id_s:(" + " OR ".join(batch) + ")",
            "fl": "obj_id_s,text_html",
            "fq": "post_type:muruca_resource",
            "rows": str(len(batch)),
            "wt": "json",
        }
        response = client.get(
            BIBIT_CATALOG_URL,
            params=params,
            timeout=timeout,
            headers={"User-Agent": "portfolio-transformer-poetry/0.1 corpus audit"},
        )
        response.raise_for_status()
        for doc in _solr_docs(response.json()):
            object_id = _required_string(doc, "obj_id_s")
            html = doc.get("text_html", "")
            if not isinstance(html, str):
                raise ValueError(f"BibIt rendered text is invalid: {object_id}")
            rendered[object_id] = html

    missing = sorted(set(selected_ids) - set(rendered))
    if missing:
        raise ValueError("BibIt sample response omitted records: " + ", ".join(missing))
    return rendered


def fetch_bibit_tei(
    object_id: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 120.0,
) -> bytes:
    """Download one TEI document by its validated BibIt object identifier."""

    _validate_object_id(object_id)
    client = session or requests.Session()
    response = client.get(
        f"{BIBIT_API_URL}/xml/{quote(object_id)}",
        timeout=timeout,
        headers={"User-Agent": "portfolio-transformer-poetry/0.1 corpus builder"},
    )
    response.raise_for_status()
    if not response.content.strip():
        raise ValueError(f"BibIt returned empty TEI: {object_id}")
    return response.content


def catalog_record_from_solr(doc: dict[str, Any]) -> BibItCatalogRecord:
    """Normalize one Solr record and reject missing identity metadata."""

    object_id = _required_string(doc, "obj_id_s")
    _validate_object_id(object_id)
    return BibItCatalogRecord(
        object_id=object_id,
        wordpress_id=_required_string(doc, "ID"),
        title=_required_string(doc, "post_title"),
        authors=_string_tuple(doc.get("author_str")),
        genres=_string_tuple(doc.get("resource_genre_str")),
        periods=_string_tuple(doc.get("resource_period_str")),
        languages=_string_tuple(doc.get("resource_language_str")),
        source_authors=_string_tuple(doc.get("source_desc_author_str")),
        source_publisher=_first_string(doc.get("source_desc_publisher_str")),
        source_publication_place=_first_string(doc.get("source_desc_pub_place_str")),
        source_publication_date=_first_string(doc.get("source_desc_pub_date_str")),
        source_identifier=_first_string(doc.get("source_desc_identifier_str")),
        source_modified_utc=str(doc.get("post_modified_gmt", "") or ""),
    )


def parse_bibit_tei(xml: bytes | str, *, object_id: str = "") -> ParsedBibItTEI:
    """Parse BibIt TEI without loading its external DTD or network resources."""

    if object_id:
        _validate_object_id(object_id)
    text = xml.decode("utf-8-sig") if isinstance(xml, bytes) else xml.lstrip("\ufeff")
    secured_text = _remove_doctype(text)
    if re.search(r"<!ENTITY\b", secured_text, re.IGNORECASE):
        raise ValueError("TEI entity declarations are not permitted")
    secured_text = _INVALID_XML_CONTROL.sub("", secured_text)
    secured_text = _replace_known_named_entities(secured_text)
    try:
        root = ET.fromstring(secured_text)
    except ET.ParseError as error:
        raise ValueError(f"invalid BibIt TEI: {error}") from error

    header = _first_descendant(root, "teiheader")
    body = _first_descendant(root, "body")
    if header is None or body is None:
        raise ValueError("BibIt TEI requires both teiHeader and body")

    resolved_object_id = object_id or _first_text(_descendants(header, "idno"))
    provenance = _parse_provenance(header, resolved_object_id)
    parent_by_id = {
        id(child): parent for parent in root.iter() for child in list(parent)
    }
    sonnet_elements = _top_level_sonnet_elements(body, parent_by_id)
    sonnets = tuple(
        _parse_sonnet_unit(element, index, parent_by_id, body)
        for index, element in enumerate(sonnet_elements, start=1)
    )
    verse_elements = _non_sonnet_verse_elements(body, sonnet_elements)
    non_sonnet_verse = tuple(
        _parse_verse_unit(element, index, parent_by_id, body)
        for index, element in enumerate(verse_elements, start=1)
    )
    structural_candidate_elements = _structural_sonnet_candidate_elements(
        body,
        sonnet_elements,
    )
    structural_sonnet_candidates = tuple(
        _parse_verse_unit(
            element,
            index,
            parent_by_id,
            body,
            unit_prefix="structural",
        )
        for index, element in enumerate(structural_candidate_elements, start=1)
    )
    sonnet_ids = {id(element) for element in sonnet_elements}
    verse_ids = {id(element) for element in verse_elements}
    structural_sonnet_candidate_ids = {
        id(element) for element in structural_candidate_elements
    }
    body_text = _render_body(body, excluded_elements=set())
    non_sonnet_text = _render_body(body, excluded_elements=sonnet_ids)
    sonnet_candidate_safe_text = _render_body(
        body,
        excluded_elements=sonnet_ids | structural_sonnet_candidate_ids,
    )
    residual_text = _render_body(
        body,
        excluded_elements=sonnet_ids | verse_ids,
    )
    return ParsedBibItTEI(
        provenance=provenance,
        body_text=body_text,
        non_sonnet_text=non_sonnet_text,
        sonnet_candidate_safe_text=sonnet_candidate_safe_text,
        residual_text=residual_text,
        sonnets=sonnets,
        non_sonnet_verse=non_sonnet_verse,
        structural_sonnet_candidates=structural_sonnet_candidates,
    )


def _parse_provenance(header: ET.Element, object_id: str) -> BibItProvenance:
    file_desc = _first_descendant(header, "filedesc")
    if file_desc is None:
        raise ValueError("BibIt TEI header is missing fileDesc")
    title_stmt = _first_child(file_desc, "titlestmt")
    publication_stmt = _first_child(file_desc, "publicationstmt")
    source_desc = _first_child(file_desc, "sourcedesc")
    source_bibl = _first_descendant(source_desc, "bibl") if source_desc is not None else None
    profile_desc = _first_descendant(header, "profiledesc")
    encoding_desc = _first_descendant(header, "encodingdesc")
    revision_desc = _first_descendant(header, "revisiondesc")

    return BibItProvenance(
        object_id=object_id,
        digital_title=_first_text(_children(title_stmt, "title")),
        digital_authors=_texts(_children(title_stmt, "author")),
        digital_publisher=_first_text(_children(publication_stmt, "publisher")),
        digital_publication_place=_first_text(_children(publication_stmt, "pubplace")),
        digital_publication_date=_first_text(_children(publication_stmt, "date")),
        availability=_compact_text(_first_descendant(publication_stmt, "availability")),
        source_titles=_texts(_children(source_bibl, "title")),
        source_authors=_texts(_children(source_bibl, "author")),
        source_editors=_texts(_children(source_bibl, "editor")),
        source_publisher=_first_text(_children(source_bibl, "publisher")),
        source_publication_place=_first_text(_children(source_bibl, "pubplace")),
        source_publication_date=_first_text(_children(source_bibl, "date")),
        source_identifier=_first_text(_children(source_bibl, "idno")),
        creation_date=_compact_text(_first_descendant(profile_desc, "creation")),
        languages=_languages(profile_desc),
        genres=_texts(_descendants(profile_desc, "term")),
        editorial_notes=_editorial_notes(encoding_desc),
        revisions=_revisions(revision_desc),
    )


def _parse_sonnet_unit(
    element: ET.Element,
    index: int,
    parent_by_id: dict[int, ET.Element],
    body: ET.Element,
) -> BibItSonnetUnit:
    text, line_count = _render_verse_unit(element)
    heading_path = _heading_path(element, parent_by_id, body)
    return BibItSonnetUnit(
        unit_id=f"sonnet_{index:04d}",
        sonnet_type=element.attrib.get("type", "sonetto"),
        heading_path=heading_path,
        text=text,
        line_count=line_count,
    )


def _parse_verse_unit(
    element: ET.Element,
    index: int,
    parent_by_id: dict[int, ET.Element],
    body: ET.Element,
    unit_prefix: str = "verse",
) -> BibItVerseUnit:
    text, line_count = _render_verse_unit(element)
    return BibItVerseUnit(
        unit_id=f"{unit_prefix}_{index:04d}",
        verse_type=element.attrib.get("type", "verse") or "verse",
        heading_path=_heading_path(element, parent_by_id, body),
        text=text,
        line_count=line_count,
    )


def _render_verse_unit(element: ET.Element) -> tuple[str, int]:
    lines = [_inline_text(line) for line in _descendants(element, "l")]
    lines = [line for line in lines if line]
    rendered_lines: list[str] = []
    child_stanzas = [child for child in list(element) if _local_name(child.tag) == "lg"]
    if child_stanzas:
        for stanza_index, stanza in enumerate(child_stanzas):
            stanza_lines = [
                line
                for line in (_inline_text(item) for item in _descendants(stanza, "l"))
                if line
            ]
            if not stanza_lines:
                continue
            if stanza_index and rendered_lines:
                rendered_lines.append("")
            rendered_lines.extend(stanza_lines)
    else:
        rendered_lines = lines
    text = "\n".join(rendered_lines).strip()
    return (text + "\n" if text else "", len(lines))


def _render_body(body: ET.Element, *, excluded_elements: set[int]) -> str:
    blocks: list[str] = []

    def visit(element: ET.Element) -> None:
        if id(element) in excluded_elements:
            return
        name = _local_name(element.tag)
        if name in _OMITTED_ELEMENTS:
            return
        if name in _BLOCK_ELEMENTS:
            value = _inline_text(element, excluded_elements=excluded_elements)
            if value:
                blocks.extend(value.splitlines())
            if name != "l":
                blocks.append("")
            return
        for child in list(element):
            visit(child)
        if name in _BOUNDARY_ELEMENTS:
            blocks.append("")

    visit(body)
    normalized: list[str] = []
    for block in blocks:
        value = _normalize_line(block)
        if value:
            normalized.append(value)
        elif normalized and normalized[-1] != "":
            normalized.append("")
    while normalized and normalized[-1] == "":
        normalized.pop()
    return "\n".join(normalized) + ("\n" if normalized else "")


def _inline_text(
    element: ET.Element,
    *,
    excluded_elements: set[int] | None = None,
) -> str:
    excluded = excluded_elements or set()
    parts: list[str] = []

    def collect(node: ET.Element, *, include_tail: bool = True) -> None:
        if id(node) in excluded:
            if include_tail and node.tail:
                parts.append(node.tail)
            return
        name = _local_name(node.tag)
        if name in _OMITTED_ELEMENTS or name == "del":
            if include_tail and node.tail:
                parts.append(node.tail)
            return
        if name == "lb":
            parts.append(_EXPLICIT_LINE_BREAK)
        elif name == "gap":
            parts.append(" [...]")
        elif name in {"app", "choice"}:
            selected = _select_critical_text_child(node)
            if selected is not None:
                collect(selected, include_tail=False)
        else:
            if node.text:
                parts.append(node.text)
            for child in list(node):
                collect(child)
        if include_tail and node.tail:
            parts.append(node.tail)

    collect(element, include_tail=False)
    joined = " ".join("".join(parts).split())
    lines = [_normalize_line(line) for line in joined.split(_EXPLICIT_LINE_BREAK)]
    return "\n".join(line for line in lines if line)


def _select_critical_text_child(element: ET.Element) -> ET.Element | None:
    children = list(element)
    if not children:
        return None
    preferences = (
        ("lem",) if _local_name(element.tag) == "app" else
        ("orig", "sic", "abbr", "expan", "corr", "reg")
    )
    for preferred in preferences:
        for child in children:
            if _local_name(child.tag) == preferred:
                return child
    return children[0]


def _top_level_sonnet_elements(
    body: ET.Element,
    parent_by_id: dict[int, ET.Element],
) -> list[ET.Element]:
    candidates = [
        element
        for element in body.iter()
        if _local_name(element.tag) == "lg"
        and _SONNET_TYPE.match(element.attrib.get("type", "").strip())
    ]
    candidate_ids = {id(element) for element in candidates}
    return [
        element
        for element in candidates
        if not _has_ancestor_in(element, candidate_ids, parent_by_id, body)
    ]


def _non_sonnet_verse_elements(
    body: ET.Element,
    sonnet_elements: Iterable[ET.Element],
) -> list[ET.Element]:
    """Select maximal verse subtrees while excluding every explicit sonnet."""

    sonnet_ids = {id(element) for element in sonnet_elements}
    selected: list[ET.Element] = []

    def visit(element: ET.Element) -> None:
        if id(element) in sonnet_ids:
            return
        name = _local_name(element.tag)
        if name in _OMITTED_ELEMENTS:
            return
        if name == "lg":
            contains_sonnet = any(id(descendant) in sonnet_ids for descendant in element.iter())
            if not contains_sonnet:
                selected.append(element)
                return
        for child in list(element):
            visit(child)

    visit(body)
    return selected


def _structural_sonnet_candidate_elements(
    body: ET.Element,
    sonnet_elements: Iterable[ET.Element],
) -> list[ET.Element]:
    """Return innermost untyped verse containers containing exactly 14 lines."""

    sonnet_ids = {id(element) for element in sonnet_elements}
    candidates: list[ET.Element] = []
    for element in body.iter():
        name = _local_name(element.tag)
        if not (name == "lg" or name.startswith("div")) or id(element) in sonnet_ids:
            continue
        if any(id(descendant) in sonnet_ids for descendant in element.iter()):
            continue
        lines = [line for line in _descendants(element, "l") if _inline_text(line)]
        if len(lines) == 14:
            candidates.append(element)
    candidate_ids = {id(element) for element in candidates}
    return [
        element
        for element in candidates
        if not any(
            id(descendant) in candidate_ids and descendant is not element
            for descendant in element.iter()
        )
    ]


def _has_ancestor_in(
    element: ET.Element,
    candidate_ids: set[int],
    parent_by_id: dict[int, ET.Element],
    stop: ET.Element,
) -> bool:
    parent = parent_by_id.get(id(element))
    while parent is not None and parent is not stop:
        if id(parent) in candidate_ids:
            return True
        parent = parent_by_id.get(id(parent))
    return False


def _heading_path(
    element: ET.Element,
    parent_by_id: dict[int, ET.Element],
    body: ET.Element,
) -> tuple[str, ...]:
    ancestors: list[ET.Element] = []
    parent = parent_by_id.get(id(element))
    while parent is not None and parent is not body:
        if _local_name(parent.tag).startswith("div"):
            ancestors.append(parent)
        parent = parent_by_id.get(id(parent))
    headings: list[str] = []
    for ancestor in reversed(ancestors):
        head = _first_child(ancestor, "head")
        value = _inline_text(head) if head is not None else ""
        if value:
            headings.append(value)
    return tuple(headings)


def _remove_doctype(text: str) -> str:
    match = re.search(r"<!DOCTYPE\b", text, re.IGNORECASE)
    if match is None:
        return text
    index = match.end()
    bracket_depth = 0
    quote_character = ""
    while index < len(text):
        character = text[index]
        if quote_character:
            if character == quote_character:
                quote_character = ""
        elif character in {"'", '"'}:
            quote_character = character
        elif character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif character == ">" and bracket_depth == 0:
            declaration = text[match.start() : index + 1]
            if re.search(r"<!ENTITY\b", declaration, re.IGNORECASE):
                raise ValueError("TEI entity declarations are not permitted")
            return text[: match.start()] + text[index + 1 :]
        index += 1
    raise ValueError("unterminated TEI doctype declaration")


def _replace_known_named_entities(text: str) -> str:
    """Resolve only the standard HTML names used by legacy BibIt DTDs."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in _XML_ENTITIES:
            return match.group(0)
        value = html_entities.html5.get(f"{name};")
        if value is None:
            value = BIBIT_LEGACY_ENTITIES.get(name)
        if value is None:
            raise ValueError(f"unsupported TEI named entity: &{name};")
        return "".join(f"&#{ord(character)};" for character in value)

    return _NAMED_ENTITY.sub(replace, text)


def _editorial_notes(encoding_desc: ET.Element | None) -> tuple[str, ...]:
    if encoding_desc is None:
        return ()
    notes = []
    for element in encoding_desc.iter():
        if _local_name(element.tag) in {"p", "samplingdecl", "editorialdecl"}:
            text = _compact_text(element)
            if text and text not in notes:
                notes.append(text)
    return tuple(notes)


def _revisions(revision_desc: ET.Element | None) -> tuple[str, ...]:
    if revision_desc is None:
        return ()
    return tuple(
        value
        for value in (_compact_text(change) for change in _children(revision_desc, "change"))
        if value
    )


def _languages(profile_desc: ET.Element | None) -> tuple[str, ...]:
    values: list[str] = []
    for element in _descendants(profile_desc, "language"):
        value = _compact_text(element)
        if not value:
            value = (
                element.attrib.get("id", "")
                or element.attrib.get("ident", "")
                or element.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "")
            ).strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


def _solr_docs(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("BibIt Solr response must be an object")
    response = payload.get("response")
    if not isinstance(response, dict) or not isinstance(response.get("docs"), list):
        raise ValueError("BibIt Solr response is missing docs")
    docs = response["docs"]
    if any(not isinstance(doc, dict) for doc in docs):
        raise ValueError("BibIt Solr docs must be objects")
    return docs


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"BibIt record is missing required field: {key}")
    return value.strip()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(item.strip() for item in value if item.strip())
    raise ValueError("BibIt metadata list contains a non-string value")


def _first_string(value: Any) -> str:
    values = _string_tuple(value)
    return values[0] if values else ""


def _validate_object_id(object_id: str) -> None:
    if not _OBJECT_ID_PATTERN.fullmatch(object_id):
        raise ValueError(f"invalid BibIt object identifier: {object_id}")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _children(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    normalized_name = name.casefold()
    return [child for child in list(element) if _local_name(child.tag) == normalized_name]


def _descendants(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    normalized_name = name.casefold()
    return [child for child in element.iter() if _local_name(child.tag) == normalized_name]


def _first_child(element: ET.Element | None, name: str) -> ET.Element | None:
    children = _children(element, name)
    return children[0] if children else None


def _first_descendant(element: ET.Element | None, name: str) -> ET.Element | None:
    descendants = _descendants(element, name)
    return descendants[0] if descendants else None


def _texts(elements: Iterable[ET.Element]) -> tuple[str, ...]:
    return tuple(value for value in (_compact_text(element) for element in elements) if value)


def _first_text(elements: Iterable[ET.Element]) -> str:
    for element in elements:
        value = _compact_text(element)
        if value:
            return value
    return ""


def _compact_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return _normalize_line("".join(element.itertext()))


def _normalize_line(text: str) -> str:
    return _WHITESPACE.sub(" ", text.replace("\r", "")).strip()


def _report(progress: CatalogProgress | None, message: str) -> None:
    if progress is not None:
        progress(message)
