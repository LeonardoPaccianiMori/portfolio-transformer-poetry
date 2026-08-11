"""Revision-pinned, non-activating Italian Wikisource page extraction audit."""

from __future__ import annotations

import bz2
import csv
import hashlib
import html
import json
import re
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import monotonic
from typing import Any

from sonnet_corpus.gutenberg_fulltext_probe import (
    TextFingerprint,
    TextReference,
    _normalized_words,
    _rolling_shingle_hashes,
    fingerprint_text,
    measure_word_shingle_containment,
)


DUMP_DATE = "20260801"
DUMP_FILENAME = "itwikisource-20260801-pages-meta-current.xml.bz2"
DUMP_SHA1 = "cacf8406058d3cadcf520a399962e9029352bddb"
EXPECTED_ELIGIBLE_ROOTS = 4_641
NEAR_DUPLICATE_THRESHOLD = 0.8

EXTRACTION_FIELDS = (
    "work_root_id",
    "root_title",
    "landing_page_url",
    "proposed_role",
    "period_bucket",
    "author_evidence",
    "hierarchy_page_count",
    "selected_leaf_page_count",
    "matched_main_page_count",
    "direct_scan_title",
    "extraction_patterns",
    "required_proofread_page_count",
    "matched_proofread_page_count",
    "missing_proofread_pages",
    "revision_mismatches",
    "extracted_character_count",
    "extracted_word_count",
    "nonempty_line_count",
    "normalized_word_sha256",
    "quality_flags",
    "unresolved_markup",
    "internal_exact_duplicate_ids",
    "internal_near_duplicate_metrics",
    "bibit_overlap_metrics",
    "gutenberg_previous_pool_overlap_metrics",
    "gutenberg_pass_1b_overlap_metrics",
    "gutenberg_resolved_overlap_metrics",
    "existing_project_corpus_overlap_metrics",
    "protected_v6_overlap_metrics",
    "rendered_validation_status",
    "rendered_validation_min_containment",
    "checkpoint_4c_decision",
    "review_reason",
    "local_text_cache_path",
    "activation_status",
)

BOUNDARY_FIELDS = (
    "work_root_id",
    "root_title",
    "page_id",
    "page_title",
    "expected_revision_id",
    "dump_revision_id",
    "hierarchy_depth",
    "boundary_selection",
    "extraction_pattern",
    "proofread_page_count",
    "extracted_character_count",
    "extracted_sha256",
    "quality_flags",
    "rendered_validation_status",
    "rendered_validation_containment",
    "activation_status",
)

REVIEW_FIELDS = (
    "review_id",
    "work_root_id",
    "root_title",
    "proposed_role",
    "checkpoint_4c_decision",
    "quality_flags",
    "missing_proofread_pages",
    "overlap_summary",
    "required_action",
    "review_status",
    "activation_status",
)

_PAGES_TAG = re.compile(r"<pages\b(?P<attrs>[^>]*)/\s*>", re.IGNORECASE | re.DOTALL)
_PAGES_PAIR = re.compile(
    r"<pages\b(?P<attrs>[^>]*)>(?P<body>.*?)</pages\s*>",
    re.IGNORECASE | re.DOTALL,
)
_ATTRIBUTE = re.compile(
    r"(?P<name>[\w:-]+)\s*=\s*(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|(?P<bare>[^\s>]+))",
    re.DOTALL,
)
_DIRECT_PAGE = re.compile(
    r"\{\{\s*:?(?P<title>Pagina\s*:[^{}|]+)(?:\|[^{}]*)?\}\}",
    re.IGNORECASE,
)
_SECTION_BEGIN = re.compile(
    r"<section\s+begin\s*=\s*(?:\"(?P<d>[^\"]+)\"|'(?P<s>[^']+)'|(?P<b>[^\s/>]+))\s*/?>",
    re.IGNORECASE,
)
_SECTION_END = re.compile(
    r"<section\s+end\s*=\s*(?:\"(?P<d>[^\"]+)\"|'(?P<s>[^']+)'|(?P<b>[^\s/>]+))\s*/?>",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_EDITORIAL = re.compile(
    r"\b(?:nota del(?:l['’])?editore|nota del trascrittore|errata(?:\s+corrige)?|"
    r"indice alfabetico|avvertenza dell['’]editore)\b",
    re.IGNORECASE,
)
_DIALECT = re.compile(
    r"\b(?:dialetto|vernacolo|romanesco|bolognese|napoletano|veneziano|siciliano)\b",
    re.IGNORECASE,
)

_DISCARD_TEMPLATES = {
    "altra colonna",
    "altracolonna",
    "altraedizione",
    "altraversione",
    "asterism",
    "asterismo",
    "blocco a destra/fine",
    "blocco a destra/inizio",
    "blocco a sinistra/fine",
    "blocco a sinistra/inizio",
    "blocco centrato/fine",
    "blocco centrato/inizio",
    "colonna",
    "colonna/fine",
    "colonne",
    "colonne fine",
    "colonne spezza",
    "fi",
    "fine blocco",
    "fine paragrafo",
    "finecolonna",
    "finecolonne",
    "gap",
    "ids",
    "includiintestazione",
    "intestazione",
    "interprogetto",
    "ms",
    "nmis",
    "no rientro",
    "noindent",
    "nota separata",
    "nota disambigua",
    "nessunaindentatura",
    "ni",
    "nbsp",
    "outdent",
    "pg",
    "r",
    "raccolta",
    "riga punteggiata",
    "rigapunteggiata",
    "sezione note",
    "sezione  note",
    "spazi",
    "space",
    "ts",
    "vs",
    "vi",
    "voce indice",
    "voceindice",
    "rigaintestazione",
    "rigaintestazione2",
    "rigaindice",
    "header",
    "footer",
    "nop",
    "nop1",
    "pagequality",
    "qualità",
    "quality",
    "centrato fine",
    "fine",
    "rule",
    "hr",
    "clear",
}
_KEEP_ARGUMENT_TEMPLATES = {
    "a destra",
    "ac",
    "annotazione a lato",
    "annotazione a lato sin",
    "annotazioni a lato",
    "autorecitato",
    "autoreignoto",
    "blocco a destra",
    "blocco a sinistra",
    "blocco centrato",
    "capoletteravar",
    "center",
    "centrato",
    "cs",
    "ct",
    "ec",
    "indent",
    "indentatura",
    "indentinverso",
    "corsivo",
    "citazione",
    "citazione2",
    "lang",
    "larger",
    "larger block",
    "maiuscoletto",
    "mem",
    "nota",
    "nowrap",
    "poem t",
    "poem",
    "pt",
    "right",
    "rosso",
    "sc",
    "sans-serif",
    "sb",
    "smaller block",
    "smaller",
    "span",
    "spaziato",
    "tc",
    "testo",
    "testo citato",
    "testocitato",
    "type",
    "type block",
    "vc",
    "verso",
    "w",
    "wl",
    "x-larger",
    "x-larger block",
    "x-smaller",
    "xx-larger",
    "xx-larger block",
    "xx-smaller",
    "xxx-larger",
}

Progress = Callable[[str], None]


@dataclass(frozen=True)
class DumpPage:
    """One current-revision page selected from the XML dump."""

    title: str
    namespace: int
    page_id: int
    revision_id: int
    timestamp: str
    text: str


@dataclass(frozen=True)
class PagesTransclusion:
    """A ProofreadPage ``<pages>`` directive and its exact requested pages."""

    raw: str
    index: str
    page_titles: tuple[str, ...]
    from_section: str = ""
    to_section: str = ""


@dataclass(frozen=True)
class WikisourcePageExtractionConfig:
    """Inputs, local cache, and public evidence outputs for checkpoint 4C."""

    repo_root: Path
    dump_path: Path
    resolution_path: Path
    hierarchy_path: Path
    extraction_path: Path
    boundaries_path: Path
    review_path: Path
    json_report_path: Path
    markdown_report_path: Path
    local_cache_dir: Path
    bibit_record_manifest_path: Path
    broader_sources_manifest_path: Path
    sonnet_manifest_path: Path
    gutenberg_previous_probe_path: Path
    gutenberg_previous_cache_dir: Path
    gutenberg_pass_1b_probe_path: Path
    gutenberg_pass_1b_cache_dir: Path
    gutenberg_resolved_manifest_path: Path
    expected_dump_sha1: str = DUMP_SHA1
    expected_eligible_roots: int = EXPECTED_ELIGIBLE_ROOTS
    progress_interval: int = 25_000
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD


def sha1_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-1 digest for a pinned dump."""

    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_mediawiki_dump(
    path: Path,
    *,
    selected_titles: set[str] | None = None,
    progress_interval: int = 25_000,
    progress: Progress | None = None,
) -> Iterator[DumpPage]:
    """Stream current revisions without retaining the decompressed XML tree."""

    opener = bz2.open if path.suffix == ".bz2" else open
    parsed = 0
    with opener(path, "rb") as handle:
        for _event, element in ET.iterparse(handle, events=("end",)):
            if _local_name(element.tag) != "page":
                continue
            parsed += 1
            title = _child_text(element, "title")
            if selected_titles is None or normalize_title(title) in selected_titles:
                revision = _child(element, "revision")
                yield DumpPage(
                    title=title,
                    namespace=int(_child_text(element, "ns") or 0),
                    page_id=int(_child_text(element, "id")),
                    revision_id=int(_child_text(revision, "id")),
                    timestamp=_child_text(revision, "timestamp"),
                    text=_child_text(revision, "text"),
                )
            element.clear()
            if progress and parsed % progress_interval == 0:
                progress(f"xml-pages parsed={parsed:,}")


def parse_pages_transclusions(wikitext: str) -> list[PagesTransclusion]:
    """Parse supported ProofreadPage ranges, including include/exclude lists."""

    directives: list[PagesTransclusion] = []
    matches = [*_PAGES_PAIR.finditer(wikitext), *_PAGES_TAG.finditer(wikitext)]
    for match in sorted(matches, key=lambda item: item.start()):
        attrs = _parse_attributes(match.group("attrs"))
        index = attrs.get("index", "").strip()
        if not index:
            raise ValueError("<pages> transclusion has no index attribute")
        labels = _page_labels(attrs)
        titles = tuple(
            normalize_title(f"Pagina:{index}/{label}")
            for label in labels
        )
        directives.append(
            PagesTransclusion(
                raw=match.group(0),
                index=normalize_title(index),
                page_titles=titles,
                from_section=attrs.get("fromsection", attrs.get("from_section", "")),
                to_section=attrs.get("tosection", attrs.get("to_section", "")),
            )
        )
    return directives


def direct_page_transclusions(wikitext: str) -> tuple[str, ...]:
    """Return directly transcluded ``Pagina:`` titles in source order."""

    return tuple(normalize_title(match.group("title")) for match in _DIRECT_PAGE.finditer(wikitext))


def extract_section(text: str, *, begin: str = "", end: str = "") -> str:
    """Apply ProofreadPage section boundaries, rejecting missing markers."""

    result = text
    if begin:
        match = _find_section(_SECTION_BEGIN, result, begin)
        if match is None:
            raise ValueError(f"missing section begin marker: {begin}")
        result = result[match.end() :]
    if end:
        match = _find_section(_SECTION_END, result, end)
        if match is None:
            raise ValueError(f"missing section end marker: {end}")
        result = result[: match.start()]
    return result


def clean_wikisource_wikitext(text: str, *, page_namespace: bool = False) -> tuple[str, list[str]]:
    """Conservatively remove markup and report every unresolved construct."""

    flags: set[str] = set()
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL)
    if page_namespace:
        value = re.sub(r"<noinclude\b[^>]*>.*?</noinclude\s*>", "", value, flags=re.I | re.S)
    value = re.sub(r"</?(?:includeonly|onlyinclude)\b[^>]*>", "", value, flags=re.I)
    value = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", "", value, flags=re.I | re.S)
    value = re.sub(r"<ref\b[^>]*/\s*>", "", value, flags=re.I)
    value = re.sub(r"<references\b[^>]*/?\s*>", "", value, flags=re.I)
    for tag in ("gallery", "timeline", "math", "score"):
        if re.search(fr"<{tag}\b", value, flags=re.I):
            flags.add(f"removed_{tag}_block")
        value = re.sub(fr"<{tag}\b[^>]*>.*?</{tag}\s*>", "", value, flags=re.I | re.S)
    if "{|" in value:
        flags.add("removed_wikitable")
        value = re.sub(r"\{\|.*?\|\}", "", value, flags=re.S)

    value, template_flags = _replace_templates(value)
    flags.update(template_flags)
    value = re.sub(r"\[\[(?:File|Immagine|Category|Categoria):[^\]]+\]\]", "", value, flags=re.I)
    value = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"\[\[([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"\[(?:https?://\S+)\s+([^\]]+)\]", r"\1", value)
    value = re.sub(r"\[(?:https?://[^\]]+)\]", "", value)
    value = re.sub(r"<section\b[^>]*/?\s*>", "", value, flags=re.I)
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(
        r"</?(?:poem|div|span|p|center|small|big|sup|sub|i|b|em|strong|font|u|s|strike|"
        r"blockquote|nowiki|ol|ul|li|dl|dt|dd)\b[^>]*>",
        "",
        value,
        flags=re.I,
    )
    if re.search(r"<[^>]+>", value):
        flags.add("unresolved_html_tag")
        value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"^\s*=+\s*(.*?)\s*=+\s*$", r"\1", value, flags=re.M)
    value = value.replace("'''", "").replace("''", "")
    value = html.unescape(value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    if "{{" in value or "}}" in value:
        flags.add("unresolved_template_markup")
    if "<pages" in value.casefold():
        flags.add("unresolved_pages_transclusion")
    if "�" in value:
        flags.add("replacement_character")
    return value, sorted(flags)


def reconstruct_page(
    wikitext: str,
    proofread_pages: dict[str, DumpPage],
) -> tuple[str, dict[str, Any]]:
    """Resolve supported page transclusions and clean one hierarchy leaf."""

    directives = parse_pages_transclusions(wikitext)
    direct_titles = direct_page_transclusions(wikitext)
    required = [title for item in directives for title in item.page_titles]
    required.extend(direct_titles)
    missing = sorted({title for title in required if title not in proofread_pages})
    flags: set[str] = set()
    value = wikitext
    for directive in reversed(directives):
        parts: list[str] = []
        for offset, title in enumerate(directive.page_titles):
            page = proofread_pages.get(title)
            if page is None:
                continue
            page_text = extract_section(
                page.text,
                begin=directive.from_section if offset == 0 else "",
                end=directive.to_section if offset == len(directive.page_titles) - 1 else "",
            )
            cleaned, page_flags = clean_wikisource_wikitext(page_text, page_namespace=True)
            flags.update(page_flags)
            if cleaned:
                parts.append(cleaned)
        value = value.replace(directive.raw, "\n\n".join(parts), 1)
    for title in direct_titles:
        page = proofread_pages.get(title)
        replacement = ""
        if page is not None:
            replacement, page_flags = clean_wikisource_wikitext(page.text, page_namespace=True)
            flags.update(page_flags)
        value = _DIRECT_PAGE.sub(replacement, value, count=1)
    cleaned, main_flags = clean_wikisource_wikitext(value)
    flags.update(main_flags)
    pattern = (
        "pages_range"
        if directives
        else "direct_page_transclusion"
        if direct_titles
        else "inline_main_wikitext"
    )
    return cleaned, {
        "pattern": pattern,
        "required_titles": required,
        "missing_titles": missing,
        "quality_flags": sorted(flags),
    }


class PageCache:
    """Small resumable SQLite cache for selected dump revisions."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS pages ("
            "title TEXT PRIMARY KEY, namespace INTEGER NOT NULL, page_id INTEGER NOT NULL, "
            "revision_id INTEGER NOT NULL, timestamp TEXT NOT NULL, text TEXT NOT NULL)"
        )

    def put_many(self, pages: Iterable[DumpPage]) -> int:
        values = [
            (page.title, page.namespace, page.page_id, page.revision_id, page.timestamp, page.text)
            for page in pages
        ]
        self.connection.executemany(
            "INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?, ?)", values
        )
        self.connection.commit()
        return len(values)

    def get_many(self, titles: Iterable[str]) -> dict[str, DumpPage]:
        result: dict[str, DumpPage] = {}
        for batch in _batches(sorted(set(titles)), 500):
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT title, namespace, page_id, revision_id, timestamp, text "
                f"FROM pages WHERE title IN ({placeholders})",
                batch,
            )
            for row in rows:
                page = DumpPage(*row)
                result[normalize_title(page.title)] = page
        return result

    def titles(self) -> set[str]:
        return {normalize_title(row[0]) for row in self.connection.execute("SELECT title FROM pages")}

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "PageCache":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def run_wikisource_page_extraction(
    config: WikisourcePageExtractionConfig,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Extract, fingerprint, and audit the complete inactive checkpoint-4C queue."""

    _validate_config(config)
    started = monotonic()
    resolution = _read_csv(config.resolution_path)
    eligible = [
        row for row in resolution
        if row["checkpoint_4b_decision"] == "eligible_page_level_audit_queue"
    ]
    if len(eligible) != config.expected_eligible_roots:
        raise ValueError(
            f"expected {config.expected_eligible_roots} eligible roots, found {len(eligible)}"
        )
    eligible_by_id = {row["work_root_id"]: row for row in eligible}
    hierarchy = [
        row for row in _read_csv(config.hierarchy_path)
        if row["work_root_id"] in eligible_by_id and row["is_redirect"] != "True"
    ]
    hierarchy_by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in hierarchy:
        hierarchy_by_root[row["work_root_id"]].append(row)
    selected_rows: list[dict[str, str]] = []
    for root_id, rows in hierarchy_by_root.items():
        selected_rows.extend(_select_leaf_rows(rows))

    expected_revisions = {
        normalize_title(row["page_title"]): int(row["latest_revision_id"])
        for row in selected_rows
    }
    main_titles = set(expected_revisions)
    cache_db = config.local_cache_dir / "selected_pages.sqlite3"
    config.local_cache_dir.mkdir(parents=True, exist_ok=True)
    with PageCache(cache_db) as cache:
        cached = cache.titles()
        missing_main = main_titles - cached
        if missing_main:
            _emit(progress, f"dump-pass-1 selected_main_pages={len(missing_main):,}")
            cache.put_many(
                iter_mediawiki_dump(
                    config.dump_path,
                    selected_titles=missing_main,
                    progress_interval=config.progress_interval,
                    progress=progress,
                )
            )
        main_pages = cache.get_many(main_titles)
        required_proofread: set[str] = set()
        for page in main_pages.values():
            try:
                for directive in parse_pages_transclusions(page.text):
                    required_proofread.update(directive.page_titles)
            except ValueError:
                # The root-level audit records this as a section/transclusion
                # error. Discovery must continue so one malformed wrapper does
                # not prevent accounting for the rest of the frozen queue.
                pass
            required_proofread.update(direct_page_transclusions(page.text))
        cached = cache.titles()
        missing_proofread = required_proofread - cached
        discovery_path = config.local_cache_dir / "proofread_discovery.json"
        required_digest = hashlib.sha256(
            "\n".join(sorted(required_proofread)).encode("utf-8")
        ).hexdigest()
        discovery_complete = False
        if discovery_path.is_file():
            discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
            discovery_complete = (
                discovery.get("dump_sha1") == config.expected_dump_sha1
                and discovery.get("required_titles_sha256") == required_digest
            )
        if missing_proofread and not discovery_complete:
            _emit(progress, f"dump-pass-2 selected_proofread_pages={len(missing_proofread):,}")
            cache.put_many(
                iter_mediawiki_dump(
                    config.dump_path,
                    selected_titles=missing_proofread,
                    progress_interval=config.progress_interval,
                    progress=progress,
                )
            )
            matched_after_scan = len(cache.titles() & required_proofread)
            _write_json(
                discovery_path,
                {
                    "dump_sha1": config.expected_dump_sha1,
                    "required_titles_sha256": required_digest,
                    "required_title_count": len(required_proofread),
                    "matched_title_count": matched_after_scan,
                    "missing_title_count": len(required_proofread) - matched_after_scan,
                },
            )
        elif discovery_complete:
            _emit(
                progress,
                f"dump-pass-2 cache-complete required={len(required_proofread):,} "
                f"known_missing={len(missing_proofread):,}",
            )
        proofread_pages = cache.get_many(required_proofread)

    root_text_dir = config.local_cache_dir / "root_texts"
    page_text_dir = config.local_cache_dir / "page_texts"
    root_text_dir.mkdir(parents=True, exist_ok=True)
    page_text_dir.mkdir(parents=True, exist_ok=True)
    boundary_rows: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []
    fingerprints: dict[str, TextFingerprint] = {}
    text_paths: dict[str, Path] = {}
    selected_by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        selected_by_root[row["work_root_id"]].append(row)

    for index, candidate in enumerate(sorted(eligible, key=lambda row: row["root_title"].casefold()), start=1):
        root_id = candidate["work_root_id"]
        page_parts: list[str] = []
        patterns: set[str] = set()
        flags: set[str] = set()
        missing: set[str] = set()
        mismatches: list[str] = []
        required_count = 0
        matched_count = 0
        for hierarchy_row in sorted(selected_by_root.get(root_id, []), key=_hierarchy_sort_key):
            title = normalize_title(hierarchy_row["page_title"])
            page = main_pages.get(title)
            if page is None:
                mismatches.append(f"{title}|missing_dump_page")
                continue
            expected_revision = int(hierarchy_row["latest_revision_id"])
            if page.revision_id != expected_revision:
                mismatches.append(
                    f"{title}|expected={expected_revision}|dump={page.revision_id}"
                )
                continue
            try:
                cleaned, evidence = reconstruct_page(page.text, proofread_pages)
            except ValueError as error:
                cleaned = ""
                evidence = {
                    "pattern": "section_error",
                    "required_titles": [],
                    "missing_titles": [],
                    "quality_flags": [f"section_error:{error}"],
                }
            patterns.add(evidence["pattern"])
            flags.update(evidence["quality_flags"])
            missing.update(evidence["missing_titles"])
            required_count += len(evidence["required_titles"])
            matched_count += len(evidence["required_titles"]) - len(evidence["missing_titles"])
            if cleaned:
                page_parts.append(cleaned)
            page_cache_path = page_text_dir / f"{page.page_id}.txt"
            _atomic_write_text(page_cache_path, cleaned + ("\n" if cleaned else ""))
            boundary_rows.append(
                {
                    "work_root_id": root_id,
                    "root_title": candidate["root_title"],
                    "page_id": page.page_id,
                    "page_title": page.title,
                    "expected_revision_id": expected_revision,
                    "dump_revision_id": page.revision_id,
                    "hierarchy_depth": hierarchy_row["hierarchy_depth"],
                    "boundary_selection": "selected_leaf_no_parent_duplication",
                    "extraction_pattern": evidence["pattern"],
                    "proofread_page_count": len(evidence["required_titles"]),
                    "extracted_character_count": len(cleaned),
                    "extracted_sha256": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
                    "quality_flags": ";".join(evidence["quality_flags"]),
                    "rendered_validation_status": "not_in_bounded_sample",
                    "rendered_validation_containment": "",
                    "activation_status": "local_interim_not_activated",
                }
            )
        text = "\n\n".join(part for part in page_parts if part).strip()
        flags.update(_text_quality_flags(text))
        path = root_text_dir / f"{root_id.replace(':', '_')}.txt"
        _atomic_write_text(path, text + ("\n" if text else ""))
        fingerprint, _hits = fingerprint_text(text)
        fingerprints[root_id] = fingerprint
        text_paths[root_id] = path
        extraction_rows.append(
            _base_extraction_row(
                candidate,
                selected_count=len(selected_by_root.get(root_id, [])),
                matched_main_count=len(selected_by_root.get(root_id, [])) - len(mismatches),
                patterns=patterns,
                required_count=required_count,
                matched_count=matched_count,
                missing=missing,
                mismatches=mismatches,
                text=text,
                fingerprint=fingerprint,
                flags=flags,
                local_path=path.relative_to(config.repo_root),
            )
        )
        if index == 1 or index == len(eligible) or index % 100 == 0:
            _emit_phase(progress, "extract", index, len(eligible), started)

    rows_by_id = {row["work_root_id"]: row for row in extraction_rows}
    internal_pairs = _attach_internal_duplicates(
        rows_by_id, fingerprints, text_paths, threshold=config.near_duplicate_threshold
    )
    references = _load_references(config)
    reference_fingerprints: dict[str, TextFingerprint] = {}
    reference_started = monotonic()
    for reference_index, (reference_id, reference) in enumerate(sorted(references.items()), start=1):
        reference_fingerprints[reference_id] = fingerprint_text(reference.read_text())[0]
        if reference_index == 1 or reference_index == len(references) or reference_index % 100 == 0:
            _emit_phase(progress, "reference-index", reference_index, len(references), reference_started)
    cross_pairs = _attach_cross_duplicates(
        rows_by_id,
        fingerprints,
        text_paths,
        references,
        reference_fingerprints,
        threshold=config.near_duplicate_threshold,
        progress=progress,
    )
    _attach_protected_v6(config, rows_by_id, text_paths, progress=progress)
    _finalize_decisions(extraction_rows)
    review_rows = _build_review_rows(extraction_rows)
    _write_csv(config.extraction_path, EXTRACTION_FIELDS, extraction_rows)
    _write_csv(config.boundaries_path, BOUNDARY_FIELDS, boundary_rows)
    _write_csv(config.review_path, REVIEW_FIELDS, review_rows)
    report = _build_report(
        config,
        eligible=eligible,
        selected_rows=selected_rows,
        required_proofread=required_proofread,
        proofread_pages=proofread_pages,
        extraction_rows=extraction_rows,
        boundary_rows=boundary_rows,
        review_rows=review_rows,
        internal_pairs=internal_pairs,
        cross_pairs=cross_pairs,
    )
    _write_json(config.json_report_path, report)
    _atomic_write_text(config.markdown_report_path, render_markdown(report))
    return report


def apply_rendered_validation(
    *,
    extraction_path: Path,
    boundaries_path: Path,
    review_path: Path,
    json_report_path: Path,
    markdown_report_path: Path,
    results: dict[int, tuple[str, float | None]],
) -> None:
    """Merge a bounded rendered-HTML validation result into public evidence."""

    boundaries = _read_csv(boundaries_path)
    by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in boundaries:
        page_id = int(row["page_id"])
        if page_id in results:
            status, containment = results[page_id]
            row["rendered_validation_status"] = status
            row["rendered_validation_containment"] = (
                f"{containment:.6f}" if containment is not None else ""
            )
            by_root[row["work_root_id"]].append(row)
    rows = _read_csv(extraction_path)
    for row in rows:
        sampled = by_root.get(row["work_root_id"], [])
        if not sampled:
            continue
        containments = [
            float(item["rendered_validation_containment"])
            for item in sampled
            if item["rendered_validation_containment"]
        ]
        statuses = {item["rendered_validation_status"] for item in sampled}
        row["rendered_validation_status"] = (
            "pass" if statuses == {"pass"} else "hold_rendered_validation"
        )
        row["rendered_validation_min_containment"] = (
            f"{min(containments):.6f}" if containments else ""
        )
    _finalize_decisions(rows)
    review_rows = _build_review_rows(rows)
    _write_csv(boundaries_path, BOUNDARY_FIELDS, boundaries)
    _write_csv(extraction_path, EXTRACTION_FIELDS, rows)
    _write_csv(review_path, REVIEW_FIELDS, review_rows)
    report = json.loads(json_report_path.read_text(encoding="utf-8"))
    report["rendered_validation"] = {
        "sampled_page_count": len(results),
        "status_counts": dict(sorted(Counter(status for status, _ in results.values()).items())),
        "minimum_passing_containment": NEAR_DUPLICATE_THRESHOLD,
        "policy": "bounded stratified rendered-HTML validation; failures remain inactive",
    }
    report["decision_counts"] = dict(sorted(Counter(row["checkpoint_4c_decision"] for row in rows).items()))
    report.update(_decision_summary(rows))
    report["review_row_count"] = len(review_rows)
    _write_json(json_report_path, report)
    _atomic_write_text(markdown_report_path, render_markdown(report))


def select_rendered_validation_pages(
    boundaries: list[dict[str, str]],
    extraction_rows: list[dict[str, str]],
    *,
    root_sample_size: int,
) -> list[dict[str, str]]:
    """Select deterministic role/pattern/risk coverage and first/middle/last leaves."""

    if root_sample_size <= 0:
        return []
    rows_by_root = {row["work_root_id"]: row for row in extraction_rows}
    pages_by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in boundaries:
        pages_by_root[row["work_root_id"]].append(row)
    ranked = sorted(
        rows_by_root.values(),
        key=lambda row: (
            -int(row["hierarchy_page_count"]),
            -int(row["required_proofread_page_count"]),
            row["root_title"].casefold(),
        ),
    )
    chosen: list[dict[str, str]] = []
    seen_roots: set[str] = set()
    seen_roles: set[str] = set()
    seen_patterns: set[str] = set()
    for row in ranked:
        role = row["proposed_role"]
        patterns = set(row["extraction_patterns"].split(";"))
        if role not in seen_roles or not patterns.issubset(seen_patterns) or len(chosen) < 5:
            chosen.append(row)
            seen_roots.add(row["work_root_id"])
            seen_roles.add(role)
            seen_patterns.update(patterns)
        if len(chosen) >= root_sample_size:
            break
    for row in ranked:
        if len(chosen) >= root_sample_size:
            break
        if row["work_root_id"] not in seen_roots:
            chosen.append(row)
            seen_roots.add(row["work_root_id"])
    selected_pages: list[dict[str, str]] = []
    for root in chosen:
        pages = sorted(pages_by_root[root["work_root_id"]], key=lambda row: int(row["page_id"]))
        indices = sorted({0, len(pages) // 2, len(pages) - 1}) if pages else []
        selected_pages.extend(pages[index] for index in indices)
    return selected_pages


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("_", " ").strip())


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Italian Wikisource Page Extraction Audit",
        "",
        "## Result",
        "",
        (
            f"Checkpoint 4C accounts for {report['eligible_root_count']:,} scan-anchored "
            "roots without activating their text."
        ),
        "",
        f"- Selected hierarchy leaves: {report['selected_leaf_page_count']:,}.",
        f"- Required `Pagina:` transcriptions: {report['required_proofread_page_count']:,}.",
        f"- Matched `Pagina:` transcriptions: {report['matched_proofread_page_count']:,}.",
        f"- Local extracted characters: {report['extracted_character_count']:,}.",
        f"- Passing inactive characters: {report['eligible_inactive_character_count']:,}.",
        f"- Review rows: {report['review_row_count']:,}.",
        "",
        "## Decisions",
        "",
        "| Decision | Roots |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{decision}` | {count:,} |"
        for decision, count in report["decision_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Duplicate And Leakage Probe",
            "",
            f"- Internal normalized exact-duplicate groups: {report['internal_exact_duplicate_group_count']:,}.",
            f"- Roots in internal exact-duplicate groups: {report['internal_exact_duplicate_root_count']:,}.",
            f"- Internal threshold pairs: {report['internal_near_duplicate_pair_count']:,}.",
            f"- Cross-corpus threshold pairs: {report['cross_corpus_near_duplicate_pair_count']:,}.",
            f"- Protected V6 overlap roots: {report['protected_v6_overlap_root_count']:,}.",
            "",
            "The comparisons cover BibIt, both Gutenberg probe pools, the resolved "
            "Gutenberg build, existing project corpora, and protected V6 validation/test sonnets.",
            "",
            "## Boundary",
            "",
            "All extracted text remains in ignored local interim storage. No Wikisource "
            "text is activated, no V7 split or mixture is created, no cache is deleted, "
            "and no GPU work occurs in this checkpoint.",
        ]
    )
    validation = report.get("rendered_validation")
    if validation:
        lines.extend(
            [
                "",
                "## Rendered Validation",
                "",
                f"Sampled pages: {validation['sampled_page_count']:,}.",
                "",
                "| Status | Pages |",
                "| --- | ---: |",
            ]
        )
        lines.extend(
            f"| `{status}` | {count:,} |"
            for status, count in validation["status_counts"].items()
        )
    return "\n".join(lines).rstrip() + "\n"


def _base_extraction_row(
    candidate: dict[str, str],
    *,
    selected_count: int,
    matched_main_count: int,
    patterns: set[str],
    required_count: int,
    matched_count: int,
    missing: set[str],
    mismatches: list[str],
    text: str,
    fingerprint: TextFingerprint,
    flags: set[str],
    local_path: Path,
) -> dict[str, Any]:
    row = {field: "" for field in EXTRACTION_FIELDS}
    row.update(
        {
            "work_root_id": candidate["work_root_id"],
            "root_title": candidate["root_title"],
            "landing_page_url": candidate["landing_page_url"],
            "proposed_role": candidate["proposed_role"],
            "period_bucket": candidate["period_bucket"],
            "author_evidence": candidate["author_evidence"],
            "hierarchy_page_count": candidate["hierarchy_page_count"],
            "selected_leaf_page_count": selected_count,
            "matched_main_page_count": matched_main_count,
            "direct_scan_title": candidate["direct_scan_titles"],
            "extraction_patterns": ";".join(sorted(patterns)),
            "required_proofread_page_count": required_count,
            "matched_proofread_page_count": matched_count,
            "missing_proofread_pages": ";".join(sorted(missing)),
            "revision_mismatches": ";".join(mismatches),
            "extracted_character_count": len(text),
            "extracted_word_count": len(_WORD.findall(text)),
            "nonempty_line_count": sum(bool(line.strip()) for line in text.splitlines()),
            "normalized_word_sha256": fingerprint.normalized_word_sha256,
            "quality_flags": ";".join(sorted(flags)),
            "unresolved_markup": ";".join(
                flag for flag in sorted(flags) if flag.startswith("unresolved_")
            ),
            "rendered_validation_status": "not_in_bounded_sample",
            "local_text_cache_path": local_path.as_posix(),
            "activation_status": "local_interim_not_activated",
        }
    )
    return row


def _finalize_decisions(rows: list[dict[str, Any]]) -> None:
    overlap_fields = (
        "internal_exact_duplicate_ids",
        "internal_near_duplicate_metrics",
        "bibit_overlap_metrics",
        "gutenberg_previous_pool_overlap_metrics",
        "gutenberg_pass_1b_overlap_metrics",
        "gutenberg_resolved_overlap_metrics",
        "existing_project_corpus_overlap_metrics",
        "protected_v6_overlap_metrics",
    )
    for row in rows:
        if row["revision_mismatches"]:
            decision, reason = "hold_revision_mismatch", "Pinned hierarchy revision did not match the dump."
        elif row["missing_proofread_pages"]:
            decision, reason = "hold_missing_transcription", "One or more requested Pagina records are missing."
        elif int(row["extracted_character_count"]) == 0:
            decision, reason = "hold_empty_extraction", "No primary text remained after conservative cleaning."
        elif row["unresolved_markup"]:
            decision, reason = "hold_unresolved_markup", "Unresolved markup may contaminate primary text."
        elif row["protected_v6_overlap_metrics"]:
            decision, reason = "hold_protected_v6_overlap", "Protected V6 validation/test material overlaps this root."
        elif any(row[field] for field in overlap_fields):
            decision, reason = "hold_duplicate_review", "A threshold internal or cross-corpus overlap requires canonical review."
        elif row["rendered_validation_status"] == "hold_rendered_validation":
            decision, reason = "hold_rendered_validation", "Local extraction disagrees with bounded rendered HTML validation."
        elif row["quality_flags"]:
            decision, reason = "hold_quality_or_editorial_review", "Quality or editorial signals require bounded review."
        else:
            decision, reason = "eligible_inactive_pending_processed_build", "Extraction and overlap gates passed; activation remains a later checkpoint."
        row["checkpoint_4c_decision"] = decision
        row["review_reason"] = reason


def _build_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    overlap_fields = [field for field in EXTRACTION_FIELDS if field.endswith("overlap_metrics")]
    for row in rows:
        if row["checkpoint_4c_decision"] == "eligible_inactive_pending_processed_build":
            continue
        overlaps = [f"{field}={row[field]}" for field in overlap_fields if row[field]]
        result.append(
            {
                "review_id": f"itws4c:{row['work_root_id'].split(':')[-1]}",
                "work_root_id": row["work_root_id"],
                "root_title": row["root_title"],
                "proposed_role": row["proposed_role"],
                "checkpoint_4c_decision": row["checkpoint_4c_decision"],
                "quality_flags": row["quality_flags"],
                "missing_proofread_pages": row["missing_proofread_pages"],
                "overlap_summary": " | ".join(overlaps),
                "required_action": row["review_reason"],
                "review_status": "pending_bounded_review",
                "activation_status": "inactive",
            }
        )
    return result


def _attach_internal_duplicates(
    rows: dict[str, dict[str, Any]],
    fingerprints: dict[str, TextFingerprint],
    text_paths: dict[str, Path],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    exact: dict[str, list[str]] = defaultdict(list)
    for root_id, fingerprint in fingerprints.items():
        if fingerprint.word_count:
            exact[fingerprint.normalized_word_sha256].append(root_id)
    exact_pairs: set[tuple[str, str]] = set()
    for ids in exact.values():
        if len(ids) < 2:
            continue
        for root_id in ids:
            rows[root_id]["internal_exact_duplicate_ids"] = ";".join(sorted(value for value in ids if value != root_id))
        exact_pairs.update(tuple(sorted(pair)) for pair in combinations(ids, 2))
    pairs = []
    for left_id, right_id in sorted(_discover_pairs(fingerprints) - exact_pairs):
        metric = measure_word_shingle_containment(
            text_paths[left_id].read_text(encoding="utf-8"),
            text_paths[right_id].read_text(encoding="utf-8"),
        )
        if metric["containment"] >= threshold:
            value = f"{right_id}|containment={metric['containment']:.6f}"
            rows[left_id]["internal_near_duplicate_metrics"] = _append(rows[left_id]["internal_near_duplicate_metrics"], value)
            value = f"{left_id}|containment={metric['containment']:.6f}"
            rows[right_id]["internal_near_duplicate_metrics"] = _append(rows[right_id]["internal_near_duplicate_metrics"], value)
            pairs.append({"left_id": left_id, "right_id": right_id, **metric})
    return pairs


def _attach_cross_duplicates(
    rows: dict[str, dict[str, Any]],
    fingerprints: dict[str, TextFingerprint],
    text_paths: dict[str, Path],
    references: dict[str, TextReference],
    reference_fingerprints: dict[str, TextFingerprint],
    *,
    threshold: float,
    progress: Progress | None,
) -> list[dict[str, Any]]:
    candidates = _discover_cross_pairs(fingerprints, reference_fingerprints)
    exact_map: dict[str, list[str]] = defaultdict(list)
    for reference_id, fingerprint in reference_fingerprints.items():
        exact_map[fingerprint.normalized_word_sha256].append(reference_id)
    for root_id, fingerprint in fingerprints.items():
        for reference_id in exact_map.get(fingerprint.normalized_word_sha256, []):
            candidates.add((root_id, reference_id))
    field_by_kind = {
        "bibit": "bibit_overlap_metrics",
        "gutenberg_previous_pool": "gutenberg_previous_pool_overlap_metrics",
        "gutenberg_pass_1b": "gutenberg_pass_1b_overlap_metrics",
        "gutenberg_resolved": "gutenberg_resolved_overlap_metrics",
        "existing_project_corpus": "existing_project_corpus_overlap_metrics",
    }
    pairs = []
    started = monotonic()
    for index, (root_id, reference_id) in enumerate(sorted(candidates), start=1):
        metric = measure_word_shingle_containment(
            text_paths[root_id].read_text(encoding="utf-8"), references[reference_id].read_text()
        )
        if metric["containment"] >= threshold:
            field = field_by_kind[references[reference_id].source_kind]
            value = (
                f"{reference_id}|candidate={metric['left_containment']:.6f}|"
                f"reference={metric['right_containment']:.6f}"
            )
            rows[root_id][field] = _append(rows[root_id][field], value)
            pairs.append({"work_root_id": root_id, "reference_id": reference_id, **metric})
        if progress and (index == len(candidates) or index % 100 == 0):
            _emit_phase(progress, "cross-overlap", index, len(candidates), started)
    return pairs


def _attach_protected_v6(
    config: WikisourcePageExtractionConfig,
    rows: dict[str, dict[str, Any]],
    text_paths: dict[str, Path],
    *,
    progress: Progress | None,
) -> None:
    protected: list[tuple[str, str]] = []
    for row in _read_csv(config.sonnet_manifest_path):
        if row["split_expanded_with_petrarch"] in {"validation", "test"}:
            path = config.repo_root / row["clean_text_path"]
            if path.is_file():
                protected.append((row["poem_id"], path.read_text(encoding="utf-8")))
    watch: dict[int, list[str]] = defaultdict(list)
    denominators: dict[str, int] = {}
    for poem_id, text in protected:
        hashes = set(_rolling_shingle_hashes(_normalized_words(text)))
        if not hashes:
            continue
        denominators[poem_id] = len(hashes)
        for value in hashes:
            watch[value].append(poem_id)
    frozen_watch = {value: tuple(ids) for value, ids in watch.items()}
    started = monotonic()
    for index, (root_id, path) in enumerate(sorted(text_paths.items()), start=1):
        _fingerprint, hits = fingerprint_text(
            path.read_text(encoding="utf-8"), watched_shingles=frozen_watch
        )
        for poem_id, values in hits.items():
            containment = len(values) / denominators[poem_id]
            if containment < NEAR_DUPLICATE_THRESHOLD:
                continue
            value = f"{poem_id}|protected_containment={containment:.6f}"
            rows[root_id]["protected_v6_overlap_metrics"] = _append(
                rows[root_id]["protected_v6_overlap_metrics"], value
            )
        if index == 1 or index == len(text_paths) or index % 100 == 0:
            _emit_phase(progress, "protected-v6", index, len(text_paths), started)


def _load_references(config: WikisourcePageExtractionConfig) -> dict[str, TextReference]:
    references: dict[str, TextReference] = {}
    for row in _read_csv(config.bibit_record_manifest_path):
        if row["artifact_status"] == "text_materialized" and row["shard_path"]:
            references[f"bibit:{row['object_id']}"] = TextReference(
                f"bibit:{row['object_id']}", "bibit", config.repo_root / row["shard_path"],
                int(row["byte_start"]), int(row["byte_end"]),
            )
    for row in _read_csv(config.broader_sources_manifest_path):
        relative = row.get("expected_clean_text_path", "")
        path = config.repo_root / relative if relative else Path("/")
        if relative and path.is_file():
            references[f"current:{row['source_id']}"] = TextReference(
                f"current:{row['source_id']}", "existing_project_corpus", path
            )
    _add_gutenberg_probe_references(
        references, config.gutenberg_previous_probe_path, config.gutenberg_previous_cache_dir,
        kind="gutenberg_previous_pool", prefix="gutenberg_previous",
    )
    _add_gutenberg_probe_references(
        references, config.gutenberg_pass_1b_probe_path, config.gutenberg_pass_1b_cache_dir,
        kind="gutenberg_pass_1b", prefix="gutenberg_pass1b",
    )
    for row in _read_csv(config.gutenberg_resolved_manifest_path):
        if row["artifact_status"] == "text_materialized_pending_v7" and row["shard_path"]:
            references[f"gutenberg_resolved:pg{row['ebook_id']}"] = TextReference(
                f"gutenberg_resolved:pg{row['ebook_id']}", "gutenberg_resolved",
                config.repo_root / row["shard_path"], int(row["byte_start"]), int(row["byte_end"]),
            )
    return references


def _add_gutenberg_probe_references(
    references: dict[str, TextReference],
    probe_path: Path,
    cache_dir: Path,
    *,
    kind: str,
    prefix: str,
) -> None:
    for row in _read_csv(probe_path):
        path = cache_dir / f"pg{row['ebook_id']}.txt"
        if path.is_file():
            reference_id = f"{prefix}:pg{row['ebook_id']}"
            references[reference_id] = TextReference(
                reference_id, kind, path, cleaning="gutenberg_boilerplate"
            )


def _discover_pairs(fingerprints: dict[str, TextFingerprint]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for attribute in ("anchors", "sketch"):
        postings: dict[int, list[str]] = defaultdict(list)
        for document_id, fingerprint in fingerprints.items():
            for value in getattr(fingerprint, attribute):
                postings[value].append(document_id)
        collisions: Counter[tuple[str, str]] = Counter()
        for ids in postings.values():
            if 1 < len(ids) <= 40:
                collisions.update(tuple(sorted(pair)) for pair in combinations(ids, 2))
        for pair, count in collisions.items():
            denominator = min(
                len(getattr(fingerprints[pair[0]], attribute)),
                len(getattr(fingerprints[pair[1]], attribute)),
            )
            if denominator and count >= 2 and count / denominator >= 0.4:
                pairs.add(pair)
    return pairs


def _discover_cross_pairs(
    left: dict[str, TextFingerprint], right: dict[str, TextFingerprint]
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for attribute in ("anchors", "sketch"):
        postings: dict[int, list[str]] = defaultdict(list)
        for reference_id, fingerprint in right.items():
            for value in getattr(fingerprint, attribute):
                postings[value].append(reference_id)
        collisions: Counter[tuple[str, str]] = Counter()
        for candidate_id, fingerprint in left.items():
            for value in getattr(fingerprint, attribute):
                refs = postings.get(value, [])
                if len(refs) <= 40:
                    collisions.update((candidate_id, reference_id) for reference_id in refs)
        for pair, count in collisions.items():
            denominator = min(
                len(getattr(left[pair[0]], attribute)),
                len(getattr(right[pair[1]], attribute)),
            )
            if denominator and count >= 2 and count / denominator >= 0.4:
                pairs.add(pair)
    return pairs


def _build_report(
    config: WikisourcePageExtractionConfig,
    *,
    eligible: list[dict[str, str]],
    selected_rows: list[dict[str, str]],
    required_proofread: set[str],
    proofread_pages: dict[str, DumpPage],
    extraction_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    internal_pairs: list[dict[str, Any]],
    cross_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "checkpoint": "4C",
        "dump": {
            "date": DUMP_DATE,
            "filename": config.dump_path.name,
            "bytes": config.dump_path.stat().st_size,
            "sha1": config.expected_dump_sha1,
        },
        "eligible_root_count": len(eligible),
        "role_counts": dict(sorted(Counter(row["proposed_role"] for row in eligible).items())),
        "selected_leaf_page_count": len(selected_rows),
        "boundary_row_count": len(boundary_rows),
        "required_proofread_page_count": len(required_proofread),
        "matched_proofread_page_count": len(proofread_pages),
        "extracted_character_count": sum(int(row["extracted_character_count"]) for row in extraction_rows),
        "decision_counts": dict(sorted(Counter(row["checkpoint_4c_decision"] for row in extraction_rows).items())),
        "quality_flag_counts": dict(sorted(Counter(flag for row in extraction_rows for flag in str(row["quality_flags"]).split(";") if flag).items())),
        "review_row_count": len(review_rows),
        "internal_exact_duplicate_group_count": sum(
            count > 1
            for count in Counter(
                row["normalized_word_sha256"]
                for row in extraction_rows
                if int(row["extracted_word_count"]) > 0
            ).values()
        ),
        "internal_exact_duplicate_root_count": sum(
            bool(row["internal_exact_duplicate_ids"]) for row in extraction_rows
        ),
        "internal_near_duplicate_pair_count": len(internal_pairs),
        "cross_corpus_near_duplicate_pair_count": len(cross_pairs),
        "cross_corpus_pair_counts": dict(sorted(Counter(item["reference_id"].split(":", 1)[0] for item in cross_pairs).items())),
        "protected_v6_overlap_root_count": sum(bool(row["protected_v6_overlap_metrics"]) for row in extraction_rows),
        "outputs": {
            "extraction": config.extraction_path.relative_to(config.repo_root).as_posix(),
            "boundaries": config.boundaries_path.relative_to(config.repo_root).as_posix(),
            "review": config.review_path.relative_to(config.repo_root).as_posix(),
        },
        "policy": {
            "source_spelling_and_punctuation_preserved": True,
            "conditioned_and_held_roots_excluded": True,
            "local_text_only": True,
            "text_activated": False,
            "v7_created": False,
            "cache_deleted": False,
            "gpu_work_started": False,
        },
    }
    report.update(_decision_summary(extraction_rows))
    return report


def _decision_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row["checkpoint_4c_decision"] == "eligible_inactive_pending_processed_build"
    ]
    return {
        "decision_counts": dict(
            sorted(Counter(row["checkpoint_4c_decision"] for row in rows).items())
        ),
        "eligible_inactive_root_count": len(eligible),
        "eligible_inactive_character_count": sum(
            int(row["extracted_character_count"]) for row in eligible
        ),
        "eligible_inactive_role_counts": dict(
            sorted(Counter(row["proposed_role"] for row in eligible).items())
        ),
        "eligible_inactive_characters_by_role": {
            role: sum(
                int(row["extracted_character_count"])
                for row in eligible
                if row["proposed_role"] == role
            )
            for role in sorted({row["proposed_role"] for row in eligible})
        },
    }


def _text_quality_flags(text: str) -> set[str]:
    flags: set[str] = set()
    if not text:
        return {"empty_extraction"}
    if _EDITORIAL.search(text):
        flags.add("editorial_marker")
    if _DIALECT.search(text):
        flags.add("language_variety_marker")
    characters = [char for char in text if not char.isspace()]
    if characters:
        alphabetic = sum(char.isalpha() for char in characters) / len(characters)
        digits = sum(char.isdigit() for char in characters) / len(characters)
        if alphabetic < 0.55:
            flags.add("low_alphabetic_ratio")
        if digits > 0.08:
            flags.add("high_digit_ratio")
    if len(text) < 100:
        flags.add("very_short_text")
    return flags


def _replace_templates(value: str) -> tuple[str, set[str]]:
    flags: set[str] = set()
    for _ in range(20):
        matches = list(re.finditer(r"\{\{([^{}]*)\}\}", value, flags=re.S))
        if not matches:
            break
        chunks = []
        last = 0
        for match in matches:
            chunks.append(value[last : match.start()])
            parts = [part.strip() for part in _split_template_parts(match.group(1))]
            name = parts[0].casefold().replace("_", " ") if parts else ""
            name = re.sub(r"\s+", " ", name).strip()
            positional = [part for part in parts[1:] if "=" not in part]
            if name in _DISCARD_TEMPLATES:
                replacement = ""
            elif name.startswith("capolettera"):
                replacement = _capolettera_text(positional[0] if positional else "")
            elif name == "testoassente":
                flags.add("transcription_gap_template")
                replacement = positional[-1] if positional else ""
            elif name in _KEEP_ARGUMENT_TEMPLATES:
                replacement = positional[-1] if positional else ""
            elif name in {"§", "§§"}:
                replacement = name
            else:
                flags.add(f"unresolved_template:{name or 'empty'}")
                replacement = ""
            chunks.append(replacement)
            last = match.end()
        chunks.append(value[last:])
        value = "".join(chunks)
    return value, flags


def _split_template_parts(value: str) -> list[str]:
    """Split template pipes while preserving pipes inside wiki links."""

    parts: list[str] = []
    start = 0
    link_depth = 0
    index = 0
    while index < len(value):
        if value.startswith("[[", index):
            link_depth += 1
            index += 2
            continue
        if value.startswith("]]", index) and link_depth:
            link_depth -= 1
            index += 2
            continue
        if value[index] == "|" and link_depth == 0:
            parts.append(value[start:index])
            start = index + 1
        index += 1
    parts.append(value[start:])
    return parts


def _capolettera_text(value: str) -> str:
    if re.match(r"\[\[(?:File|Immagine):", value, flags=re.I):
        inner = value.removeprefix("[[").removesuffix("]]" )
        parts = inner.split("|")
        return parts[-1].strip() if len(parts) > 1 else ""
    return value


def _page_labels(attrs: dict[str, str]) -> list[str]:
    include = attrs.get("include", "").strip()
    if include:
        labels = [item.strip() for item in re.split(r"[,;]", include) if item.strip()]
    else:
        start = attrs.get("from", "").strip()
        end = attrs.get("to", start).strip()
        if not start:
            raise ValueError("<pages> transclusion has no from/include attribute")
        if start.isdigit() and end.isdigit():
            first, last = int(start), int(end)
            if last < first or last - first > 10_000:
                raise ValueError(f"invalid <pages> range: {start}..{end}")
            step = int(attrs.get("step", "1"))
            if step <= 0:
                raise ValueError("<pages> step must be positive")
            labels = [str(value) for value in range(first, last + 1, step)]
        elif start == end:
            labels = [start]
        else:
            raise ValueError(f"non-numeric multi-page range is unsupported: {start}..{end}")
    excluded = {item.strip() for item in re.split(r"[,;]", attrs.get("exclude", "")) if item.strip()}
    return [label for label in labels if label not in excluded]


def _parse_attributes(value: str) -> dict[str, str]:
    return {
        match.group("name").casefold(): (
            match.group("double") if match.group("double") is not None
            else match.group("single") if match.group("single") is not None
            else match.group("bare") or ""
        )
        for match in _ATTRIBUTE.finditer(value)
    }


def _find_section(pattern: re.Pattern[str], text: str, name: str) -> re.Match[str] | None:
    for match in pattern.finditer(text):
        value = match.group("d") or match.group("s") or match.group("b") or ""
        if value == name:
            return match
    return None


def _select_leaf_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    titles = {normalize_title(row["page_title"]) for row in rows}
    return [
        row for row in rows
        if not any(other.startswith(normalize_title(row["page_title"]) + "/") for other in titles)
    ]


def _hierarchy_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    parts = normalize_title(row["relative_title"]).split("/")
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold()) for part in parts)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element:
    for child in element:
        if _local_name(child.tag) == name:
            return child
    raise ValueError(f"MediaWiki XML page omitted {name}")


def _child_text(element: ET.Element, name: str) -> str:
    child = _child(element, name)
    return child.text or ""


def _batches(values: list[str], size: int) -> Iterator[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _append(current: str, value: str) -> str:
    return ";".join(item for item in (current, value) if item)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _emit(progress: Progress | None, message: str) -> None:
    if progress:
        progress(message)


def _emit_phase(progress: Progress | None, phase: str, completed: int, total: int, started: float) -> None:
    elapsed = monotonic() - started
    rate = completed / elapsed if elapsed else 0.0
    remaining = (total - completed) / rate if rate else 0.0
    _emit(progress, f"{phase} completed={completed:,}/{total:,} percent={completed / max(1, total):.1%} elapsed={elapsed:.1f}s eta={remaining:.1f}s")


def _validate_config(config: WikisourcePageExtractionConfig) -> None:
    if not config.dump_path.is_file():
        raise FileNotFoundError(config.dump_path)
    if sha1_file(config.dump_path) != config.expected_dump_sha1:
        raise ValueError(f"Wikisource dump SHA-1 mismatch: {config.dump_path}")
    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    if not 0 < config.near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
