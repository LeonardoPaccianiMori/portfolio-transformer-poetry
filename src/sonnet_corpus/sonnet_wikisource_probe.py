"""Audit Italian Wikisource sonnet collections without changing the corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from .cleaning import clean_poem_text, count_poem_lines
from .italian_wikisource import (
    FetchedItalianWikisourcePageCollection,
    fetch_italian_wikisource_page_collection,
    fetch_italian_wikisource_two_level_page_collection,
)
from .wikisource import extract_poem_text, url_from_title


FetchPageCollection = Callable[..., FetchedItalianWikisourcePageCollection]


@dataclass(frozen=True)
class SonnetWikisourceSource:
    """One source row approved for an audit-only sonnet probe."""

    source_id: str
    title: str
    author: str
    source_archive: str
    landing_page_url: str
    language_variety: str
    role: str
    status: str
    license_or_reuse_status: str
    attribution_required: str


@dataclass(frozen=True)
class SonnetCollectionExpectation:
    """Stable collection identity known before an initial page-level audit."""

    root_page_title: str
    expected_first_subpage: str = ""
    expected_last_subpage: str = ""
    explicit_page_titles: tuple[str, ...] = ()
    edition_page_title_suffix: str = ""
    included_subpage_prefixes: tuple[str, ...] = ()
    excluded_subpage_prefixes: tuple[str, ...] = ()
    direct_root_text_links: bool = False
    index_page_titles: tuple[str, ...] = ()
    two_level_leaf_link_mode: str = "nested_subpages"
    retain_text_samples: bool = True
    audit_status: str = "audit_then_include"


SONNET_COLLECTION_EXPECTATIONS = {
    "ws_alfieri_rime_1912": SonnetCollectionExpectation(
        root_page_title="Rime varie (Alfieri, 1912)",
    ),
    "ws_foscolo_sonetti": SonnetCollectionExpectation(
        root_page_title="Opera:Sonetti (Foscolo)",
        explicit_page_titles=(
            "Opera:Alla Sera",
            "Opera:Non son chi fui; perì di noi gran parte",
            "Opera:Te nudrice alle Muse, ospite e Dea",
            "Opera:Perchè taccia il rumor di mia catena",
            "Opera:Così gl'interi giorni in lungo, incerto",
            "Opera:Meritamente, però ch'io potei",
            "Opera:Solcata ho fronte, occhi incavati intenti",
            "Opera:E tu ne' carmi avrai perenne vita",
            "Opera:A Zacinto",
            "Opera:In morte del fratello Giovanni",
            "Opera:Alla Musa (Foscolo)",
            "Opera:Che stai? già il secol l'orma ultima lascia",
        ),
        edition_page_title_suffix="1835)",
    ),
    "ws_varchi_infermita": SonnetCollectionExpectation(
        root_page_title="Sonetti per la infermità, e guarigione di Cosimo I dei Medici",
        expected_first_subpage=(
            "Sonetti per la infermità, e guarigione di Cosimo I dei Medici/Sonetto I"
        ),
        expected_last_subpage=(
            "Sonetti per la infermità, e guarigione di Cosimo I dei Medici/Sonetto XXXIII"
        ),
        excluded_subpage_prefixes=(
            "Sonetti per la infermità, e guarigione di Cosimo I dei Medici/Avviso dell'editore",
            "Sonetti per la infermità, e guarigione di Cosimo I dei Medici/Dedica",
        ),
    ),
    "ws_andreini_rime_1601": SonnetCollectionExpectation(
        root_page_title="Rime (Andreini)",
        included_subpage_prefixes=(
            "Rime (Andreini)/Sonetto",
            "Rime (Andreini)/Sonetti",
        ),
    ),
    "ws_colonna_rime_1760": SonnetCollectionExpectation(
        root_page_title="Rime (Vittoria Colonna)",
        included_subpage_prefixes=("Rime (Vittoria Colonna)/Sonetto",),
    ),
    "ws_stampa_rime_1913": SonnetCollectionExpectation(
        root_page_title="Rime (Stampa)",
    ),
    "ws_ariosto_rime_varie_1857": SonnetCollectionExpectation(
        root_page_title="Opere minori (Ariosto)/Rime varie",
        included_subpage_prefixes=("Opere minori (Ariosto)/Rime varie/Sonetto",),
    ),
    "ws_sannazaro_rime_disperse": SonnetCollectionExpectation(
        root_page_title="Rime disperse",
        direct_root_text_links=True,
    ),
    "ws_belli_sonetti_romaneschi": SonnetCollectionExpectation(
        root_page_title="Sonetti romaneschi",
        index_page_titles=(
            "Sonetti romaneschi/Sonetti apocrifi",
            "Sonetti romaneschi/Sonetti dal 1818 al 1829",
            "Sonetti romaneschi/Sonetti dal 1828 al 1847",
            "Sonetti romaneschi/Sonetti del 1830",
            "Sonetti romaneschi/Sonetti del 1831",
            "Sonetti romaneschi/Sonetti del 1832",
            "Sonetti romaneschi/Sonetti del 1833",
            "Sonetti romaneschi/Sonetti del 1834",
            "Sonetti romaneschi/Sonetti del 1835",
            "Sonetti romaneschi/Sonetti del 1836",
            "Sonetti romaneschi/Sonetti del 1837",
            "Sonetti romaneschi/Sonetti del 1838",
            "Sonetti romaneschi/Sonetti del 1839-1942",
            "Sonetti romaneschi/Sonetti del 1843",
            "Sonetti romaneschi/Sonetti del 1844",
            "Sonetti romaneschi/Sonetti del 1845",
            "Sonetti romaneschi/Sonetti del 1846",
            "Sonetti romaneschi/Sonetti del 1847 e 1849",
            "Sonetti romaneschi/Sonetti italiani",
            "Sonetti romaneschi/Sonetti senza data I",
            "Sonetti romaneschi/Sonetti senza data II",
        ),
        two_level_leaf_link_mode="direct_text_links",
        audit_status="audit_only_auxiliary",
    ),
    "ws_aretino_sonetti_lussuriosi_1792": SonnetCollectionExpectation(
        root_page_title="Sonetti lussuriosi (edizione 1792)",
        expected_first_subpage="Sonetti lussuriosi (edizione 1792)/I",
        expected_last_subpage="Sonetti lussuriosi (edizione 1792)/XXVI",
        retain_text_samples=False,
        audit_status="audit_only_explicit_content",
    ),
}


def probe_sonnet_wikisource_source(
    *,
    source_manifest_path: Path,
    active_poems_manifest_path: Path,
    repo_root: Path,
    source_id: str,
    report_path: Path,
    request_delay: float = 1.0,
    fetch_collection: FetchPageCollection = fetch_italian_wikisource_page_collection,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Write a local, revision-pinned inspection report for one collection.

    The report contains page metadata, line-count decisions, exact duplicate
    identifiers, and bounded samples. It deliberately excludes full raw and
    cleaned text, so this audit cannot be mistaken for corpus activation.
    """

    source = select_sonnet_wikisource_source(read_sonnet_source_manifest(source_manifest_path), source_id)
    expectation = SONNET_COLLECTION_EXPECTATIONS.get(source.source_id)
    if expectation is None:
        raise ValueError(f"no sonnet collection expectation for source: {source.source_id}")

    started_at = utc_now()
    _write_progress(progress, f"probing source: {source.source_id}")
    active_texts = read_active_poem_texts(active_poems_manifest_path, repo_root)
    active_poem_count = sum(len(poem_ids) for poem_ids in active_texts.values())
    _write_progress(
        progress,
        "loaded "
        f"{active_poem_count} active poems across {len(active_texts)} "
        "exact-text fingerprints for duplicate checks",
    )
    if expectation.index_page_titles:
        collection = fetch_italian_wikisource_two_level_page_collection(
            source.landing_page_url,
            expected_title=expectation.root_page_title,
            index_page_titles=list(expectation.index_page_titles),
            leaf_link_mode=expectation.two_level_leaf_link_mode,
            request_delay=request_delay,
            progress=progress,
        )
    else:
        collection = fetch_collection(
            source.landing_page_url,
            expected_title=expectation.root_page_title,
            expected_first_subpage=expectation.expected_first_subpage,
            expected_last_subpage=expectation.expected_last_subpage,
            explicit_page_titles=list(expectation.explicit_page_titles) or None,
            edition_page_title_suffix=expectation.edition_page_title_suffix or None,
            included_subpage_prefixes=expectation.included_subpage_prefixes,
            excluded_subpage_prefixes=expectation.excluded_subpage_prefixes,
            direct_text_links=expectation.direct_root_text_links,
            request_delay=request_delay,
            progress=progress,
        )

    candidates: list[dict[str, object]] = []
    for page in collection.pages:
        segments = extract_sonnet_candidate_segments(
            source_id=source.source_id,
            page_title=page.revision.title,
            html=page.html,
        )
        for segment in segments:
            cleaned_text = clean_poem_text(segment.raw_text)
            duplicate_ids = active_texts.get(
                normalize_poem_for_duplicate_check(cleaned_text), []
            )
            line_count_clean = count_poem_lines(cleaned_text)
            status = candidate_status(
                raw_text=segment.raw_text,
                line_count_clean=line_count_clean,
                duplicate_ids=duplicate_ids,
            )
            candidate: dict[str, object] = {
                "page_title": page.revision.title,
                "page_url": url_from_title(page.revision.title),
                "revision_id": page.revision.revision_id,
                "revision_timestamp": page.revision.revision_timestamp,
                "segment_index": segment.index,
                "segment_label": segment.label,
                "raw_character_count": len(segment.raw_text),
                "cleaned_character_count": len(cleaned_text.strip()),
                "line_count_raw": count_poem_lines(segment.raw_text),
                "line_count_clean": line_count_clean,
                "status": status,
                "exact_active_duplicate_poem_ids": duplicate_ids,
                "cleaned_text_sha256": hashlib.sha256(
                    cleaned_text.encode("utf-8")
                ).hexdigest(),
            }
            if expectation.retain_text_samples:
                candidate["first_characters"] = bounded_sample(
                    cleaned_text, from_end=False
                )
                candidate["last_characters"] = bounded_sample(cleaned_text, from_end=True)
            if page.source_record_revision is not None:
                candidate["source_record"] = {
                    "page_title": page.source_record_revision.title,
                    "page_url": url_from_title(page.source_record_revision.title),
                    "revision_id": page.source_record_revision.revision_id,
                    "revision_timestamp": page.source_record_revision.revision_timestamp,
                }
            candidates.append(candidate)

    status_counts = Counter(str(candidate["status"]) for candidate in candidates)
    report = {
        "started_at_utc": started_at,
        "source_manifest_path": portable_path(source_manifest_path, repo_root),
        "active_poems_manifest_path": portable_path(active_poems_manifest_path, repo_root),
        "source": asdict(source),
        "activation_status": source.status,
        "edition_page_title_suffix": expectation.edition_page_title_suffix or None,
        "root_revision": asdict(collection.root_revision),
        "index_revisions": [asdict(revision) for revision in collection.index_revisions],
        "page_count": len(collection.pages),
        "candidate_count": len(candidates),
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "candidates": candidates,
        "finished_at_utc": utc_now(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_progress(progress, f"wrote inspection report: {report_path}")
    return report


@dataclass(frozen=True)
class SonnetCandidateSegment:
    """One candidate poem extracted from an audited Wikisource page."""

    index: int
    label: str
    raw_text: str


def extract_sonnet_candidate_segments(
    *, source_id: str, page_title: str, html: str
) -> list[SonnetCandidateSegment]:
    """Return page text, splitting only Andreini pages with two printed sonnets."""

    if (
        source_id == "ws_andreini_rime_1601"
        and is_andreini_paired_sonnet_page(page_title)
    ):
        return extract_andreini_paired_sonnet_segments(html)
    return [
        SonnetCandidateSegment(index=1, label="", raw_text=extract_poem_text(html))
    ]


def is_andreini_paired_sonnet_page(page_title: str) -> bool:
    """Identify paired Andreini index pages without guessing text boundaries."""

    return re.search(r"/Sonetti [IVXLCDM]+-[IVXLCDM]+$", page_title) is not None


def extract_andreini_paired_sonnet_segments(html: str) -> list[SonnetCandidateSegment]:
    """Split an Andreini paired page at its printed ``SONETTO`` headings only."""

    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(".mw-parser-output")
    page_output = root.select_one(".prp-pages-output") if root is not None else None
    if page_output is None:
        raise ValueError("paired Andreini page is missing its scan-page output")

    segments: list[SonnetCandidateSegment] = []
    label = ""
    poem_blocks: list[str] = []
    for element in page_output.find_all(recursive=False):
        if not isinstance(element, Tag):
            continue
        if "centertext" in element.get("class", []):
            heading = " ".join(element.get_text(" ", strip=True).split())
            if re.fullmatch(r"SONETTO\s+[IVXLCDM]+\.", heading):
                if label:
                    segments.append(
                        SonnetCandidateSegment(
                            index=len(segments) + 1,
                            label=label,
                            raw_text=extract_poem_text(
                                "<div class='mw-parser-output'>"
                                + "".join(poem_blocks)
                                + "</div>"
                            ),
                        )
                    )
                label = heading.removesuffix(".")
                poem_blocks = []
                continue
        if "poem" in element.get("class", []) and label:
            poem_blocks.append(str(element))

    if label:
        segments.append(
            SonnetCandidateSegment(
                index=len(segments) + 1,
                label=label,
                raw_text=extract_poem_text(
                    "<div class='mw-parser-output'>"
                    + "".join(poem_blocks)
                    + "</div>"
                ),
            )
        )
    if len(segments) != 2 or any(not segment.raw_text for segment in segments):
        raise ValueError("paired Andreini page did not yield exactly two headed poems")
    return segments


def read_sonnet_source_manifest(path: Path) -> list[SonnetWikisourceSource]:
    """Read the public source-decision manifest used by the expansion audit."""

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "source_id",
        "title",
        "author",
        "source_archive",
        "landing_page_url",
        "language_variety",
        "role",
        "status",
        "license_or_reuse_status",
        "attribution_required",
    }
    if not rows or set(rows[0]) < required:
        raise ValueError("sonnet source manifest is missing required columns")
    return [SonnetWikisourceSource(**{field: row[field] for field in required}) for row in rows]


def select_sonnet_wikisource_source(
    rows: list[SonnetWikisourceSource], source_id: str
) -> SonnetWikisourceSource:
    """Select exactly one Italian-Wikisource source approved for auditing."""

    matches = [row for row in rows if row.source_id == source_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one sonnet source row: {source_id}")
    source = matches[0]
    if source.source_archive != "Italian Wikisource":
        raise ValueError(f"source is not Italian Wikisource: {source_id}")
    expectation = SONNET_COLLECTION_EXPECTATIONS.get(source_id)
    if expectation is None:
        raise ValueError(f"no sonnet collection expectation for source: {source_id}")
    if source.status != expectation.audit_status:
        raise ValueError(f"source is not approved for this audit: {source_id}")
    return source


def read_active_poem_texts(manifest_path: Path, repo_root: Path) -> dict[str, list[str]]:
    """Map normalized active corpus text to poem IDs for exact duplicate checks."""

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    texts: dict[str, list[str]] = {}
    for row in rows:
        if row.get("include_in_training") != "True":
            continue
        clean_path = repo_root / row["clean_text_path"]
        text = clean_path.read_text(encoding="utf-8")
        normalized = normalize_poem_for_duplicate_check(text)
        texts.setdefault(normalized, []).append(row["poem_id"])
    return texts


def normalize_poem_for_duplicate_check(text: str) -> str:
    """Normalize only case and whitespace for conservative exact duplicate checks."""

    return re.sub(r"\s+", " ", text).strip().casefold()


def candidate_status(
    *, raw_text: str,
    line_count_clean: int,
    duplicate_ids: list[str],
) -> str:
    """Return the first activation gate that prevents a page from being eligible."""

    if not raw_text.strip():
        return "empty_after_extraction"
    if line_count_clean != 14:
        return "not_14_cleaned_lines"
    if duplicate_ids:
        return "exact_duplicate_active_corpus"
    return "eligible_14_lines"


def bounded_sample(text: str, *, from_end: bool, limit: int = 240) -> str:
    """Return a bounded audit sample without retaining an entire poem in the report."""

    compact = text.strip()
    if len(compact) <= limit:
        return compact
    if from_end:
        return compact[-limit:]
    return compact[:limit]


def portable_path(path: Path, repo_root: Path) -> str:
    """Use repository-relative paths when possible in a portable report."""

    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def utc_now() -> str:
    """Return a second-precision UTC timestamp for an inspection report."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
