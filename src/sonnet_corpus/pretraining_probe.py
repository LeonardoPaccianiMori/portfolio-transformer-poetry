"""Source probes for broader Italian pretraining data."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from .bpe import BytePairEncodingTokenizer
from .gutenberg import (
    FetchedGutenbergText,
    GutenbergRateLimiter,
    fetch_gutenberg_text,
    strip_gutenberg_boilerplate,
)
from .pretraining_manifest import PretrainingSourceRow, read_pretraining_manifest


FetchGutenbergText = Callable[..., FetchedGutenbergText]


@dataclass(frozen=True)
class GutenbergProbeResult:
    source_id: str
    title: str
    author: str
    ebook_id: str
    landing_page_url: str
    status: str
    error: str
    fetched_url: str
    raw_character_count: int
    cleaned_character_count: int
    cleaned_word_count: int
    bpe_token_count: int | None
    bpe_tokenization_error: str
    cleaning_notes: str


def probe_gutenberg_sources(
    *,
    manifest_path: Path,
    report_path: Path,
    tokenizer_path: Path | None = None,
    request_delay: float = 1.0,
    fetch_text: FetchGutenbergText = fetch_gutenberg_text,
    session: requests.Session | None = None,
) -> dict[str, object]:
    """Probe active Project Gutenberg prose rows and write a compact JSON report."""

    started_at = _utc_now()
    rows = read_pretraining_manifest(manifest_path)
    probe_rows = select_gutenberg_probe_rows(rows)
    tokenizer = _load_tokenizer(tokenizer_path)
    rate_limiter = GutenbergRateLimiter(request_delay=request_delay)

    results: list[GutenbergProbeResult] = []
    for row in probe_rows:
        rate_limiter.wait()
        results.append(
            _probe_gutenberg_row(
                row=row,
                tokenizer=tokenizer,
                fetch_text=fetch_text,
                session=session,
            )
        )

    report = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "manifest_path": _portable_path(manifest_path),
        "tokenizer_path": (
            _portable_path(tokenizer_path) if tokenizer_path is not None else ""
        ),
        "selected_rows": len(probe_rows),
        "skipped_rows": len(rows) - len(probe_rows),
        "total_cleaned_characters": sum(
            result.cleaned_character_count for result in results
        ),
        "total_cleaned_words": sum(result.cleaned_word_count for result in results),
        "bpe_tokenized_rows": sum(
            result.bpe_token_count is not None for result in results
        ),
        "bpe_tokenization_error_rows": sum(
            bool(result.bpe_tokenization_error) for result in results
        ),
        "total_bpe_tokens": _complete_bpe_token_total(results, tokenizer),
        "results": [asdict(result) for result in results],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def select_gutenberg_probe_rows(rows: list[PretrainingSourceRow]) -> list[PretrainingSourceRow]:
    """Return active Project Gutenberg prose rows for the first source probe."""

    return [
        row
        for row in rows
        if row.source_archive == "Project Gutenberg"
        and row.inclusion_status == "include_probe"
        and row.text_kind == "prose"
    ]


def count_whitespace_words(text: str) -> int:
    """Count simple whitespace-delimited word-like units."""

    return len(re.findall(r"\S+", text))


def _probe_gutenberg_row(
    *,
    row: PretrainingSourceRow,
    tokenizer: BytePairEncodingTokenizer | None,
    fetch_text: FetchGutenbergText,
    session: requests.Session | None,
) -> GutenbergProbeResult:
    try:
        fetched = fetch_text(row.ebook_id, session=session)
    except Exception as exc:
        return GutenbergProbeResult(
            source_id=row.source_id,
            title=row.title,
            author=row.author,
            ebook_id=row.ebook_id,
            landing_page_url=row.landing_page_url,
            status="error",
            error=str(exc),
            fetched_url="",
            raw_character_count=0,
            cleaned_character_count=0,
            cleaned_word_count=0,
            bpe_token_count=None,
            bpe_tokenization_error="",
            cleaning_notes=row.cleaning_notes,
        )

    cleaned = strip_gutenberg_boilerplate(fetched.text)
    bpe_token_count, bpe_tokenization_error = _try_count_bpe_tokens(
        cleaned,
        tokenizer,
    )
    return GutenbergProbeResult(
        source_id=row.source_id,
        title=row.title,
        author=row.author,
        ebook_id=row.ebook_id,
        landing_page_url=row.landing_page_url,
        status="ok",
        error="",
        fetched_url=fetched.url,
        raw_character_count=len(fetched.text),
        cleaned_character_count=len(cleaned),
        cleaned_word_count=count_whitespace_words(cleaned),
        bpe_token_count=bpe_token_count,
        bpe_tokenization_error=bpe_tokenization_error,
        cleaning_notes=row.cleaning_notes,
    )


def _try_count_bpe_tokens(
    text: str,
    tokenizer: BytePairEncodingTokenizer | None,
) -> tuple[int | None, str]:
    if tokenizer is None:
        return None, ""

    try:
        return len(tokenizer.encode(text)), ""
    except KeyError as exc:
        missing_token = exc.args[0]
        return None, f"tokenizer vocabulary does not contain {missing_token!r}"


def _complete_bpe_token_total(
    results: list[GutenbergProbeResult],
    tokenizer: BytePairEncodingTokenizer | None,
) -> int | None:
    if tokenizer is None or any(
        result.status != "ok" or result.bpe_token_count is None
        for result in results
    ):
        return None

    return sum(result.bpe_token_count for result in results)


def _load_tokenizer(tokenizer_path: Path | None) -> BytePairEncodingTokenizer | None:
    if tokenizer_path is None:
        return None
    return BytePairEncodingTokenizer.load(tokenizer_path)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
