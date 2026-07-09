"""Build the broader Italian pretraining corpus from the audited manifest."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from .gutenberg import (
    FetchedGutenbergText,
    GutenbergRateLimiter,
    fetch_gutenberg_text,
    strip_gutenberg_boilerplate,
)
from .liber_liber import (
    FetchedLiberLiberText,
    LiberLiberRateLimiter,
    fetch_liber_liber_text,
)
from .pretraining_cleaning import clean_pretraining_text, validate_cleaned_text
from .pretraining_manifest import PretrainingSourceRow, read_pretraining_manifest
from .pretraining_probe import count_whitespace_words


FetchGutenbergText = Callable[..., FetchedGutenbergText]
FetchLiberLiberText = Callable[..., FetchedLiberLiberText]


@dataclass(frozen=True)
class PretrainingBuildConfig:
    """Configuration for a deterministic broader-corpus build."""

    manifest_path: Path = Path("data/metadata/broader_prose_sources_manifest.csv")
    processed_dir: Path = Path("data/local/pretraining/processed")
    report_path: Path = Path("data/local/pretraining/build_report.json")
    temp_dir: Path = Path("data/interim/pretraining_build")
    corpus_version: str = "broader_prose_v1"
    request_delay_seconds: float = 1.0
    min_character_count: int = 200


@dataclass(frozen=True)
class BuiltPretrainingSource:
    """One successfully built source in the broader pretraining corpus."""

    source_id: str
    title: str
    author: str
    source_archive: str
    landing_page_url: str
    fetched_url: str
    processed_path: str
    raw_character_count: int
    cleaned_character_count: int
    cleaned_word_count: int
    first_characters: str
    last_characters: str
    license_notes: str
    cleaning_notes: str


@dataclass(frozen=True)
class PretrainingBuildReport:
    """Compact provenance and size report for one corpus build."""

    corpus_version: str
    started_at_utc: str
    finished_at_utc: str
    manifest_path: str
    processed_dir: str
    combined_corpus_path: str
    selected_rows: int
    skipped_rows: int
    total_cleaned_characters: int
    total_cleaned_words: int
    source_archive_shares: dict[str, dict[str, int | float]]
    author_shares: dict[str, dict[str, int | float]]
    sources: list[BuiltPretrainingSource]


def select_pretraining_build_rows(
    rows: list[PretrainingSourceRow],
) -> list[PretrainingSourceRow]:
    """Return active prose rows that are approved for the broader-corpus build."""

    return [
        row
        for row in rows
        if row.inclusion_status == "include_probe"
        and row.text_kind == "prose"
        and row.source_archive in {"Project Gutenberg", "Liber Liber"}
    ]


def build_pretraining_corpus(
    config: PretrainingBuildConfig,
    *,
    fetch_gutenberg: FetchGutenbergText = fetch_gutenberg_text,
    fetch_liber_liber: FetchLiberLiberText = fetch_liber_liber_text,
    session: requests.Session | None = None,
) -> PretrainingBuildReport:
    """Fetch, clean, validate, and write the broader Italian pretraining corpus."""

    started_at = _utc_now()
    rows = read_pretraining_manifest(config.manifest_path)
    selected_rows = select_pretraining_build_rows(rows)
    skipped_rows = len(rows) - len(selected_rows)
    if not selected_rows:
        raise ValueError("no active broader pretraining prose rows selected")

    temp_root = config.temp_dir
    raw_dir = temp_root / "raw"
    interim_dir = temp_root / "interim"
    staged_processed_dir = temp_root / "processed"
    staged_sources_dir = staged_processed_dir / "sources"
    combined_corpus_path = staged_processed_dir / "corpus.txt"

    _validate_deletable_directory(temp_root, label="temp_dir")
    _validate_deletable_directory(config.processed_dir, label="processed_dir")
    _prepare_temp_tree(temp_root, raw_dir, interim_dir, staged_sources_dir)
    gutenberg_limiter = GutenbergRateLimiter(config.request_delay_seconds)
    liber_liber_limiter = LiberLiberRateLimiter(config.request_delay_seconds)

    try:
        built_sources: list[BuiltPretrainingSource] = []
        combined_parts: list[str] = []
        for row in selected_rows:
            if row.source_archive == "Project Gutenberg":
                gutenberg_limiter.wait()
                raw_text, fetched_url = _fetch_gutenberg_source(
                    row,
                    fetch_gutenberg=fetch_gutenberg,
                    session=session,
                )
            elif row.source_archive == "Liber Liber":
                liber_liber_limiter.wait()
                raw_text, fetched_url = _fetch_liber_liber_source(
                    row,
                    fetch_liber_liber=fetch_liber_liber,
                    session=session,
                )
            else:
                raise ValueError(f"unsupported source archive: {row.source_archive}")

            cleaned_text = clean_pretraining_text(raw_text, source_id=row.source_id)
            validate_cleaned_text(
                cleaned_text,
                source_id=row.source_id,
                min_character_count=config.min_character_count,
            )

            raw_path = raw_dir / f"{row.source_id}.txt"
            interim_path = interim_dir / f"{row.source_id}.txt"
            source_path = staged_sources_dir / f"{row.source_id}.txt"
            raw_path.write_text(raw_text, encoding="utf-8")
            interim_path.write_text(cleaned_text, encoding="utf-8")
            source_path.write_text(cleaned_text, encoding="utf-8")
            combined_parts.append(cleaned_text.strip() + "\n")

            built_sources.append(
                BuiltPretrainingSource(
                    source_id=row.source_id,
                    title=row.title,
                    author=row.author,
                    source_archive=row.source_archive,
                    landing_page_url=row.landing_page_url,
                    fetched_url=fetched_url,
                    processed_path=_portable_path(
                        config.processed_dir / "sources" / f"{row.source_id}.txt"
                    ),
                    raw_character_count=len(raw_text),
                    cleaned_character_count=len(cleaned_text),
                    cleaned_word_count=count_whitespace_words(cleaned_text),
                    first_characters=_sample_start(cleaned_text),
                    last_characters=_sample_end(cleaned_text),
                    license_notes=row.license_notes,
                    cleaning_notes=row.cleaning_notes,
                )
            )

        combined_corpus_path.write_text("\n".join(combined_parts), encoding="utf-8")
        report = _make_report(
            config=config,
            started_at=started_at,
            selected_rows=len(selected_rows),
            skipped_rows=skipped_rows,
            sources=built_sources,
        )
        staged_report_path = staged_processed_dir / config.report_path.name
        _write_report(report, staged_report_path)
        _publish_processed_tree(
            staged_processed_dir=staged_processed_dir,
            processed_dir=config.processed_dir,
            report_path=config.report_path,
        )
    except Exception:
        raise
    else:
        shutil.rmtree(temp_root)

    return report


def _fetch_gutenberg_source(
    row: PretrainingSourceRow,
    *,
    fetch_gutenberg: FetchGutenbergText,
    session: requests.Session | None,
) -> tuple[str, str]:
    fetched = fetch_gutenberg(row.ebook_id, session=session)
    return strip_gutenberg_boilerplate(fetched.text), fetched.url


def _fetch_liber_liber_source(
    row: PretrainingSourceRow,
    *,
    fetch_liber_liber: FetchLiberLiberText,
    session: requests.Session | None,
) -> tuple[str, str]:
    fetched = fetch_liber_liber(
        row.landing_page_url,
        title=row.title,
        session=session,
    )
    return fetched.text, fetched.archive_url


def _prepare_temp_tree(
    temp_root: Path,
    raw_dir: Path,
    interim_dir: Path,
    staged_sources_dir: Path,
) -> None:
    if temp_root.exists():
        shutil.rmtree(temp_root)
    raw_dir.mkdir(parents=True)
    interim_dir.mkdir(parents=True)
    staged_sources_dir.mkdir(parents=True)


def _make_report(
    *,
    config: PretrainingBuildConfig,
    started_at: str,
    selected_rows: int,
    skipped_rows: int,
    sources: list[BuiltPretrainingSource],
) -> PretrainingBuildReport:
    total_characters = sum(source.cleaned_character_count for source in sources)
    total_words = sum(source.cleaned_word_count for source in sources)
    return PretrainingBuildReport(
        corpus_version=config.corpus_version,
        started_at_utc=started_at,
        finished_at_utc=_utc_now(),
        manifest_path=_portable_path(config.manifest_path),
        processed_dir=_portable_path(config.processed_dir),
        combined_corpus_path=_portable_path(config.processed_dir / "corpus.txt"),
        selected_rows=selected_rows,
        skipped_rows=skipped_rows,
        total_cleaned_characters=total_characters,
        total_cleaned_words=total_words,
        source_archive_shares=_share_report(_count_by_archive(sources), total_characters),
        author_shares=_share_report(_count_by_author(sources), total_characters),
        sources=sources,
    )


def _count_by_archive(sources: list[BuiltPretrainingSource]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for source in sources:
        counts[source.source_archive] += source.cleaned_character_count
    return counts


def _count_by_author(sources: list[BuiltPretrainingSource]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for source in sources:
        counts[source.author] += source.cleaned_character_count
    return counts


def _share_report(
    counts: Counter[str],
    total: int,
) -> dict[str, dict[str, int | float]]:
    return {
        key: {
            "cleaned_character_count": count,
            "share": count / total if total else 0.0,
        }
        for key, count in sorted(counts.items())
    }


def _write_report(report: PretrainingBuildReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _publish_processed_tree(
    *,
    staged_processed_dir: Path,
    processed_dir: Path,
    report_path: Path,
) -> None:
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged_processed_dir), str(processed_dir))
    built_report_path = processed_dir / report_path.name
    if report_path != built_report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_report_path, report_path)


def _validate_deletable_directory(path: Path, *, label: str) -> None:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    forbidden = {Path("/").resolve(), cwd, cwd.parent}
    if resolved in forbidden:
        raise ValueError(f"{label} is too broad to delete or replace safely: {path}")


def _sample_start(text: str, size: int = 240) -> str:
    return _compact_sample(text[:size])


def _sample_end(text: str, size: int = 240) -> str:
    return _compact_sample(text[-size:])


def _compact_sample(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
