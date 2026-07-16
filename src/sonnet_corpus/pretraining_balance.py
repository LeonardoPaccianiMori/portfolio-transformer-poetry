"""Audit source and author concentration in a local pretraining corpus."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .pretraining_build import select_pretraining_build_rows
from .pretraining_manifest import PretrainingSourceRow, read_pretraining_manifest
from .pretraining_probe import count_whitespace_words


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class PretrainingBalanceConfig:
    """Input paths and concentration thresholds for one balance audit."""

    manifest_path: Path
    processed_dir: Path
    json_report_path: Path
    markdown_report_path: Path
    max_source_share: float = 0.15
    max_author_share: float = 0.20


@dataclass(frozen=True)
class PretrainingBalanceEntry:
    """Corpus contribution from one source work or one author."""

    name: str
    cleaned_character_count: int
    cleaned_word_count: int
    character_share: float
    word_share: float
    exceeds_character_cap: bool
    exceeds_word_cap: bool


@dataclass(frozen=True)
class PretrainingBalanceReport:
    """Concentration measurements for an already-built local corpus."""

    started_at_utc: str
    finished_at_utc: str
    manifest_path: str
    processed_dir: str
    selected_source_count: int
    total_cleaned_characters: int
    total_cleaned_words: int
    max_source_share: float
    max_author_share: float
    source_entries: list[PretrainingBalanceEntry]
    author_entries: list[PretrainingBalanceEntry]
    source_character_cap_violations: list[str]
    source_word_cap_violations: list[str]
    author_character_cap_violations: list[str]
    author_word_cap_violations: list[str]


def audit_pretraining_corpus_balance(
    config: PretrainingBalanceConfig,
    *,
    progress: ProgressCallback | None = None,
) -> PretrainingBalanceReport:
    """Measure source and author shares without changing corpus text."""

    _validate_share(config.max_source_share, label="max_source_share")
    _validate_share(config.max_author_share, label="max_author_share")
    started_at = _utc_now()
    rows = select_pretraining_build_rows(read_pretraining_manifest(config.manifest_path))
    if not rows:
        raise ValueError("no active broader pretraining prose rows selected")

    source_dir = config.processed_dir / "sources"
    _validate_source_files(source_dir, rows)
    source_counts: dict[str, tuple[int, int]] = {}
    author_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

    for index, row in enumerate(rows, start=1):
        _write_progress(progress, f"auditing source {index}/{len(rows)}: {row.source_id}")
        text = (source_dir / f"{row.source_id}.txt").read_text(encoding="utf-8")
        character_count = len(text)
        word_count = count_whitespace_words(text)
        source_counts[row.source_id] = (character_count, word_count)
        author_counts[row.author][0] += character_count
        author_counts[row.author][1] += word_count

    total_characters = sum(counts[0] for counts in source_counts.values())
    total_words = sum(counts[1] for counts in source_counts.values())
    source_entries = _make_entries(
        source_counts,
        total_characters=total_characters,
        total_words=total_words,
        max_share=config.max_source_share,
    )
    author_entries = _make_entries(
        author_counts,
        total_characters=total_characters,
        total_words=total_words,
        max_share=config.max_author_share,
    )
    report = PretrainingBalanceReport(
        started_at_utc=started_at,
        finished_at_utc=_utc_now(),
        manifest_path=_portable_path(config.manifest_path),
        processed_dir=_portable_path(config.processed_dir),
        selected_source_count=len(rows),
        total_cleaned_characters=total_characters,
        total_cleaned_words=total_words,
        max_source_share=config.max_source_share,
        max_author_share=config.max_author_share,
        source_entries=source_entries,
        author_entries=author_entries,
        source_character_cap_violations=_violation_names(
            source_entries,
            attribute="exceeds_character_cap",
        ),
        source_word_cap_violations=_violation_names(
            source_entries,
            attribute="exceeds_word_cap",
        ),
        author_character_cap_violations=_violation_names(
            author_entries,
            attribute="exceeds_character_cap",
        ),
        author_word_cap_violations=_violation_names(
            author_entries,
            attribute="exceeds_word_cap",
        ),
    )
    _write_json_report(report, config.json_report_path)
    _write_markdown_report(report, config.markdown_report_path)
    return report


def _validate_source_files(source_dir: Path, rows: list[PretrainingSourceRow]) -> None:
    if not source_dir.is_dir():
        raise ValueError(f"processed source directory does not exist: {source_dir}")

    expected_ids = {row.source_id for row in rows}
    actual_ids = {path.stem for path in source_dir.glob("*.txt")}
    missing_ids = sorted(expected_ids - actual_ids)
    unexpected_ids = sorted(actual_ids - expected_ids)
    if missing_ids or unexpected_ids:
        details = []
        if missing_ids:
            details.append(f"missing={','.join(missing_ids)}")
        if unexpected_ids:
            details.append(f"unexpected={','.join(unexpected_ids)}")
        raise ValueError("processed source files do not match active manifest rows: " + "; ".join(details))


def _make_entries(
    counts: dict[str, tuple[int, int]],
    *,
    total_characters: int,
    total_words: int,
    max_share: float,
) -> list[PretrainingBalanceEntry]:
    entries = [
        PretrainingBalanceEntry(
            name=name,
            cleaned_character_count=character_count,
            cleaned_word_count=word_count,
            character_share=character_count / total_characters,
            word_share=word_count / total_words,
            exceeds_character_cap=character_count / total_characters > max_share,
            exceeds_word_cap=word_count / total_words > max_share,
        )
        for name, (character_count, word_count) in counts.items()
    ]
    return sorted(
        entries,
        key=lambda entry: (-entry.cleaned_character_count, entry.name),
    )


def _violation_names(
    entries: list[PretrainingBalanceEntry],
    *,
    attribute: str,
) -> list[str]:
    return [entry.name for entry in entries if getattr(entry, attribute)]


def _validate_share(value: float, *, label: str) -> None:
    if not 0 < value <= 1:
        raise ValueError(f"{label} must be greater than zero and at most one")


def _write_json_report(report: PretrainingBalanceReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_markdown_report(report: PretrainingBalanceReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pretraining Corpus Balance Audit",
        "",
        f"- Sources: {report.selected_source_count}",
        f"- Cleaned characters: {report.total_cleaned_characters:,}",
        f"- Whitespace-delimited units: {report.total_cleaned_words:,}",
        f"- Work cap: {report.max_source_share:.0%}",
        f"- Author cap: {report.max_author_share:.0%}",
        "",
        "## Work Shares",
        "",
        _markdown_table(report.source_entries),
        "",
        "## Author Shares",
        "",
        _markdown_table(report.author_entries),
        "",
        "## Cap Violations",
        "",
        _markdown_violations("Work character", report.source_character_cap_violations),
        _markdown_violations("Work word", report.source_word_cap_violations),
        _markdown_violations("Author character", report.author_character_cap_violations),
        _markdown_violations("Author word", report.author_word_cap_violations),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(entries: list[PretrainingBalanceEntry]) -> str:
    lines = [
        "| Name | Characters | Character share | Units | Unit share |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        "| "
        f"{entry.name} | {entry.cleaned_character_count:,} | {entry.character_share:.2%} | "
        f"{entry.cleaned_word_count:,} | {entry.word_share:.2%} |"
        for entry in entries
    )
    return "\n".join(lines)


def _markdown_violations(label: str, names: list[str]) -> str:
    value = ", ".join(names) if names else "none"
    return f"- {label} cap: {value}"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _write_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
