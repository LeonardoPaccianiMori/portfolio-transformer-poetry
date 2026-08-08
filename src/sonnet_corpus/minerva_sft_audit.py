"""Audit the V5 sonnet texts before another Minerva adaptation recipe."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sonnet_corpus.dataset_text import (
    load_poem_text,
    read_manifest_rows,
    select_manifest_rows,
    validate_manifest_rows,
)
from sonnet_corpus.task_format import SONNET_LINE_COUNT


SPLITS = ("train", "validation", "test")
SUSPICIOUS_MARKERS = {
    "wiki_template": re.compile(r"\{\{|\}\}"),
    "wiki_link": re.compile(r"\[\[|\]\]"),
    "html_or_reference": re.compile(r"</?(?:ref|div|span|br|p)\b", re.IGNORECASE),
    "web_address": re.compile(r"https?://|www\.", re.IGNORECASE),
    "replacement_character": re.compile("\ufffd"),
}


def audit_minerva_sft_corpus(
    *,
    repo_root: Path,
    manifest_path: Path,
    dataset: str,
    json_report_path: Path,
    markdown_report_path: Path,
    review_sample_path: Path,
    review_sample_size: int = 24,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Audit every selected poem and write public evidence plus a review sample."""
    if review_sample_size <= 0:
        raise ValueError("review_sample_size must be greater than 0")
    resolved_manifest = _resolve_path(repo_root, manifest_path)
    rows = read_manifest_rows(resolved_manifest)
    validate_manifest_rows(rows, dataset)
    split_rows = {
        split: select_manifest_rows(rows, dataset=dataset, split=split)
        for split in SPLITS
    }
    if any(not split_rows[split] for split in SPLITS):
        raise ValueError("Minerva SFT audit requires non-empty train/validation/test splits")

    selected_rows = [row for split in SPLITS for row in split_rows[split]]
    _report(progress, f"loading {len(selected_rows)} selected V5 poems")
    text_by_id: dict[str, str] = {}
    issues: list[dict[str, str]] = []
    fingerprints: dict[str, list[str]] = defaultdict(list)
    marker_counts: Counter[str] = Counter()
    line_lengths: list[int] = []

    for index, row in enumerate(selected_rows, start=1):
        text = load_poem_text(row, repo_root=repo_root)
        text_by_id[row["poem_id"]] = text
        fingerprints[_text_fingerprint(text)].append(row["poem_id"])
        poem_issues, poem_marker_counts, poem_line_lengths = _audit_poem(row, text)
        issues.extend(poem_issues)
        marker_counts.update(poem_marker_counts)
        line_lengths.extend(poem_line_lengths)
        if index % 250 == 0 or index == len(selected_rows):
            _report(progress, f"audited poem {index}/{len(selected_rows)}")

    split_by_poem_id = {
        row["poem_id"]: split for split, rows in split_rows.items() for row in rows
    }
    duplicates = [
        {
            "poem_ids": poem_ids,
            "count": len(poem_ids),
            "splits": [split_by_poem_id[poem_id] for poem_id in poem_ids],
            "cross_split": len({split_by_poem_id[poem_id] for poem_id in poem_ids}) > 1,
        }
        for poem_ids in fingerprints.values()
        if len(poem_ids) > 1
    ]
    duplicates.sort(key=lambda row: (-row["count"], row["poem_ids"]))
    report = {
        "audit_version": "minerva_v5_sft_corpus_audit_v1",
        "dataset": dataset,
        "manifest_path": _portable_path(resolved_manifest, repo_root),
        "manifest_sha256": _file_sha256(resolved_manifest),
        "selected_poem_count": len(selected_rows),
        "split_poem_counts": {
            split: len(split_rows[split]) for split in SPLITS
        },
        "period_counts": _count_field(selected_rows, "period"),
        "author_counts": _count_field(selected_rows, "author"),
        "source_archive_counts": _count_field(selected_rows, "source_archive"),
        "source_collection_counts": _count_field(selected_rows, "source_collection"),
        "train_author_concentration": _concentration(split_rows["train"], "author"),
        "train_source_concentration": _concentration(
            split_rows["train"], "source_collection"
        ),
        "cleaning_metadata": {
            "editorial_brackets_removed": _true_count(
                selected_rows, "editorial_brackets_removed"
            ),
            "line_markers_removed": _true_count(selected_rows, "line_markers_removed"),
            "rows_with_cleaning_notes": sum(
                bool(row.get("cleaning_notes", "").strip()) for row in selected_rows
            ),
            "rows_with_audit_notes": sum(
                bool(row.get("audit_notes", "").strip()) for row in selected_rows
            ),
        },
        "structural_issue_count": len(issues),
        "structural_issues": issues,
        "suspicious_marker_counts": dict(sorted(marker_counts.items())),
        "exact_normalized_duplicate_group_count": len(duplicates),
        "cross_split_duplicate_group_count": sum(
            row["cross_split"] for row in duplicates
        ),
        "exact_normalized_duplicate_groups": duplicates,
        "line_length_characters": {
            "minimum": min(line_lengths),
            "maximum": max(line_lengths),
            "mean": sum(line_lengths) / len(line_lengths),
            "lines_over_120": sum(length > 120 for length in line_lengths),
            "lines_under_4": sum(length < 4 for length in line_lengths),
        },
        "automated_structural_gate": (
            "pass" if not issues and not duplicates else "review_required"
        ),
        "editorial_conclusion": (
            "Automated checks cannot certify historical grammar; review the "
            "deterministic sample before freezing another Minerva recipe."
        ),
    }

    sample_rows = _select_review_sample(split_rows["train"], review_sample_size)
    _write_json(_resolve_path(repo_root, json_report_path), report)
    _write_text(
        _resolve_path(repo_root, markdown_report_path),
        _render_markdown_report(report),
    )
    _write_text(
        _resolve_path(repo_root, review_sample_path),
        _render_review_sample(sample_rows, text_by_id),
    )
    _report(progress, f"wrote audit report: {markdown_report_path}")
    _report(progress, f"wrote review sample: {review_sample_path}")
    return report


def _audit_poem(
    row: dict[str, str],
    text: str,
) -> tuple[list[dict[str, str]], Counter[str], list[int]]:
    issues: list[dict[str, str]] = []
    markers: Counter[str] = Counter()
    lines = text.splitlines()
    if len(lines) != SONNET_LINE_COUNT:
        issues.append(_issue(row, "line_count", f"found {len(lines)} lines"))
    if any(not line.strip() for line in lines):
        issues.append(_issue(row, "empty_line", "contains an empty line"))
    if "\r" in text:
        issues.append(_issue(row, "carriage_return", "contains carriage returns"))
    if not text.endswith("\n"):
        issues.append(_issue(row, "terminal_newline", "missing terminal newline"))
    if any(ord(char) < 32 and char not in "\n\t" for char in text):
        issues.append(_issue(row, "control_character", "contains a control character"))

    for marker_name, pattern in SUSPICIOUS_MARKERS.items():
        count = len(pattern.findall(text))
        if count:
            markers[marker_name] += count
            issues.append(
                _issue(row, f"marker:{marker_name}", f"found {count} occurrence(s)")
            )

    lengths = [len(line.strip()) for line in lines]
    if any(length > 120 for length in lengths):
        issues.append(_issue(row, "long_line", "contains a line over 120 characters"))
    if any(length < 4 for length in lengths):
        issues.append(_issue(row, "short_line", "contains a line under 4 characters"))
    return issues, markers, lengths


def _issue(row: dict[str, str], issue_type: str, detail: str) -> dict[str, str]:
    return {
        "poem_id": row["poem_id"],
        "author": row.get("author", ""),
        "issue_type": issue_type,
        "detail": detail,
    }


def _text_fingerprint(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _count_field(rows: Sequence[dict[str, str]], field: str) -> dict[str, int]:
    counts = Counter(row.get(field, "") or "(missing)" for row in rows)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _concentration(rows: Sequence[dict[str, str]], field: str) -> list[dict[str, Any]]:
    total = len(rows)
    counts = Counter(row.get(field, "") or "(missing)" for row in rows)
    return [
        {"name": name, "count": count, "share": count / total}
        for name, count in counts.most_common(10)
    ]


def _true_count(rows: Sequence[dict[str, str]], field: str) -> int:
    return sum(row.get(field) == "True" for row in rows)


def _select_review_sample(
    train_rows: Sequence[dict[str, str]],
    sample_size: int,
) -> list[dict[str, str]]:
    by_period: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        by_period[row.get("period", "(missing)")].append(row)
    for rows in by_period.values():
        rows.sort(key=lambda row: _sample_key(row["poem_id"]))

    sample: list[dict[str, str]] = []
    periods = sorted(by_period)
    index = 0
    while len(sample) < min(sample_size, len(train_rows)):
        added = False
        for period in periods:
            rows = by_period[period]
            if index < len(rows):
                sample.append(rows[index])
                added = True
                if len(sample) == sample_size:
                    break
        if not added:
            break
        index += 1
    return sample


def _sample_key(poem_id: str) -> str:
    return hashlib.sha256(f"minerva-v5-audit:{poem_id}".encode("utf-8")).hexdigest()


def _render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Minerva V5 SFT Corpus Audit",
        "",
        "## Scope",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Manifest: `{report['manifest_path']}`",
        f"- Manifest SHA-256: `{report['manifest_sha256']}`",
        f"- Selected poems: {report['selected_poem_count']:,}",
        "- Split counts: " + " / ".join(
            f"{split} {report['split_poem_counts'][split]:,}" for split in SPLITS
        ),
        "",
        "## Automated Structural Gate",
        "",
        f"- Result: **{report['automated_structural_gate']}**",
        f"- Structural issues: {report['structural_issue_count']:,}",
        "- Exact normalized duplicate groups: "
        f"{report['exact_normalized_duplicate_group_count']:,}",
        "- Cross-split duplicate groups: "
        f"{report['cross_split_duplicate_group_count']:,}",
        f"- Suspicious markers: `{report['suspicious_marker_counts']}`",
        "",
        "## Composition",
        "",
        "### Periods",
        "",
        "| Period | Poems |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {name} | {count:,} |" for name, count in report["period_counts"].items()
    )
    lines.extend([
        "",
        "### Largest Training Authors",
        "",
        "| Author | Poems | Share |",
        "| --- | ---: | ---: |",
    ])
    lines.extend(
        f"| {row['name']} | {row['count']:,} | {row['share']:.1%} |"
        for row in report["train_author_concentration"]
    )
    lines.extend([
        "",
        "### Largest Training Collections",
        "",
        "| Collection | Poems | Share |",
        "| --- | ---: | ---: |",
    ])
    lines.extend(
        f"| {row['name']} | {row['count']:,} | {row['share']:.1%} |"
        for row in report["train_source_concentration"]
    )
    lengths = report["line_length_characters"]
    cleaning = report["cleaning_metadata"]
    lines.extend([
        "",
        "## Cleaning And Line Diagnostics",
        "",
        f"- Line length: minimum {lengths['minimum']}, mean {lengths['mean']:.1f}, "
        f"maximum {lengths['maximum']} characters.",
        f"- Lines over 120 characters: {lengths['lines_over_120']:,}.",
        f"- Lines under 4 characters: {lengths['lines_under_4']:,}.",
        "- Poems with recorded editorial-bracket removal: "
        f"{cleaning['editorial_brackets_removed']:,}.",
        "- Poems with recorded line-marker removal: "
        f"{cleaning['line_markers_removed']:,}.",
        f"- Poems with cleaning notes: {cleaning['rows_with_cleaning_notes']:,}.",
        f"- Poems with audit notes: {cleaning['rows_with_audit_notes']:,}.",
        "",
        "## Interpretation",
        "",
        report["editorial_conclusion"],
        "The companion review sample is not training data duplication; it is a "
        "deterministic view of committed V5 texts for editorial inspection.",
    ])
    if report["structural_issues"]:
        lines.extend([
            "",
            "## Structural Issues",
            "",
            "| Poem | Author | Type | Detail |",
            "| --- | --- | --- | --- |",
        ])
        lines.extend(
            f"| {row['poem_id']} | {row['author']} | {row['issue_type']} | "
            f"{row['detail']} |"
            for row in report["structural_issues"]
        )
    if report["exact_normalized_duplicate_groups"]:
        lines.extend([
            "",
            "## Exact Duplicate Groups",
            "",
            "| Poem IDs | Splits | Cross-split leakage |",
            "| --- | --- | --- |",
        ])
        lines.extend(
            f"| {'; '.join(row['poem_ids'])} | {'; '.join(row['splits'])} | "
            f"{'yes' if row['cross_split'] else 'no'} |"
            for row in report["exact_normalized_duplicate_groups"]
        )
    return "\n".join(lines) + "\n"


def _render_review_sample(
    rows: Sequence[dict[str, str]],
    text_by_id: dict[str, str],
) -> str:
    sections = [
        "# Minerva V5 Training-Text Review Sample",
        "",
        "This deterministic, period-stratified sample supports editorial review "
        "of syntax, edition quality, and cleaning artifacts before another run.",
    ]
    for row in rows:
        sections.extend([
            "",
            f"## {row['poem_id']}",
            "",
            f"- Author: {row.get('author', '')}",
            f"- Period: {row.get('period', '')}",
            f"- Collection: {row.get('source_collection', '')}",
            f"- Cleaning notes: {row.get('cleaning_notes', '') or '(none)'}",
            f"- Audit notes: {row.get('audit_notes', '') or '(none)'}",
            "- Syntax/edition review: TODO acceptable / concern",
            "- Review notes: TODO",
            "",
            "```text",
            text_by_id[row["poem_id"]].rstrip(),
            "```",
        ])
    return "\n".join(sections) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
