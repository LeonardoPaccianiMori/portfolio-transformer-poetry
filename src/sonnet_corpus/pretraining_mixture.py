"""Publish reviewed pretraining components and assemble deterministic mixtures."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PretrainingComponent:
    """One published component that contributes source files to a mixture."""

    component_id: str
    processed_dir: Path
    report_path: Path


@dataclass(frozen=True)
class PretrainingMixtureConfig:
    """Approved composition policy for one combined pretraining corpus."""

    corpus_version: str
    components: tuple[PretrainingComponent, ...]
    processed_dir: Path
    report_path: Path
    markdown_report_path: Path
    work_cap_warning: float
    author_cap_warning: float
    approved_work_cap_exceptions: tuple[str, ...] = ()
    approved_author_cap_exceptions: tuple[str, ...] = ()


def load_pretraining_mixture_config(path: Path) -> PretrainingMixtureConfig:
    """Load a checked, repository-relative mixture configuration."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    components = tuple(
        PretrainingComponent(
            component_id=str(component["component_id"]),
            processed_dir=Path(component["processed_dir"]),
            report_path=Path(component["report_path"]),
        )
        for component in payload["components"]
    )
    return PretrainingMixtureConfig(
        corpus_version=str(payload["corpus_version"]),
        components=components,
        processed_dir=Path(payload["processed_dir"]),
        report_path=Path(payload["report_path"]),
        markdown_report_path=Path(payload["markdown_report_path"]),
        work_cap_warning=float(payload["work_cap_warning"]),
        author_cap_warning=float(payload["author_cap_warning"]),
        approved_work_cap_exceptions=tuple(payload.get("approved_work_cap_exceptions", [])),
        approved_author_cap_exceptions=tuple(
            payload.get("approved_author_cap_exceptions", [])
        ),
    )


def publish_pretraining_component(
    *,
    source_processed_dir: Path,
    source_report_path: Path,
    target_processed_dir: Path,
    target_report_path: Path,
) -> dict[str, object]:
    """Copy one reviewed component into public data and rewrite report paths."""

    report = _read_report(source_report_path)
    source_records = _read_source_records(report, source_processed_dir)
    stage_dir = target_processed_dir.parent / f".{target_processed_dir.name}.stage"
    _replace_directory(stage_dir)
    stage_sources = stage_dir / "sources"
    stage_sources.mkdir(parents=True)

    for source, source_path in source_records:
        shutil.copy2(source_path, stage_sources / f"{source['source_id']}.txt")
    shutil.copy2(source_processed_dir / "corpus.txt", stage_dir / "corpus.txt")

    rewritten_report = _rewrite_component_report_paths(
        report,
        target_processed_dir=target_processed_dir,
    )
    staged_report_path = stage_dir / target_report_path.name
    _write_json(rewritten_report, staged_report_path)
    _publish_directory(stage_dir, target_processed_dir)
    _write_json(rewritten_report, target_report_path)
    return rewritten_report


def build_pretraining_mixture(config: PretrainingMixtureConfig) -> dict[str, object]:
    """Join approved source files once each and write composition evidence."""

    _validate_config(config)
    source_entries: list[dict[str, object]] = []
    seen_source_ids: set[str] = set()
    corpus_parts: list[str] = []

    for component in config.components:
        report = _read_report(component.report_path)
        for source, source_path in _read_source_records(report, component.processed_dir):
            source_id = str(source["source_id"])
            if source_id in seen_source_ids:
                raise ValueError(f"pretraining mixture contains duplicate source ID: {source_id}")
            seen_source_ids.add(source_id)
            text = source_path.read_text(encoding="utf-8")
            corpus_parts.append(text.strip() + "\n")
            source_entries.append(
                {
                    "component_id": component.component_id,
                    "source_id": source_id,
                    "title": str(source["title"]),
                    "author": str(source["author"]),
                    "source_archive": str(source["source_archive"]),
                    "source_path": str(source_path),
                    "cleaned_character_count": int(source["cleaned_character_count"]),
                    "cleaned_word_count": int(source["cleaned_word_count"]),
                    "sampling_weight": 1.0,
                }
            )

    total_characters = sum(int(source["cleaned_character_count"]) for source in source_entries)
    total_words = sum(int(source["cleaned_word_count"]) for source in source_entries)
    for source in source_entries:
        source["character_share"] = int(source["cleaned_character_count"]) / total_characters
        source["word_share"] = int(source["cleaned_word_count"]) / total_words

    author_entries = _author_entries(source_entries, total_characters, total_words)
    report = {
        "corpus_version": config.corpus_version,
        "components": [
            {
                "component_id": component.component_id,
                "processed_dir": str(component.processed_dir),
                "report_path": str(component.report_path),
            }
            for component in config.components
        ],
        "sampling_policy": {
            "strategy": "one_pass_concatenation",
            "default_source_weight": 1.0,
            "cap_enforcement": "none",
            "work_cap_warning": config.work_cap_warning,
            "author_cap_warning": config.author_cap_warning,
            "approved_work_cap_exceptions": list(config.approved_work_cap_exceptions),
            "approved_author_cap_exceptions": list(config.approved_author_cap_exceptions),
        },
        "source_count": len(source_entries),
        "total_cleaned_characters": total_characters,
        "total_cleaned_words": total_words,
        "sources": source_entries,
        "authors": author_entries,
        "concentration_warnings": _concentration_warnings(config, source_entries, author_entries),
    }

    stage_dir = config.processed_dir.parent / f".{config.processed_dir.name}.stage"
    _replace_directory(stage_dir)
    (stage_dir / "corpus.txt").write_text("\n".join(corpus_parts), encoding="utf-8")
    _write_json(report, stage_dir / config.report_path.name)
    _publish_directory(stage_dir, config.processed_dir)
    _write_json(report, config.report_path)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(_render_markdown_report(report), encoding="utf-8")
    return report


def _read_report(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report.get("sources"), list) or not report["sources"]:
        raise ValueError(f"pretraining component report has no sources: {path}")
    return report


def _read_source_records(
    report: dict[str, object], processed_dir: Path
) -> list[tuple[dict[str, object], Path]]:
    records: list[tuple[dict[str, object], Path]] = []
    for source in report["sources"]:
        if not isinstance(source, dict):
            raise ValueError("pretraining component report has an invalid source record")
        source_id = str(source["source_id"])
        source_path = processed_dir / "sources" / f"{source_id}.txt"
        if not source_path.is_file():
            raise ValueError(f"pretraining component source file is missing: {source_path}")
        text_length = len(source_path.read_text(encoding="utf-8"))
        if text_length != int(source["cleaned_character_count"]):
            raise ValueError(
                "pretraining component source size does not match report: " f"{source_id}"
            )
        records.append((source, source_path))
    return records


def _rewrite_component_report_paths(
    report: dict[str, object], *, target_processed_dir: Path
) -> dict[str, object]:
    rewritten = json.loads(json.dumps(report))
    rewritten["processed_dir"] = str(target_processed_dir)
    rewritten["combined_corpus_path"] = str(target_processed_dir / "corpus.txt")
    for source in rewritten["sources"]:
        source["processed_path"] = str(
            target_processed_dir / "sources" / f"{source['source_id']}.txt"
        )
    return rewritten


def _author_entries(
    sources: list[dict[str, object]], total_characters: int, total_words: int
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"characters": 0, "words": 0})
    for source in sources:
        grouped[str(source["author"])]["characters"] += int(source["cleaned_character_count"])
        grouped[str(source["author"])]["words"] += int(source["cleaned_word_count"])
    return [
        {
            "author": author,
            "cleaned_character_count": counts["characters"],
            "cleaned_word_count": counts["words"],
            "character_share": counts["characters"] / total_characters,
            "word_share": counts["words"] / total_words,
        }
        for author, counts in sorted(
            grouped.items(), key=lambda item: item[1]["characters"], reverse=True
        )
    ]


def _concentration_warnings(
    config: PretrainingMixtureConfig,
    sources: list[dict[str, object]],
    authors: list[dict[str, object]],
) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    for source in sources:
        if float(source["character_share"]) > config.work_cap_warning:
            warnings.append(
                {
                    "level": "work",
                    "name": source["source_id"],
                    "character_share": source["character_share"],
                    "approved_exception": source["source_id"]
                    in config.approved_work_cap_exceptions,
                }
            )
    for author in authors:
        if float(author["character_share"]) > config.author_cap_warning:
            warnings.append(
                {
                    "level": "author",
                    "name": author["author"],
                    "character_share": author["character_share"],
                    "approved_exception": author["author"]
                    in config.approved_author_cap_exceptions,
                }
            )
    return warnings


def _render_markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Pretraining Historical Italian V2 Composition",
        "",
        "This corpus concatenates every approved source exactly once. It applies no",
        "synthetic resampling or hard cap; concentration thresholds are reported for",
        "review only.",
        "",
        "## Scale",
        "",
        "| Measurement | Value |",
        "| --- | ---: |",
        f"| Sources | {report['source_count']} |",
        f"| Cleaned characters | {report['total_cleaned_characters']:,} |",
        f"| Cleaned words | {report['total_cleaned_words']:,} |",
        "",
        "## Largest Authors",
        "",
        "| Author | Characters | Share |",
        "| --- | ---: | ---: |",
    ]
    for author in report["authors"][:10]:
        lines.append(
            f"| {author['author']} | {author['cleaned_character_count']:,} | "
            f"{author['character_share']:.2%} |"
        )
    lines.extend(["", "## Concentration Warnings", ""])
    warnings = report["concentration_warnings"]
    if not warnings:
        lines.append("No source or author exceeds the recorded warning thresholds.")
    else:
        lines.extend(["| Level | Name | Share | Approved exception |", "| --- | --- | ---: | --- |"])
        for warning in warnings:
            lines.append(
                f"| {warning['level']} | {warning['name']} | "
                f"{warning['character_share']:.2%} | {warning['approved_exception']} |"
            )
    return "\n".join(lines) + "\n"


def _validate_config(config: PretrainingMixtureConfig) -> None:
    if not config.components:
        raise ValueError("pretraining mixture needs at least one component")
    component_ids = [component.component_id for component in config.components]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("pretraining mixture has duplicate component IDs")
    for value, label in (
        (config.work_cap_warning, "work_cap_warning"),
        (config.author_cap_warning, "author_cap_warning"),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"{label} must be between zero and one")


def _replace_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _publish_directory(stage_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(stage_dir), str(target_dir))


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
