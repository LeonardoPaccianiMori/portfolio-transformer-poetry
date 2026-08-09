"""Build lossless processed shards from resolved Biblioteca Italiana decisions."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, BinaryIO

from .biblioteca_italiana import ParsedBibItTEI, parse_bibit_tei
from .bibit_review_resolution import clean_bibit_editorial_brackets


RECORD_MANIFEST_FIELDS = (
    "object_id",
    "title",
    "authors",
    "source_archive",
    "source_url",
    "final_role",
    "cleaning_policy",
    "tei_sha256",
    "artifact_status",
    "shard_path",
    "byte_start",
    "byte_end",
    "cleaned_character_count",
    "cleaned_byte_count",
    "cleaned_sha256",
)

SONNET_MANIFEST_FIELDS = (
    "candidate_id",
    "object_id",
    "title",
    "author",
    "author_resolution",
    "periods",
    "source_archive",
    "source_url",
    "source_kind",
    "tei_type",
    "heading_path",
    "line_count",
    "stanza_pattern",
    "final_role",
    "cleaning_policy",
    "source_text_sha256",
    "shard_path",
    "byte_start",
    "byte_end",
    "cleaned_character_count",
    "cleaned_byte_count",
    "cleaned_line_count",
    "cleaned_sha256",
)

_RECORD_OUTPUT_ROLES = {
    "historical_general": "historical_general",
    "historical_non_sonnet_poetry": "historical_non_sonnet_poetry",
    "nineteenth_century_bridge": "nineteenth_century_bridge",
}
_STANDARD_SONNET_ROLES = {
    "sonnet_core_standard_14_line",
    "sonnet_core_inferred_14_line",
}
_CONDITIONED_SONNET_ROLE = "sonnet_variant_conditioned_auxiliary"
_CLEANING_POLICIES = {
    "preserve_rendered_tei_text",
    "strip_editorial_square_delimiters_and_labels",
}


@dataclass(frozen=True)
class BibItProcessedBuildConfig:
    """Inputs and bounded shard settings for the resolved BibIt build."""

    repo_root: Path
    record_decisions_path: Path
    sonnet_decisions_path: Path
    tei_cache_dir: Path
    output_dir: Path
    markdown_report_path: Path
    max_shard_bytes: int = 48 * 1024 * 1024
    progress_interval: int = 25


Progress = Callable[[str], None]


def build_bibit_processed_corpus(
    config: BibItProcessedBuildConfig,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Materialize all activated records and poems into deterministic text shards."""

    _validate_config(config)
    started = monotonic()
    record_rows = _read_csv(config.record_decisions_path)
    sonnet_rows = _read_csv(config.sonnet_decisions_path)
    active_records = sorted(
        (row for row in record_rows if row["decision"].startswith("activate_")),
        key=lambda row: row["object_id"],
    )
    active_sonnets = sorted(
        (row for row in sonnet_rows if row["decision"].startswith("activate_")),
        key=lambda row: row["candidate_id"],
    )
    _validate_decisions(active_records, active_sonnets)
    sonnets_by_object_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in active_sonnets:
        sonnets_by_object_id[row["object_id"]].append(row)

    output_parent = config.output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{config.output_dir.name}.", dir=output_parent)
    )
    final_prefix = _portable(config.output_dir, config.repo_root)
    writers = {
        role: _ShardWriter(
            temp_dir / directory,
            f"{final_prefix}/{directory}",
            max_shard_bytes=config.max_shard_bytes,
        )
        for role, directory in (
            ("historical_general", "historical_general"),
            ("historical_non_sonnet_poetry", "historical_non_sonnet_poetry"),
            ("nineteenth_century_bridge", "nineteenth_century_bridge"),
            ("standard_sonnets", "standard_sonnets"),
            ("conditioned_sonnet_variants", "conditioned_sonnet_variants"),
        )
    }
    record_manifest: list[dict[str, Any]] = []
    sonnet_manifest: list[dict[str, Any]] = []

    try:
        for index, row in enumerate(active_records, start=1):
            object_id = row["object_id"]
            cache_path = config.tei_cache_dir / f"{object_id}.xml"
            if not cache_path.is_file():
                raise FileNotFoundError(f"missing cached BibIt TEI: {cache_path}")
            xml = cache_path.read_bytes()
            actual_tei_sha256 = _sha256_bytes(xml)
            if actual_tei_sha256 != row["tei_sha256"]:
                raise ValueError(f"cached TEI hash mismatch: {object_id}")
            parsed = parse_bibit_tei(xml, object_id=object_id)
            raw_record_text = parsed.sonnet_candidate_safe_text
            expected_characters = int(row["included_characters"])
            if len(raw_record_text) != expected_characters:
                raise ValueError(
                    f"routed text length changed for {object_id}: "
                    f"expected {expected_characters}, found {len(raw_record_text)}"
                )
            cleaned_record_text = _clean_text(raw_record_text, row["cleaning_policy"])
            record_role = _RECORD_OUTPUT_ROLES[row["final_role"]]
            if cleaned_record_text.strip():
                artifact_status = "text_materialized"
                location = writers[record_role].add(object_id, cleaned_record_text)
            else:
                artifact_status = "sonnet_source_without_residual_record_text"
                location = {"shard_path": "", "byte_start": "", "byte_end": ""}
            record_manifest.append(
                {
                    "object_id": object_id,
                    "title": row["title"],
                    "authors": row["authors"],
                    "source_archive": "Biblioteca Italiana",
                    "source_url": row["landing_page_url"],
                    "final_role": row["final_role"],
                    "cleaning_policy": row["cleaning_policy"],
                    "tei_sha256": actual_tei_sha256,
                    "artifact_status": artifact_status,
                    **location,
                    **_text_stats(cleaned_record_text),
                }
            )

            candidate_text = _candidate_text_map(parsed)
            for sonnet_row in sonnets_by_object_id.get(object_id, []):
                candidate_id = sonnet_row["candidate_id"]
                unit_id = candidate_id.split(":", maxsplit=1)[1]
                if unit_id not in candidate_text:
                    raise ValueError(f"candidate unit missing from pinned TEI: {candidate_id}")
                raw_sonnet_text = candidate_text[unit_id]
                if _sha256_text(raw_sonnet_text) != sonnet_row["text_sha256"]:
                    raise ValueError(f"candidate text hash mismatch: {candidate_id}")
                if len(raw_sonnet_text) != int(sonnet_row["character_count"]):
                    raise ValueError(f"candidate text length changed: {candidate_id}")
                cleaned_sonnet_text = _clean_text(
                    raw_sonnet_text,
                    sonnet_row["cleaning_policy"],
                )
                output_role = (
                    "standard_sonnets"
                    if sonnet_row["final_role"] in _STANDARD_SONNET_ROLES
                    else "conditioned_sonnet_variants"
                )
                poem_location = writers[output_role].add(
                    candidate_id,
                    cleaned_sonnet_text,
                )
                sonnet_manifest.append(
                    {
                        "candidate_id": candidate_id,
                        "object_id": object_id,
                        "title": sonnet_row["title"],
                        "author": sonnet_row["candidate_author"],
                        "author_resolution": sonnet_row["author_resolution"],
                        "periods": sonnet_row["periods"],
                        "source_archive": "Biblioteca Italiana",
                        "source_url": sonnet_row["landing_page_url"],
                        "source_kind": sonnet_row["source_kind"],
                        "tei_type": sonnet_row["tei_type"],
                        "heading_path": sonnet_row["heading_path"],
                        "line_count": sonnet_row["line_count"],
                        "stanza_pattern": sonnet_row["stanza_pattern"],
                        "final_role": sonnet_row["final_role"],
                        "cleaning_policy": sonnet_row["cleaning_policy"],
                        "source_text_sha256": sonnet_row["text_sha256"],
                        **poem_location,
                        **_text_stats(cleaned_sonnet_text, include_lines=True),
                    }
                )

            if index == 1 or index % config.progress_interval == 0 or index == len(active_records):
                elapsed = monotonic() - started
                eta = elapsed / index * (len(active_records) - index)
                _report(
                    progress,
                    f"record {index:,}/{len(active_records):,} ({index / len(active_records):.1%}) "
                    f"id={object_id} poems={len(sonnet_manifest):,} "
                    f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                )

        missing_poems = set(row["candidate_id"] for row in active_sonnets) - set(
            row["candidate_id"] for row in sonnet_manifest
        )
        if missing_poems:
            raise ValueError(
                f"activated sonnets belong to inactive or missing records: {len(missing_poems)}"
            )
        shard_reports = {
            role: writer.close() for role, writer in writers.items()
        }
        record_manifest_path = temp_dir / "records_manifest.csv"
        sonnet_manifest_path = temp_dir / "sonnets_manifest.csv"
        _write_csv(record_manifest_path, RECORD_MANIFEST_FIELDS, record_manifest)
        _write_csv(sonnet_manifest_path, SONNET_MANIFEST_FIELDS, sonnet_manifest)
        report = _build_report(
            config,
            record_manifest=record_manifest,
            sonnet_manifest=sonnet_manifest,
            shard_reports=shard_reports,
            record_manifest_path=record_manifest_path,
            sonnet_manifest_path=sonnet_manifest_path,
        )
        _write_json(temp_dir / "build_report.json", report)
        _validate_shard_sizes(temp_dir, config.max_shard_bytes)

        if config.output_dir.exists():
            shutil.rmtree(config.output_dir)
        os.replace(temp_dir, config.output_dir)
    except Exception:
        for writer in writers.values():
            writer.abort()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_bibit_processed_build_markdown(report),
        encoding="utf-8",
    )
    return report


class _ShardWriter:
    def __init__(self, directory: Path, portable_directory: str, *, max_shard_bytes: int):
        self.directory = directory
        self.portable_directory = portable_directory
        self.max_shard_bytes = max_shard_bytes
        self._handle: BinaryIO | None = None
        self._path: Path | None = None
        self._portable_path = ""
        self._bytes = 0
        self._items = 0
        self._hasher = hashlib.sha256()
        self._reports: list[dict[str, Any]] = []

    def add(self, item_id: str, text: str) -> dict[str, Any]:
        if not text.strip():
            raise ValueError(f"cannot shard empty text: {item_id}")
        canonical_text = text if text.endswith("\n") else text + "\n"
        payload = canonical_text.encode("utf-8")
        if len(payload) > self.max_shard_bytes:
            raise ValueError(
                f"single item exceeds max shard bytes: {item_id} ({len(payload):,})"
            )
        separator_bytes = 1 if self._items else 0
        if self._handle is None or self._bytes + separator_bytes + len(payload) > self.max_shard_bytes:
            self._finish_shard()
            self._start_shard()
            separator_bytes = 0
        assert self._handle is not None
        if separator_bytes:
            self._write(b"\n")
        byte_start = self._bytes
        self._write(payload)
        self._items += 1
        return {
            "shard_path": self._portable_path,
            "byte_start": byte_start,
            "byte_end": self._bytes,
        }

    def close(self) -> list[dict[str, Any]]:
        self._finish_shard()
        return list(self._reports)

    def abort(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _start_shard(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        part = len(self._reports) + 1
        name = f"part-{part:04d}.txt"
        self._path = self.directory / name
        self._portable_path = f"{self.portable_directory}/{name}"
        self._handle = self._path.open("wb")
        self._bytes = 0
        self._items = 0
        self._hasher = hashlib.sha256()

    def _write(self, payload: bytes) -> None:
        assert self._handle is not None
        self._handle.write(payload)
        self._hasher.update(payload)
        self._bytes += len(payload)

    def _finish_shard(self) -> None:
        if self._handle is None or self._path is None:
            return
        self._handle.close()
        self._reports.append(
            {
                "path": self._portable_path,
                "item_count": self._items,
                "byte_count": self._bytes,
                "sha256": self._hasher.hexdigest(),
            }
        )
        self._handle = None
        self._path = None


def _validate_config(config: BibItProcessedBuildConfig) -> None:
    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    if config.max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")


def _validate_decisions(
    records: list[dict[str, str]],
    sonnets: list[dict[str, str]],
) -> None:
    if not records:
        raise ValueError("record decision CSV has no activated records")
    object_ids = [row["object_id"] for row in records]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("activated record decisions contain duplicate object IDs")
    unsupported_record_roles = sorted(
        {row["final_role"] for row in records} - set(_RECORD_OUTPUT_ROLES)
    )
    if unsupported_record_roles:
        raise ValueError(f"unsupported activated record roles: {unsupported_record_roles}")
    supported_sonnet_roles = _STANDARD_SONNET_ROLES | {_CONDITIONED_SONNET_ROLE}
    unsupported_sonnet_roles = sorted(
        {row["final_role"] for row in sonnets} - supported_sonnet_roles
    )
    if unsupported_sonnet_roles:
        raise ValueError(f"unsupported activated sonnet roles: {unsupported_sonnet_roles}")
    cleaning_policies = {
        row["cleaning_policy"] for row in (*records, *sonnets)
    }
    unsupported_cleaning = sorted(cleaning_policies - _CLEANING_POLICIES)
    if unsupported_cleaning:
        raise ValueError(f"unsupported BibIt cleaning policies: {unsupported_cleaning}")


def _candidate_text_map(parsed: ParsedBibItTEI) -> dict[str, str]:
    units = (
        *parsed.sonnets,
        *parsed.structural_sonnet_candidates,
        *parsed.structural_sonnet_variants,
    )
    return {unit.unit_id: unit.text for unit in units}


def _clean_text(text: str, cleaning_policy: str) -> str:
    if cleaning_policy == "preserve_rendered_tei_text":
        return text
    if cleaning_policy == "strip_editorial_square_delimiters_and_labels":
        return clean_bibit_editorial_brackets(text)
    raise ValueError(f"unsupported BibIt cleaning policy: {cleaning_policy}")


def _text_stats(text: str, *, include_lines: bool = False) -> dict[str, Any]:
    payload = text.encode("utf-8")
    stats: dict[str, Any] = {
        "cleaned_character_count": len(text),
        "cleaned_byte_count": len(payload),
        "cleaned_sha256": _sha256_bytes(payload),
    }
    if include_lines:
        stats["cleaned_line_count"] = sum(bool(line.strip()) for line in text.splitlines())
    return stats


def _build_report(
    config: BibItProcessedBuildConfig,
    *,
    record_manifest: list[dict[str, Any]],
    sonnet_manifest: list[dict[str, Any]],
    shard_reports: dict[str, list[dict[str, Any]]],
    record_manifest_path: Path,
    sonnet_manifest_path: Path,
) -> dict[str, Any]:
    record_counts = Counter(row["final_role"] for row in record_manifest)
    record_text_counts = Counter(
        row["final_role"]
        for row in record_manifest
        if row["artifact_status"] == "text_materialized"
    )
    record_characters = Counter()
    for row in record_manifest:
        record_characters[row["final_role"]] += int(row["cleaned_character_count"])
    sonnet_counts = Counter(row["final_role"] for row in sonnet_manifest)
    sonnet_characters = Counter()
    for row in sonnet_manifest:
        sonnet_characters[row["final_role"]] += int(row["cleaned_character_count"])
    return {
        "build_version": "bibit_resolved_v1",
        "inputs": {
            "record_decisions_path": _portable(config.record_decisions_path, config.repo_root),
            "record_decisions_sha256": _sha256_file(config.record_decisions_path),
            "sonnet_decisions_path": _portable(config.sonnet_decisions_path, config.repo_root),
            "sonnet_decisions_sha256": _sha256_file(config.sonnet_decisions_path),
            "tei_cache_path": _portable(config.tei_cache_dir, config.repo_root),
        },
        "outputs": {
            "output_dir": _portable(config.output_dir, config.repo_root),
            "record_manifest_path": f"{_portable(config.output_dir, config.repo_root)}/records_manifest.csv",
            "record_manifest_sha256": _sha256_file(record_manifest_path),
            "sonnet_manifest_path": f"{_portable(config.output_dir, config.repo_root)}/sonnets_manifest.csv",
            "sonnet_manifest_sha256": _sha256_file(sonnet_manifest_path),
            "markdown_report_path": _portable(config.markdown_report_path, config.repo_root),
        },
        "max_shard_bytes": config.max_shard_bytes,
        "record_count": len(record_manifest),
        "record_text_count": sum(record_text_counts.values()),
        "empty_record_text_count": len(record_manifest) - sum(record_text_counts.values()),
        "record_counts_by_role": dict(sorted(record_counts.items())),
        "record_text_counts_by_role": dict(sorted(record_text_counts.items())),
        "record_characters_by_role": dict(sorted(record_characters.items())),
        "sonnet_count": len(sonnet_manifest),
        "sonnet_counts_by_role": dict(sorted(sonnet_counts.items())),
        "sonnet_characters_by_role": dict(sorted(sonnet_characters.items())),
        "shards": shard_reports,
        "policy": {
            "text_storage": "plain_utf8_shards_with_manifest_byte_ranges",
            "source_spelling_and_punctuation_preserved": True,
            "activated_sonnets_quarantined_from_record_text": True,
            "all_audited_sonnet_candidates_quarantined_from_record_text": True,
            "held_out_v6_identities_excluded": True,
            "v7_split_assigned": False,
            "training_mixture_weight_assigned": False,
        },
    }


def render_bibit_processed_build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Biblioteca Italiana Resolved Corpus Build",
        "",
        "## Result",
        "",
        (
            f"Built text for {report['record_text_count']:,} of "
            f"{report['record_count']:,} activated source records and "
            f"{report['sonnet_count']:,} activated sonnets from the pinned TEI cache."
        ),
        "",
        "The text is stored once in bounded UTF-8 shards. The manifests record exact",
        "byte ranges and SHA-256 hashes, so every source or poem can be recovered and",
        "verified independently without creating tens of thousands of tiny files.",
        "",
        "## Record Roles",
        "",
        "| Role | Text records | Characters |",
        "| --- | ---: | ---: |",
    ]
    for role, count in report["record_counts_by_role"].items():
        lines.append(
            f"| `{role}` | {report['record_text_counts_by_role'].get(role, 0):,} | "
            f"{report['record_characters_by_role'][role]:,} |"
        )
    lines.extend(
        [
            "",
            "## Sonnet Roles",
            "",
            "| Role | Sonnets | Characters |",
            "| --- | ---: | ---: |",
        ]
    )
    for role, count in report["sonnet_counts_by_role"].items():
        lines.append(
            f"| `{role}` | {count:,} | {report['sonnet_characters_by_role'][role]:,} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- No V7 train/validation/test assignment is made by this build.",
            "- No final training-mixture weight is authorized by this build.",
            "- All audited sonnet candidates remain quarantined from record text.",
            "- V6 validation/test identity conflicts remain excluded.",
            (
                f"- {report['empty_record_text_count']:,} activated sonnet-source records "
                "have no residual record text after poem quarantine."
            ),
            "",
            "## Artifacts",
            "",
            f"- Record manifest: `{report['outputs']['record_manifest_path']}`",
            f"- Sonnet manifest: `{report['outputs']['sonnet_manifest_path']}`",
            f"- Machine-readable report: `{report['outputs']['output_dir']}/build_report.json`",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_shard_sizes(output_dir: Path, max_shard_bytes: int) -> None:
    oversized = [
        path for path in output_dir.rglob("part-*.txt") if path.stat().st_size > max_shard_bytes
    ]
    if oversized:
        raise ValueError(f"build created oversized shards: {oversized}")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
