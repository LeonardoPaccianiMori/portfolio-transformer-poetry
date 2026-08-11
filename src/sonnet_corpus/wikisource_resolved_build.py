"""Build inactive, role-specific Wikisource shards from frozen 4D decisions."""

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

from .gutenberg_fulltext_probe import (
    _normalized_words,
    _rolling_shingle_hashes,
    fingerprint_text,
    measure_word_shingle_containment,
)
from .wikisource_page_extraction import _discover_cross_pairs, _discover_pairs
from .wikisource_review_resolution import (
    RIGHTS_FIELDS,
    _clean_verse_line,
    _load_protected_sonnets,
    _load_text_references,
)


RECORD_MANIFEST_FIELDS = (
    "work_root_id", "root_title", "source_archive", "source_url", "author_evidence",
    "period_bucket", "input_role", "final_role", "direct_scan_title", "scan_rights_id",
    "final_decision", "resolution_reason", "canonical_reference_ids", "removed_reference_ids",
    "rights_decision", "activation_status", "artifact_status", "source_cache_path",
    "source_sha256", "source_character_count", "retained_source_character_count",
    "excluded_source_character_count", "shard_path", "byte_start", "byte_end",
    "cleaned_character_count", "cleaned_byte_count", "cleaned_sha256",
)

SEGMENT_MANIFEST_FIELDS = (
    "segment_id", "work_root_id", "source_sha256", "character_start", "character_end",
    "character_count", "segment_sha256", "segment_decision", "final_role", "reason",
    "reference_ids", "activation_status", "artifact_status", "output_shard_path",
    "output_byte_start", "output_byte_end", "output_sha256",
)

SONNET_MANIFEST_FIELDS = (
    "candidate_id", "work_root_id", "root_title", "source_record_author", "poem_author",
    "poem_author_resolution", "period_bucket", "source_url", "source_scan_title",
    "source_kind", "stanza_pattern", "line_count", "first_line", "last_line",
    "character_start", "character_end", "source_text_sha256", "cleaned_text_sha256",
    "exact_reference_ids", "near_reference_ids", "protected_v6_reference_ids",
    "candidate_decision", "final_role", "activation_status", "artifact_status",
    "shard_path", "byte_start", "byte_end", "cleaned_character_count",
    "cleaned_byte_count", "cleaned_sha256",
)

ATTRIBUTION_MANIFEST_FIELDS = (
    "work_root_id", "root_title", "source_url", "source_history_url", "scan_rights_id",
) + RIGHTS_FIELDS[1:]

_ROLES = {
    "historical_general", "historical_non_sonnet_poetry", "nineteenth_century_bridge"
}
Progress = Callable[[str], None]


@dataclass(frozen=True)
class WikisourceResolvedBuildConfig:
    repo_root: Path
    root_decisions_path: Path
    segment_decisions_path: Path
    sonnet_decisions_path: Path
    scan_rights_path: Path
    review_report_path: Path
    output_dir: Path
    markdown_report_path: Path
    bibit_record_manifest_path: Path
    broader_sources_manifest_path: Path
    gutenberg_previous_probe_path: Path
    gutenberg_previous_cache_dir: Path
    gutenberg_pass_1b_probe_path: Path
    gutenberg_pass_1b_cache_dir: Path
    gutenberg_resolved_record_manifest_path: Path
    protected_sonnet_manifest_path: Path
    max_shard_bytes: int = 64 * 1024 * 1024
    near_duplicate_threshold: float = 0.8
    progress_interval: int = 100


def build_wikisource_resolved_corpus(
    config: WikisourceResolvedBuildConfig,
    *, progress: Progress | None = None,
) -> dict[str, Any]:
    """Materialize only frozen rights-cleared 4D records and sonnets, inactive."""

    _validate_inputs(config)
    started = monotonic()
    roots = _read_csv(config.root_decisions_path)
    segments = _read_csv(config.segment_decisions_path)
    sonnets = _read_csv(config.sonnet_decisions_path)
    rights = _read_csv(config.scan_rights_path)
    rights_by_id = _unique(rights, "scan_rights_id")
    segments_by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    sonnets_by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in segments:
        segments_by_root[row["work_root_id"]].append(row)
    for row in sonnets:
        sonnets_by_root[row["work_root_id"]].append(row)

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{config.output_dir.name}.", dir=config.output_dir.parent))
    final_prefix = config.output_dir.relative_to(config.repo_root).as_posix()
    writers = {
        role: _ShardWriter(temp_dir / role, f"{final_prefix}/{role}", config.max_shard_bytes)
        for role in (*sorted(_ROLES), "standard_sonnets")
    }
    record_manifest = []
    segment_manifest = []
    sonnet_manifest = []
    attribution_manifest = []
    materialized_texts: dict[str, str] = {}
    try:
        for index, root in enumerate(sorted(roots, key=lambda row: int(row["work_root_id"].split(":")[-1])), start=1):
            root_id = root["work_root_id"]
            source = _read_source(config, root)
            output_segments = [dict(row) for row in sorted(
                segments_by_root.get(root_id, []), key=lambda row: int(row["character_start"])
            )]
            selected = [row for row in output_segments if row["segment_decision"] == "include_broader_text"]
            record_text, ranges = _compose_segments(source, selected)
            location: dict[str, Any] = {"shard_path": "", "byte_start": "", "byte_end": ""}
            artifact_status = "not_materialized_excluded_or_sonnet_only"
            final_role = root["final_broader_role"]
            if root["final_decision"] == "eligible_inactive_processed_build" and record_text.strip():
                if final_role not in _ROLES:
                    raise ValueError(f"unsupported Wikisource role: {final_role}")
                record_text = _canonical(record_text)
                location = writers[final_role].add(root_id, record_text)
                artifact_status = "text_materialized_inactive"
                materialized_texts[root_id] = record_text
                for segment, relative_start, relative_end in ranges:
                    segment["artifact_status"] = "materialized_in_inactive_record"
                    segment["output_shard_path"] = location["shard_path"]
                    segment["output_byte_start"] = int(location["byte_start"]) + relative_start
                    segment["output_byte_end"] = int(location["byte_start"]) + relative_end
                    segment["output_sha256"] = segment["segment_sha256"]
            for segment in output_segments:
                segment.setdefault("artifact_status", "not_materialized")
                segment.setdefault("output_shard_path", "")
                segment.setdefault("output_byte_start", "")
                segment.setdefault("output_byte_end", "")
                segment.setdefault("output_sha256", "")
            segment_manifest.extend(output_segments)

            for sonnet in sorted(sonnets_by_root.get(root_id, []), key=lambda row: int(row["character_start"])):
                sonnet_manifest.append(_materialize_sonnet(sonnet, source, writers["standard_sonnets"]))

            stats = _stats(record_text)
            record_manifest.append({
                "work_root_id": root_id, "root_title": root["root_title"],
                "source_archive": "Italian Wikisource", "source_url": root["landing_page_url"],
                "author_evidence": root["author_evidence"], "period_bucket": root["period_bucket"],
                "input_role": root["input_role"], "final_role": final_role,
                "direct_scan_title": root["direct_scan_title"], "scan_rights_id": root["scan_rights_id"],
                "final_decision": root["final_decision"], "resolution_reason": root["resolution_reason"],
                "canonical_reference_ids": root["canonical_reference_ids"],
                "removed_reference_ids": root["removed_reference_ids"], "rights_decision": root["rights_decision"],
                "activation_status": "inactive_pending_cross_archive_freeze", "artifact_status": artifact_status,
                "source_cache_path": root["source_cache_path"], "source_sha256": root["source_sha256"],
                "source_character_count": root["source_character_count"],
                "retained_source_character_count": root["retained_broader_character_count"],
                "excluded_source_character_count": root["excluded_character_count"],
                **location, **stats,
            })
            if artifact_status == "text_materialized_inactive" or any(
                row["artifact_status"] == "sonnet_materialized_inactive" for row in sonnet_manifest[-len(sonnets_by_root.get(root_id, [])):] if sonnets_by_root.get(root_id)
            ):
                attribution_manifest.append(_attribution(root, rights_by_id[root["scan_rights_id"]]))
            if progress and (index == 1 or index % config.progress_interval == 0 or index == len(roots)):
                _progress(progress, "build", index, len(roots), started)

        shard_reports = {role: writer.close() for role, writer in writers.items()}
        verification = _verify_final(config, materialized_texts, progress=progress)
        _write_csv(temp_dir / "records_manifest.csv", RECORD_MANIFEST_FIELDS, record_manifest)
        _write_csv(temp_dir / "segments_manifest.csv", SEGMENT_MANIFEST_FIELDS, segment_manifest)
        _write_csv(temp_dir / "sonnets_manifest.csv", SONNET_MANIFEST_FIELDS, sonnet_manifest)
        _write_csv(temp_dir / "attribution_manifest.csv", ATTRIBUTION_MANIFEST_FIELDS, attribution_manifest)
        report = _report(
            config, record_manifest, segment_manifest, sonnet_manifest,
            attribution_manifest, shard_reports, verification, temp_dir,
        )
        _write_json(temp_dir / "build_report.json", report)
        _validate_artifacts(temp_dir, final_prefix, record_manifest, segment_manifest, sonnet_manifest)
        _replace_verified_output(temp_dir, config.output_dir)
    except BaseException:
        for writer in writers.values():
            writer.abort()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    return (
        "# Italian Wikisource Resolved Corpus Build\n\n"
        "## Result\n\n"
        f"The deterministic inactive build materializes {report['materialized_record_count']:,} broader records "
        f"and {report['materialized_sonnet_count']:,} verified sonnet candidates.\n\n"
        f"- Retained broader characters: {report['materialized_broader_character_count']:,}.\n"
        f"- Standard-sonnet characters: {report['materialized_sonnet_character_count']:,}.\n"
        f"- Shards: {report['shard_count']:,}.\n"
        f"- Attribution rows: {report['attribution_count']:,}.\n"
        "- Final exact, near, cross-corpus, and protected-V6 checks pass.\n\n"
        "## Boundary\n\n"
        "All shards are inactive pending checkpoint 7 cross-archive canonicalization and checkpoint 8 V7/mixture freeze. "
        "No conditioned material, V7 split, mixture weight, cache deletion, or GPU work is included.\n"
    )


class _ShardWriter:
    def __init__(self, directory: Path, portable: str, maximum: int) -> None:
        self.directory, self.portable, self.maximum = directory, portable, maximum
        self.handle: BinaryIO | None = None
        self.path: Path | None = None
        self.bytes = self.items = 0
        self.hasher = hashlib.sha256()
        self.reports: list[dict[str, Any]] = []

    def add(self, item_id: str, text: str) -> dict[str, Any]:
        payload = _canonical(text).encode("utf-8")
        if not payload.strip() or len(payload) > self.maximum:
            raise ValueError(f"invalid shard item {item_id}: {len(payload):,} bytes")
        separator = 1 if self.items else 0
        if self.handle is None or self.bytes + separator + len(payload) > self.maximum:
            self._finish(); self._start(); separator = 0
        if separator:
            self._write(b"\n")
        start = self.bytes
        self._write(payload); self.items += 1
        return {"shard_path": f"{self.portable}/{self.path.name}", "byte_start": start, "byte_end": self.bytes}

    def close(self) -> list[dict[str, Any]]:
        self._finish(); return self.reports

    def abort(self) -> None:
        if self.handle:
            self.handle.close(); self.handle = None

    def _start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"part-{len(self.reports)+1:04d}.txt"
        self.handle = self.path.open("wb"); self.bytes = self.items = 0; self.hasher = hashlib.sha256()

    def _write(self, payload: bytes) -> None:
        assert self.handle is not None
        self.handle.write(payload); self.hasher.update(payload); self.bytes += len(payload)

    def _finish(self) -> None:
        if self.handle is None or self.path is None:
            return
        self.handle.close()
        self.reports.append({"path": f"{self.portable}/{self.path.name}", "item_count": self.items, "byte_count": self.bytes, "sha256": self.hasher.hexdigest()})
        self.handle = None; self.path = None


def _read_source(config: WikisourceResolvedBuildConfig, root: dict[str, str]) -> str:
    path = config.repo_root / root["source_cache_path"]
    try:
        path.resolve().relative_to(config.repo_root.resolve())
    except ValueError as error:
        raise ValueError(f"source cache escapes repository: {path}") from error
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    if _sha(text) != root["source_sha256"] or len(text) != int(root["source_character_count"]):
        raise ValueError(f"source cache mismatch: {root['work_root_id']}")
    return text


def _compose_segments(text: str, rows: list[dict[str, str]]) -> tuple[str, list[tuple[dict[str, str], int, int]]]:
    payload = bytearray(); ranges = []; parts = []
    for index, row in enumerate(rows):
        start, end = int(row["character_start"]), int(row["character_end"])
        part = text[start:end]
        if _sha(part) != row["segment_sha256"]:
            raise ValueError(f"segment hash mismatch: {row['segment_id']}")
        if index:
            payload.extend(b"\n"); parts.append("\n")
        relative_start = len(payload); encoded = part.encode("utf-8"); payload.extend(encoded)
        ranges.append((row, relative_start, len(payload))); parts.append(part)
    result = "".join(parts)
    if result.encode("utf-8") != payload:
        raise AssertionError("segment byte composition changed")
    return result, ranges


def _materialize_sonnet(row: dict[str, str], source: str, writer: _ShardWriter) -> dict[str, Any]:
    result = dict(row)
    start, end = int(row["character_start"]), int(row["character_end"])
    raw = source[start:end]
    if _sha(raw) != row["source_text_sha256"]:
        raise ValueError(f"sonnet source hash mismatch: {row['candidate_id']}")
    cleaned = "\n".join(_clean_verse_line(line) for line in raw.splitlines() if line.strip()).strip() + "\n"
    if _sha(cleaned) != row["cleaned_text_sha256"] or len(cleaned.strip().splitlines()) != 14:
        raise ValueError(f"sonnet cleaning mismatch: {row['candidate_id']}")
    if row["candidate_decision"] == "eligible_standard_sonnet_inactive_pending_v7":
        location = writer.add(row["candidate_id"], cleaned)
        result.update({"artifact_status": "sonnet_materialized_inactive", **location, **_stats(cleaned)})
    else:
        result.update({"artifact_status": "not_materialized_duplicate_or_protected", "shard_path": "", "byte_start": "", "byte_end": "", **_stats("")})
    return result


def _attribution(root: dict[str, str], rights: dict[str, str]) -> dict[str, Any]:
    return {
        "work_root_id": root["work_root_id"], "root_title": root["root_title"],
        "source_url": root["landing_page_url"], "source_history_url": root["landing_page_url"] + "?action=history",
        "scan_rights_id": root["scan_rights_id"],
        **{field: rights[field] for field in RIGHTS_FIELDS[1:]},
    }


def _verify_final(config: WikisourceResolvedBuildConfig, texts: dict[str, str], *, progress: Progress | None) -> dict[str, Any]:
    fingerprints = {key: fingerprint_text(text)[0] for key, text in texts.items()}
    exact = Counter(value.normalized_word_sha256 for value in fingerprints.values() if value.word_count)
    if any(count > 1 for count in exact.values()):
        raise ValueError("final Wikisource records contain exact duplicates")
    internal_checked = 0
    for left, right in _discover_pairs(fingerprints):
        internal_checked += 1
        if measure_word_shingle_containment(texts[left], texts[right])["containment"] >= config.near_duplicate_threshold:
            raise ValueError(f"final internal near duplicate: {left} / {right}")
    references = _load_text_references(config)
    reference_fingerprints = {key: fingerprint_text(value.read_text())[0] for key, value in references.items()}
    cross_checked = 0
    for root_id, reference_id in _discover_cross_pairs(fingerprints, reference_fingerprints):
        cross_checked += 1
        if measure_word_shingle_containment(texts[root_id], references[reference_id].read_text())["containment"] >= config.near_duplicate_threshold:
            raise ValueError(f"final cross-corpus near duplicate: {root_id} / {reference_id}")
    protected = _load_protected_sonnets(config)
    watch: dict[int, list[str]] = defaultdict(list)
    denominators = {}
    for poem_id, poem in protected.items():
        hashes = set(_rolling_shingle_hashes(_normalized_words(poem)))
        if not hashes:
            continue
        denominators[poem_id] = len(hashes)
        for value in hashes:
            watch[value].append(poem_id)
    frozen_watch = {value: tuple(ids) for value, ids in watch.items()}
    protected_checked = 0
    protected_started = monotonic()
    for index, (root_id, text) in enumerate(sorted(texts.items()), start=1):
        _fingerprint, hits = fingerprint_text(text, watched_shingles=frozen_watch)
        protected_checked += len(hits)
        for poem_id, values in hits.items():
            if len(values) / denominators[poem_id] >= config.near_duplicate_threshold:
                raise ValueError(f"protected V6 overlap remains: {root_id} / {poem_id}")
        if progress and (index == 1 or index % config.progress_interval == 0 or index == len(texts)):
            _progress(progress, "protected-verification", index, len(texts), protected_started)
    return {
        "internal_candidate_pairs_checked": internal_checked,
        "cross_candidate_pairs_checked": cross_checked,
        "protected_candidate_pairs_checked": protected_checked,
        "exact_duplicate_count": 0, "near_duplicate_count": 0,
        "cross_duplicate_count": 0, "protected_overlap_count": 0,
    }


def _validate_inputs(config: WikisourceResolvedBuildConfig) -> None:
    required = (config.root_decisions_path, config.segment_decisions_path, config.sonnet_decisions_path, config.scan_rights_path, config.review_report_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen 4D build inputs: {missing}")
    report = json.loads(config.review_report_path.read_text(encoding="utf-8"))
    paths = {
        "roots": config.root_decisions_path, "segments": config.segment_decisions_path,
        "sonnets": config.sonnet_decisions_path, "scan_rights": config.scan_rights_path,
    }
    for key, path in paths.items():
        if _sha_file(path) != report["output_sha256"][key]:
            raise ValueError(f"stale or modified 4D ledger: {key}")
    if config.max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")


def _validate_artifacts(temp: Path, prefix: str, records: list[dict[str, Any]], segments: list[dict[str, Any]], sonnets: list[dict[str, Any]]) -> None:
    cache: dict[Path, bytes] = {}
    for row in [*records, *sonnets]:
        if not row["shard_path"]:
            continue
        relative = Path(row["shard_path"]).relative_to(prefix)
        payload = cache.setdefault(relative, (temp / relative).read_bytes())
        part = payload[int(row["byte_start"]):int(row["byte_end"])]
        if hashlib.sha256(part).hexdigest() != row["cleaned_sha256"]:
            raise ValueError(f"manifest artifact mismatch: {row.get('work_root_id') or row.get('candidate_id')}")
    for row in segments:
        if row["artifact_status"] != "materialized_in_inactive_record":
            continue
        relative = Path(row["output_shard_path"]).relative_to(prefix)
        payload = cache.setdefault(relative, (temp / relative).read_bytes())
        part = payload[int(row["output_byte_start"]):int(row["output_byte_end"])]
        if hashlib.sha256(part).hexdigest() != row["output_sha256"]:
            raise ValueError(f"segment artifact mismatch: {row['segment_id']}")


def _report(config: WikisourceResolvedBuildConfig, records: list[dict[str, Any]], segments: list[dict[str, Any]], sonnets: list[dict[str, Any]], attribution: list[dict[str, Any]], shards: dict[str, list[dict[str, Any]]], verification: dict[str, Any], temp: Path) -> dict[str, Any]:
    materialized = [row for row in records if row["artifact_status"] == "text_materialized_inactive"]
    expected_materialized = [
        row for row in records
        if row["final_decision"] == "eligible_inactive_processed_build"
    ]
    if len(materialized) != len(expected_materialized):
        raise ValueError(
            "resolved broader-root count does not match materialized record count: "
            f"{len(expected_materialized)} != {len(materialized)}"
        )
    poems = [row for row in sonnets if row["artifact_status"] == "sonnet_materialized_inactive"]
    manifest_names = ("records_manifest.csv", "segments_manifest.csv", "sonnets_manifest.csv", "attribution_manifest.csv")
    return {
        "checkpoint": "4D-processed-build", "root_count": len(records),
        "materialized_record_count": len(materialized), "sonnet_candidate_count": len(sonnets),
        "materialized_sonnet_count": len(poems), "attribution_count": len(attribution),
        "materialized_broader_character_count": sum(int(row["cleaned_character_count"]) for row in materialized),
        "materialized_sonnet_character_count": sum(int(row["cleaned_character_count"]) for row in poems),
        "materialized_role_counts": dict(sorted(Counter(row["final_role"] for row in materialized).items())),
        "shard_count": sum(len(value) for value in shards.values()), "shards": shards,
        "verification": verification,
        "manifest_sha256": {name: _sha_file(temp / name) for name in manifest_names},
        "policy": {"text_materialized_inactive": True, "text_activated": False, "conditioned_material_excluded": True, "v7_created": False, "mixture_assigned": False, "cache_deleted": False, "gpu_work_started": False},
    }


def _replace_verified_output(temp: Path, output: Path) -> None:
    backup = output.parent / f".{output.name}.previous"
    if backup.exists():
        raise FileExistsError(f"stale build backup exists: {backup}")
    moved = False
    try:
        if output.exists():
            os.replace(output, backup); moved = True
        os.replace(temp, output)
    except Exception:
        if moved and not output.exists() and backup.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _canonical(text: str) -> str:
    return text.rstrip() + "\n" if text.strip() else ""


def _stats(text: str) -> dict[str, Any]:
    payload = text.encode("utf-8")
    return {"cleaned_character_count": len(text), "cleaned_byte_count": len(payload), "cleaned_sha256": hashlib.sha256(payload).hexdigest()}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _unique(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        if row[key] in result: raise ValueError(f"duplicate {key}: {row[key]}")
        result[row[key]] = row
    return result


def _progress(progress: Progress, phase: str, completed: int, total: int, started: float) -> None:
    elapsed = monotonic() - started; rate = completed / elapsed if elapsed else 0.0
    eta = (total - completed) / rate if rate else 0.0
    progress(f"{phase} completed={completed:,}/{total:,} percent={completed/max(1,total):.1%} elapsed={elapsed:.1f}s eta={eta:.1f}s")
