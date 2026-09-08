"""Build the V8 logical corpus view without changing frozen V1/V7 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sonnet_corpus import biblioteca_italiana as bibit
from sonnet_corpus.canonical_corpus_reader import CanonicalCorpusReader
from sonnet_corpus.gutenberg_fulltext_probe import (
    _normalized_words,
    _rolling_shingle_hashes,
)
from sonnet_corpus.sonnet_v7_split import canonicalize_author_label


V8_VERSION = "sonnets_expanded_v8"
V8_BUILD_DATE = "2026-09-08"
TARGET_SOURCE_COUNTS = {"bibit000118": 48, "bibit001444": 21}
REPAIRED_UNITS = {
    "bibit:sonnet:bibit000118:sonnet_0012",
    "bibit:sonnet:bibit001444:structural_0009",
}
GENERIC_AUTHOR_KEYS = {
    canonicalize_author_label(value)
    for value in ("", "Anonimo", "Non definito", "Various", "unresolved", "unknown")
}


@dataclass(frozen=True)
class SonnetV8CorrectionConfig:
    repo_root: Path
    canonical_corpus_dir: Path
    v7_manifest_path: Path
    correction_ledger_path: Path
    alias_evidence_path: Path
    policy_path: Path
    tei_cache_dir: Path
    output_dir: Path
    overlap_report_path: Path
    json_report_path: Path
    markdown_report_path: Path


def build_sonnet_v8_corpus(config: SonnetV8CorrectionConfig) -> dict[str, Any]:
    """Build a deterministic manifest overlay and enforce all V8 gates."""

    policy = _load_policy(config.policy_path)
    reader = CanonicalCorpusReader(config.repo_root, config.canonical_corpus_dir)
    base_records = _read_csv(config.canonical_corpus_dir / "records_manifest.csv")
    base_record_by_id = {row["unit_id"]: row for row in base_records}
    base_sonnets = _read_csv(config.canonical_corpus_dir / "sonnets_manifest.csv")
    base_storage = {row["unit_id"]: row for row in _read_csv(config.canonical_corpus_dir / "storage_manifest.csv")}
    base_attributions = _read_csv(config.canonical_corpus_dir / "attribution_manifest.csv")
    v7_rows = {row["unit_id"]: row for row in _read_csv(config.v7_manifest_path)}
    ledger_rows = _read_csv(config.correction_ledger_path)
    alias_evidence_rows = _read_csv(config.alias_evidence_path)
    _validate_ledger(ledger_rows, v7_rows, alias_evidence_rows, config.repo_root)

    evidence = _extract_source_evidence(config, ledger_rows, policy)
    for row in ledger_rows:
        found = evidence[row["unit_id"]]
        if found["source_author_literal"] != row["source_author_literal"]:
            raise ValueError(f"TEI author evidence changed: {row['unit_id']}")
        if found["source_element"] != row["source_element"]:
            raise ValueError(f"TEI author selector changed: {row['unit_id']}")
        if row["corrected_author"] != row["source_author_literal"]:
            if row["resolution_method"] != "curated_alias_to_existing_source_identity":
                raise ValueError(f"unsupported author expansion: {row['unit_id']}")
            if not row["alias_evidence"].strip():
                raise ValueError(f"curated author alias lacks evidence: {row['unit_id']}")

    unit_by_id = {unit.unit_id: unit for unit in reader.units}
    excluded_records = set(policy["excluded_broader_unit_ids"])
    record_rows = [
        {
            **row,
            "v8_decision": "retain_v1_slice",
            "v8_normalization_policy": "long_s_u017f_to_ascii_s_v1",
            "long_s_replacement_count": "0",
            "source_storage_path": row["storage_path"],
            "source_storage_kind": row["storage_kind"],
            "source_byte_start": row["byte_start"],
            "source_byte_end": row["byte_end"],
            "source_logical_character_count": row["logical_character_count"],
            "source_logical_byte_count": row["logical_byte_count"],
            "source_logical_sha256": row["logical_sha256"],
        }
        for row in base_records
        if row["training_eligible"] == "true" and row["unit_id"] not in excluded_records
    ]
    if excluded_records & {row["unit_id"] for row in record_rows}:
        raise ValueError("excluded anthology residual remains in broader training")
    if excluded_records - {row["unit_id"] for row in base_records}:
        raise ValueError("V8 broader exclusion is absent from the V1 manifest")

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{config.output_dir.name}.", dir=config.output_dir.parent))
    try:
        derived_path = temp_dir / "derived_sonnets" / "part-0001.txt"
        derived_path.parent.mkdir(parents=True)
        derived_payload = bytearray()
        broader_path = temp_dir / "derived_broader_long_s" / "part-0001.txt"
        broader_path.parent.mkdir(parents=True)
        sonnet_rows: list[dict[str, str]] = []
        corrected_authors = {row["unit_id"]: row["corrected_author"] for row in ledger_rows}

        for base_row in base_sonnets:
            unit_id = base_row["unit_id"]
            v7 = v7_rows[unit_id]
            if v7["include_in_v7"] != "true":
                continue
            row = dict(base_row)
            row["training_eligible"] = v7["v7_training_eligible"]
            row["activation_status"] = f"v8_{v7['v7_split']}"
            row.update({key: v7[key] for key in (
                "canonical_author_key", "author_group_id", "author_resolution_status",
                "work_group_id", "split_group_id", "v7_split", "v7_split_tier",
                "v7_split_decision",
            )})
            row["v8_rendering_policy"] = "derived_source_lines_4+4+3+3_v1"
            row["source_tei_sha256"] = ""
            row["source_line_text_sha256"] = ""
            row["source_storage_path"] = base_row["storage_path"]
            row["source_byte_start"] = base_row["byte_start"]
            row["source_byte_end"] = base_row["byte_end"]
            row["source_logical_sha256"] = base_row["logical_sha256"]
            row["supersedes_logical_sha256"] = ""

            old_text = reader.read_text(unit_by_id[unit_id])
            if unit_id in corrected_authors:
                source = evidence[unit_id]
                row["author"] = corrected_authors[unit_id]
                row["canonical_author_key"] = canonicalize_author_label(row["author"])
                row["author_group_id"] = _author_group_id(row["canonical_author_key"])
                ledger = next(item for item in ledger_rows if item["unit_id"] == unit_id)
                row["author_resolution_status"] = ledger["resolution_status"]
                row["line_count"] = "14"
                row["source_tei_sha256"] = source["tei_sha256"]
                row["source_line_text_sha256"] = _sha256_text("\n".join(source["lines"]))
                if unit_id in REPAIRED_UNITS:
                    row["supersedes_logical_sha256"] = row["logical_sha256"]
                source_lines = source["lines"]
            else:
                source_lines = [line for line in old_text.splitlines() if line.strip()]
            source_rendering = _render_4_4_3_3(source_lines)
            long_s_count = source_rendering.count("ſ")
            text = source_rendering.replace("ſ", "s")
            row["v8_normalization_policy"] = "long_s_u017f_to_ascii_s_v1"
            row["long_s_replacement_count"] = str(long_s_count)
            if unit_id in REPAIRED_UNITS and old_text == text:
                raise ValueError(f"declared repair does not change text: {unit_id}")
            if derived_payload:
                derived_payload.extend(b"\n")
            start = len(derived_payload)
            payload = text.encode("utf-8")
            derived_payload.extend(payload)
            row["storage_kind"] = "v8_derived_4+4+3+3"
            row["storage_path"] = f"{_portable(config.output_dir, config.repo_root)}/derived_sonnets/part-0001.txt"
            row["byte_start"] = str(start)
            row["byte_end"] = str(start + len(payload))
            row["logical_character_count"] = str(len(text))
            row["logical_byte_count"] = str(len(payload))
            row["logical_sha256"] = _sha256_bytes(payload)
            sonnet_rows.append(row)

        _assign_v8_splits(sonnet_rows, ledger_rows)
        derived_path.write_bytes(bytes(derived_payload))
        derived_file_sha256 = _sha256_bytes(bytes(derived_payload))
        broader_payload = _write_normalized_broader_shard(
            config,
            reader,
            unit_by_id,
            base_record_by_id,
            record_rows,
            broader_path,
        )
        broader_file_sha256 = _sha256_bytes(bytes(broader_payload))
        _validate_sonnets(sonnet_rows, ledger_rows)

        initial_storage_rows = _build_storage_rows(
            record_rows,
            sonnet_rows,
            base_storage,
            derived_file_sha256,
            broader_file_sha256,
        )
        initial_overlap = _audit_broader_against_heldout(
            config.repo_root,
            temp_dir,
            record_rows,
            sonnet_rows,
            initial_storage_rows,
            policy,
            phase="pre_quarantine",
        )
        _validate_initial_overlap(initial_overlap, policy)
        overlap_quarantine_ids = {
            item["broader_unit_id"] for item in initial_overlap["meaningful_pairs"]
        }
        record_rows = [
            row for row in record_rows if row["unit_id"] not in overlap_quarantine_ids
        ]
        broader_payload = _write_normalized_broader_shard(
            config,
            reader,
            unit_by_id,
            base_record_by_id,
            record_rows,
            broader_path,
        )
        broader_file_sha256 = _sha256_bytes(bytes(broader_payload))
        storage_rows = _build_storage_rows(
            record_rows,
            sonnet_rows,
            base_storage,
            derived_file_sha256,
            broader_file_sha256,
        )

        used_attributions = {row["attribution_id"] for row in (*record_rows, *sonnet_rows)}
        attribution_rows = [
            _v8_attribution_row(row)
            for row in base_attributions
            if row["attribution_id"] in used_attributions
        ]
        if {row["attribution_id"] for row in attribution_rows} != used_attributions:
            raise ValueError("V8 attribution join is incomplete")

        _write_csv(temp_dir / "records_manifest.csv", record_rows)
        _write_csv(temp_dir / "sonnets_manifest.csv", sonnet_rows)
        _write_csv(temp_dir / "storage_manifest.csv", storage_rows)
        _write_csv(temp_dir / "attribution_manifest.csv", attribution_rows)
        exclusions = [
            {
                "unit_id": unit_id,
                "decision": "exclude_anthology_residual_from_v8_broader_training",
                "reason": "source contains extracted sonnets with poem-level attribution distinct from anthology editor",
                "meaningful_pair_count": "",
                "maximum_heldout_containment": "",
            }
            for unit_id in sorted(excluded_records)
        ]
        initial_findings_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for finding in initial_overlap["meaningful_pairs"]:
            initial_findings_by_record[finding["broader_unit_id"]].append(finding)
        exclusions.extend(
            {
                "unit_id": unit_id,
                "decision": "quarantine_initial_broader_heldout_overlap",
                "reason": "initial normalized eight-word-shingle audit met the approved meaningful-overlap thresholds",
                "meaningful_pair_count": str(len(findings)),
                "maximum_heldout_containment": f"{max(item['heldout_containment'] for item in findings):.8f}",
            }
            for unit_id, findings in sorted(initial_findings_by_record.items())
        )
        _write_csv(temp_dir / "exclusions_manifest.csv", exclusions)

        final_overlap = _audit_broader_against_heldout(
            config.repo_root,
            temp_dir,
            record_rows,
            sonnet_rows,
            storage_rows,
            policy,
            phase="post_quarantine",
        )
        if final_overlap["meaningful_pair_count"]:
            raise ValueError("meaningful broader-training/heldout-sonnet overlap remains")
        overlap = {
            "audit_version": "sonnets_expanded_v8_broader_heldout_overlap_v1",
            "status": "pass",
            "quarantine_policy": "quarantine_every_broader_record_in_initial_meaningful_findings",
            "pre_quarantine_audit": initial_overlap,
            "quarantined_broader_unit_count": len(overlap_quarantine_ids),
            "quarantined_broader_unit_ids": sorted(overlap_quarantine_ids),
            "post_quarantine_audit": final_overlap,
        }

        report = _build_report(
            config,
            record_rows,
            sonnet_rows,
            derived_payload,
            broader_payload,
            overlap,
            exclusions,
            attribution_rows,
        )
        _write_json(temp_dir / "build_report.json", report)
        if config.output_dir.exists():
            shutil.rmtree(config.output_dir)
        os.replace(temp_dir, config.output_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    _write_json(config.overlap_report_path, overlap)
    _write_json(config.json_report_path, report)
    config.markdown_report_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("version") != "sonnets_expanded_v8_correction_policy_v1":
        raise ValueError("unexpected V8 correction policy version")
    if policy.get("base_corpus_version") != "canonical_italian_corpora_v1":
        raise ValueError("V8 policy must preserve the frozen V1 corpus")
    if policy.get("base_split_version") != "sonnets_expanded_v7":
        raise ValueError("V8 policy must preserve the frozen V7 split")
    if policy.get("rendering", {}).get("stanza_pattern") != [4, 4, 3, 3]:
        raise ValueError("V8 rendering policy must be 4+4+3+3")
    if policy.get("text_normalization") != {
        "name": "long_s_u017f_to_ascii_s_v1",
        "replacement": "s",
        "source_character": "ſ",
    }:
        raise ValueError("V8 long-s normalization policy drifted")
    expected_sources = policy.get("tei_sources")
    if expected_sources != {
        "bibit000118": "2a8a3051f1111fc3fb978a53de8c5af0b8d1fee20fea10bc85dd1b0a669051a9",
        "bibit001444": "90e5ddca0960ff6af5019b0825054290f6384abe3ba5bfd86d503afaa977a500",
    }:
        raise ValueError("V8 TEI source pins drifted")
    return policy


def _validate_ledger(
    rows: list[dict[str, str]],
    v7: dict[str, dict[str, str]],
    alias_evidence_rows: list[dict[str, str]],
    repo_root: Path,
) -> None:
    counts = Counter(row["source_record_id"] for row in rows)
    if dict(counts) != TARGET_SOURCE_COUNTS:
        raise ValueError(f"unexpected correction-ledger counts: {dict(counts)}")
    if len({row["unit_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate V8 correction-ledger identity")
    active = {
        unit_id for unit_id, row in v7.items()
        if row["include_in_v7"] == "true" and any(
            row["source_id"].startswith(source + ":") for source in TARGET_SOURCE_COUNTS
        )
    }
    if active != {row["unit_id"] for row in rows}:
        raise ValueError("correction ledger does not exactly cover active anthology sonnets")
    aliases = {
        row["corrected_author"]: row for row in rows
        if row["resolution_method"] == "curated_alias_to_existing_source_identity"
    }
    evidence = {row["corrected_author"]: row for row in alias_evidence_rows}
    if set(evidence) != set(aliases):
        raise ValueError("alias evidence does not exactly cover curated V8 identities")
    for author, evidence_row in evidence.items():
        path = repo_root / evidence_row["evidence_path"]
        source_rows = _read_csv(path)
        matches = [
            row for row in source_rows
            if row[evidence_row["evidence_id_field"]]
            == evidence_row["evidence_id_value"]
        ]
        if len(matches) != 1:
            raise ValueError(f"alias evidence identity is not unique: {author}")
        source_author = matches[0][evidence_row["evidence_author_field"]]
        if canonicalize_author_label(source_author) != canonicalize_author_label(author):
            raise ValueError(f"alias evidence author mismatch: {author}")


def _extract_source_evidence(
    config: SonnetV8CorrectionConfig,
    rows: list[dict[str, str]],
    policy: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    wanted = {row["unit_id"] for row in rows}
    result: dict[str, dict[str, Any]] = {}
    for source_id in TARGET_SOURCE_COUNTS:
        path = config.tei_cache_dir / f"{source_id}.xml"
        payload = path.read_bytes()
        actual_tei_sha256 = _verify_pinned_tei_payload(
            source_id,
            payload,
            policy["tei_sources"][source_id],
        )
        text = bibit._remove_doctype(payload.decode("utf-8-sig"))
        text = bibit._INVALID_XML_CONTROL.sub("", text)
        root = ET.fromstring(bibit._replace_known_named_entities(text))
        body = bibit._first_descendant(root, "body")
        if body is None:
            raise ValueError(f"missing TEI body: {source_id}")
        parents = {id(child): parent for parent in root.iter() for child in list(parent)}
        explicit = bibit._top_level_sonnet_elements(body, parents)
        elements = explicit if source_id == "bibit000118" else bibit._structural_sonnet_candidate_elements(body, explicit)
        prefix = "sonnet" if source_id == "bibit000118" else "structural"
        for index, element in enumerate(elements, 1):
            unit_id = f"bibit:sonnet:{source_id}:{prefix}_{index:04d}"
            if unit_id not in wanted:
                continue
            author, selector = _structured_author(source_id, element, parents, body)
            lines = [bibit._inline_text(line) for line in bibit._descendants(element, "l")]
            lines = [line for line in lines if line]
            result[unit_id] = {
                "source_author_literal": author,
                "source_element": selector,
                "lines": lines,
                "tei_sha256": actual_tei_sha256,
            }
    if set(result) != wanted:
        raise ValueError("failed to locate every correction-ledger unit in pinned TEI")
    return result


def _structured_author(source_id: str, element: ET.Element, parents: dict[int, ET.Element], body: ET.Element) -> tuple[str, str]:
    ancestor = parents.get(id(element))
    while ancestor is not None and ancestor is not body:
        if bibit._local_name(ancestor.tag).startswith("div"):
            if source_id == "bibit001444":
                value = ancestor.attrib.get("n", "").strip()
                opener = bibit._first_child(ancestor, "opener")
                byline = bibit._first_descendant(opener, "byline") if opener is not None else None
                byline_text = bibit._inline_text(byline) if byline is not None else ""
                if value and byline_text:
                    if canonicalize_author_label(value) != canonicalize_author_label(byline_text):
                        raise ValueError(f"div@n/byline disagreement: {value!r} / {byline_text!r}")
                    return value, "ancestor_div@n+opener/byline"
            else:
                head = bibit._first_child(ancestor, "head")
                heading = bibit._inline_text(head) if head is not None else ""
                match = re.match(r"^\s*[IVXLCDM]+\s+[–-]\s+(.+?)\s*$", heading)
                if match:
                    return match.group(1), "ancestor_div/head_suffix"
        ancestor = parents.get(id(ancestor))
    raise ValueError("source-structured sonnet author is unresolved")


def _render_4_4_3_3(lines: list[str]) -> str:
    if len(lines) != 14 or any(not line.strip() for line in lines):
        raise ValueError("source sonnet must contain exactly 14 non-empty TEI lines")
    return "\n\n".join(("\n".join(lines[:4]), "\n".join(lines[4:8]), "\n".join(lines[8:11]), "\n".join(lines[11:])))


def _verify_pinned_tei_payload(
    source_id: str,
    payload: bytes,
    expected_sha256: str,
) -> str:
    actual_sha256 = _sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"pinned TEI hash mismatch: {source_id} "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return actual_sha256


def _v8_attribution_row(row: dict[str, str]) -> dict[str, str]:
    result = dict(row)
    prior_notice = result["modification_notice"].rstrip().rstrip(".")
    v8_notice = (
        "V8 derives a 4+4+3+3 stanza view for sonnets, repairs two source-backed "
        "TEI line renderings, and replaces U+017F long s with ASCII s where present"
    )
    result["modification_notice"] = f"{prior_notice}. {v8_notice}."
    result["activation_status"] = "v8_future_training_view"
    return result


def _validate_sonnets(rows: list[dict[str, str]], ledger: list[dict[str, str]]) -> None:
    if any(int(row["line_count"]) != 14 for row in rows):
        raise ValueError("V8 includes a sonnet without exactly 14 source lines")
    corrected = {row["unit_id"] for row in ledger}
    selected = {row["unit_id"] for row in rows}
    if corrected - selected:
        raise ValueError("corrected anthology sonnet is absent from V8")
    for row in rows:
        if row["unit_id"] in corrected and canonicalize_author_label(row["author"]) in GENERIC_AUTHOR_KEYS:
            raise ValueError(f"corrected author remains unresolved: {row['unit_id']}")
        if row["v8_split"] in {"validation", "test"} and row["v8_training_eligible"] == "true":
            raise ValueError("held-out V8 sonnet is training eligible")


def _assign_v8_splits(rows: list[dict[str, str]], ledger_rows: list[dict[str, str]]) -> None:
    ledger = {row["unit_id"]: row for row in ledger_rows}
    heldout_keys = {
        row["canonical_author_key"]
        for row in rows
        if row["v7_split"] in {"validation", "test"} and row["canonical_author_key"]
    }
    heldout_works = {
        row["work_group_id"]
        for row in rows
        if row["v7_split"] in {"validation", "test"} and row["work_group_id"]
    }
    for row in rows:
        row["base_v7_split"] = row["v7_split"]
        row["base_v7_training_eligible"] = row["training_eligible"]
        row["v8_split"] = row["v7_split"]
        row["v8_split_group_id"] = _v8_split_group_id(
            row["author_group_id"], row["work_group_id"]
        )
        row["v8_split_decision"] = "preserve_v7_assignment_after_v8_check"
        row["v8_training_eligible"] = row["training_eligible"]
        resolution = ledger.get(row["unit_id"], {}).get("resolution_status", "")
        if resolution == "quarantine_unresolved_abbreviation":
            row["v8_split"] = "quarantine"
            row["v8_training_eligible"] = "false"
            row["training_eligible"] = "false"
            row["v8_split_decision"] = "quarantine_unresolved_source_abbreviation"
        elif row["v7_split"] == "train" and (
            row["canonical_author_key"] in heldout_keys
            or row["work_group_id"] in heldout_works
        ):
            row["v8_split"] = "quarantine"
            row["v8_training_eligible"] = "false"
            row["training_eligible"] = "false"
            row["v8_split_decision"] = "quarantine_author_or_work_collision_with_fixed_heldout"
        elif row["unit_id"] in ledger:
            row["v8_split"] = "train"
            row["v8_training_eligible"] = "true"
            row["training_eligible"] = "true"
            row["v8_split_decision"] = "retain_train_after_corrected_author_heldout_check"

    v8_train_keys = {
        row["canonical_author_key"] for row in rows
        if row["v8_training_eligible"] == "true" and row["canonical_author_key"]
    }
    v8_heldout_keys = {
        row["canonical_author_key"] for row in rows
        if row["v8_split"] in {"validation", "test"} and row["canonical_author_key"]
    }
    overlap = v8_train_keys & v8_heldout_keys
    if overlap:
        raise ValueError(f"V8 corrected author split leakage remains: {sorted(overlap)[:3]}")
    v8_train_works = {
        row["work_group_id"] for row in rows
        if row["v8_training_eligible"] == "true" and row["work_group_id"]
    }
    v8_heldout_works = {
        row["work_group_id"] for row in rows
        if row["v8_split"] in {"validation", "test"} and row["work_group_id"]
    }
    if v8_train_works & v8_heldout_works:
        raise ValueError("V8 train/heldout work-group leakage remains")


def _write_normalized_broader_shard(
    config: SonnetV8CorrectionConfig,
    reader: CanonicalCorpusReader,
    unit_by_id: dict[str, Any],
    base_record_by_id: dict[str, dict[str, str]],
    rows: list[dict[str, str]],
    path: Path,
) -> bytearray:
    payload = bytearray()
    for row in rows:
        base = base_record_by_id[row["unit_id"]]
        for key in (
            "storage_kind",
            "storage_path",
            "byte_start",
            "byte_end",
            "logical_character_count",
            "logical_byte_count",
            "logical_sha256",
        ):
            row[key] = base[key]
        row["v8_decision"] = "retain_v1_slice"
        source_text = reader.read_text(unit_by_id[row["unit_id"]])
        count = source_text.count("ſ")
        row["long_s_replacement_count"] = str(count)
        if not count:
            continue
        if payload:
            payload.extend(b"\n")
        start = len(payload)
        normalized = source_text.replace("ſ", "s")
        encoded = normalized.encode("utf-8")
        payload.extend(encoded)
        row["v8_decision"] = "derive_long_s_normalized_slice"
        row["storage_kind"] = "v8_derived_long_s_normalization"
        row["storage_path"] = (
            f"{_portable(config.output_dir, config.repo_root)}"
            "/derived_broader_long_s/part-0001.txt"
        )
        row["byte_start"] = str(start)
        row["byte_end"] = str(start + len(encoded))
        row["logical_character_count"] = str(len(normalized))
        row["logical_byte_count"] = str(len(encoded))
        row["logical_sha256"] = _sha256_bytes(encoded)
    path.write_bytes(bytes(payload))
    return payload


def _build_storage_rows(
    records: list[dict[str, str]],
    sonnets: list[dict[str, str]],
    base_storage: dict[str, dict[str, str]],
    derived_sonnet_sha256: str,
    derived_broader_sha256: str,
) -> list[dict[str, str]]:
    result = []
    for row in (*records, *sonnets):
        source = base_storage[row["unit_id"]]
        if row["storage_kind"] == "v8_derived_4+4+3+3":
            physical_sha256 = derived_sonnet_sha256
        elif row["storage_kind"] == "v8_derived_long_s_normalization":
            physical_sha256 = derived_broader_sha256
        else:
            physical_sha256 = source["physical_file_sha256"]
        result.append(
            {
                "storage_id": f"storage:{row['unit_id']}",
                "unit_id": row["unit_id"],
                "unit_kind": "broader" if ":record:" in row["unit_id"] else "standard_sonnet",
                "final_role": row.get("final_role", "standard_sonnets"),
                "storage_kind": row["storage_kind"],
                "storage_path": row["storage_path"],
                "byte_start": row["byte_start"],
                "byte_end": row["byte_end"],
                "logical_character_count": row["logical_character_count"],
                "logical_byte_count": row["logical_byte_count"],
                "logical_sha256": row["logical_sha256"],
                "physical_file_sha256": physical_sha256,
                "public_repository_status": "committed_v1_reference_or_v8_delta",
            }
        )
    return result


def _validate_initial_overlap(audit: dict[str, Any], policy: dict[str, Any]) -> None:
    expected = policy["overlap_audit"]["expected_initial_audit"]
    actual = {
        "broader_training_unit_count": audit["broader_training_unit_count"],
        "meaningful_pair_count": audit["meaningful_pair_count"],
        "quarantined_broader_unit_count": len(
            {item["broader_unit_id"] for item in audit["meaningful_pairs"]}
        ),
        "maximum_heldout_containment": audit["maximum_heldout_containment"],
    }
    if actual != expected:
        raise ValueError(f"initial V8 overlap audit drifted: {actual!r}")


def _audit_broader_against_heldout(
    repo_root: Path,
    overlay_dir: Path,
    records: list[dict[str, str]],
    sonnets: list[dict[str, str]],
    storage: list[dict[str, str]],
    policy: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    storage_by_id = {row["unit_id"]: row for row in storage}
    heldout = [row for row in sonnets if row["v8_split"] in {"validation", "test"}]
    postings: dict[int, list[str]] = defaultdict(list)
    denominators: dict[str, int] = {}
    for row in heldout:
        text = _read_row_text(repo_root, overlay_dir, row, storage_by_id[row["unit_id"]])
        if "ſ" in text:
            raise ValueError(f"long s remains in V8 heldout sonnet: {row['unit_id']}")
        hashes = set(_rolling_shingle_hashes(_normalized_words(text)))
        denominators[row["unit_id"]] = len(hashes)
        for value in hashes:
            postings[value].append(row["unit_id"])
    min_matches = int(policy["overlap_audit"]["meaningful_min_matching_shingles"])
    min_containment = float(policy["overlap_audit"]["meaningful_heldout_containment"])
    findings: list[dict[str, Any]] = []
    matching_pairs = 0
    max_containment = 0.0
    for row in records:
        text = _read_row_text(repo_root, overlay_dir, row, storage_by_id[row["unit_id"]])
        if "ſ" in text:
            raise ValueError(f"long s remains in V8 broader training text: {row['unit_id']}")
        hits = Counter()
        for value in set(_rolling_shingle_hashes(_normalized_words(text))):
            for heldout_id in postings.get(value, ()):
                hits[heldout_id] += 1
        for heldout_id, count in hits.items():
            matching_pairs += 1
            containment = count / denominators[heldout_id] if denominators[heldout_id] else 0.0
            max_containment = max(max_containment, containment)
            if count >= min_matches and containment >= min_containment:
                findings.append({
                    "broader_unit_id": row["unit_id"],
                    "heldout_unit_id": heldout_id,
                    "matching_shingles": count,
                    "heldout_unique_shingles": denominators[heldout_id],
                    "heldout_containment": round(containment, 8),
                })
    findings.sort(key=lambda item: (-item["heldout_containment"], item["broader_unit_id"], item["heldout_unit_id"]))
    return {
        "audit_version": "sonnets_expanded_v8_broader_heldout_overlap_v1",
        "phase": phase,
        "status": "pass" if not findings else "fail",
        "shingle_size_words": 8,
        "normalization": "casefold_nfkd_letters_and_digits_from_existing_corpus_probe",
        "broader_training_unit_count": len(records),
        "heldout_sonnet_count": len(heldout),
        "heldout_unique_shingle_count": sum(denominators.values()),
        "candidate_pair_count_with_any_shared_shingle": matching_pairs,
        "meaningful_min_matching_shingles": min_matches,
        "meaningful_heldout_containment": min_containment,
        "maximum_heldout_containment": round(max_containment, 8),
        "meaningful_pair_count": len(findings),
        "meaningful_pairs": findings,
    }


def _read_row_text(repo_root: Path, overlay_dir: Path, row: dict[str, str], storage: dict[str, str]) -> str:
    path = repo_root / row["storage_path"]
    if row["storage_kind"] == "v8_derived_4+4+3+3":
        path = overlay_dir / "derived_sonnets" / "part-0001.txt"
    elif row["storage_kind"] == "v8_derived_long_s_normalization":
        path = overlay_dir / "derived_broader_long_s" / "part-0001.txt"
    with path.open("rb") as handle:
        handle.seek(int(row["byte_start"]))
        payload = handle.read(int(row["logical_byte_count"]))
    if _sha256_bytes(payload) != row["logical_sha256"]:
        raise ValueError(f"logical slice hash mismatch during overlap audit: {row['unit_id']}")
    return payload.decode("utf-8")


def _build_report(
    config: SonnetV8CorrectionConfig,
    records: list[dict[str, str]],
    sonnets: list[dict[str, str]],
    derived_payload: bytearray,
    broader_payload: bytearray,
    overlap: dict[str, Any],
    exclusions: list[dict[str, str]],
    attributions: list[dict[str, str]],
) -> dict[str, Any]:
    split_counts = Counter(row["v8_split"] for row in sonnets)
    normalization_counts = Counter()
    affected_units = 0
    for row in (*records, *sonnets):
        count = int(row["long_s_replacement_count"])
        normalization_counts[row.get("final_role", "standard_sonnets")] += count
        affected_units += count > 0
    quarantined = sum(row["v8_split"] == "quarantine" for row in sonnets)
    quarantine_reasons = Counter(
        row["v8_split_decision"] for row in sonnets if row["v8_split"] == "quarantine"
    )
    ledger_resolution = Counter(
        row["resolution_status"] for row in _read_csv(config.correction_ledger_path)
    )
    return {
        "build_version": V8_VERSION,
        "build_date": V8_BUILD_DATE,
        "status": "pass",
        "base_corpus_version": "canonical_italian_corpora_v1",
        "base_split_version": "sonnets_expanded_v7",
        "broader_training_record_count": len(records),
        "excluded_broader_record_count": len(exclusions),
        "anthology_residual_exclusion_count": sum(
            row["decision"] == "exclude_anthology_residual_from_v8_broader_training"
            for row in exclusions
        ),
        "overlap_quarantine_count": overlap["quarantined_broader_unit_count"],
        "sonnet_count": len(sonnets),
        "sonnet_split_counts": dict(sorted(split_counts.items())),
        "reviewed_authorship_count": sum(TARGET_SOURCE_COUNTS.values()),
        "resolved_authorship_count": sum(TARGET_SOURCE_COUNTS.values())
        - ledger_resolution["quarantine_unresolved_abbreviation"],
        "unresolved_authorship_count": ledger_resolution["quarantine_unresolved_abbreviation"],
        "curated_alias_row_count": ledger_resolution["resolved_curated_alias"],
        "curated_alias_identity_count": len({
            row["corrected_author"]
            for row in _read_csv(config.correction_ledger_path)
            if row["resolution_method"] == "curated_alias_to_existing_source_identity"
        }),
        "source_authorship_counts": TARGET_SOURCE_COUNTS,
        "repaired_sonnet_count": len(REPAIRED_UNITS),
        "derived_sonnet_shard_bytes": len(derived_payload),
        "derived_broader_long_s_shard_bytes": len(broader_payload),
        "quarantined_train_sonnet_count": quarantined,
        "quarantine_reason_counts": dict(sorted(quarantine_reasons.items())),
        "long_s_normalization": {
            "policy": "long_s_u017f_to_ascii_s_v1",
            "affected_unit_count": affected_units,
            "replacement_count_by_role": dict(sorted(normalization_counts.items())),
            "replacement_count": sum(normalization_counts.values()),
        },
        "overlap_audit": overlap,
        "gates": {
            "frozen_v1_preserved": True,
            "frozen_v7_preserved": True,
            "exact_active_authorship_coverage": True,
            "pinned_tei_sources_verified": True,
            "exact_alias_evidence_verified": True,
            "v8_attribution_notices_applied": all(
                row["activation_status"] == "v8_future_training_view"
                and "replaces U+017F long s with ASCII s" in row["modification_notice"]
                for row in attributions
            ),
            "all_included_sonnets_exactly_14_lines": True,
            "no_unresolved_attribution_is_training_eligible": all(
                row["v8_training_eligible"] == "false"
                for row in sonnets
                if row["author_resolution_status"] == "quarantine_unresolved_abbreviation"
            ),
            "anthology_residual_broader_records_excluded": True,
            "initial_meaningful_broader_records_quarantined": (
                overlap["quarantined_broader_unit_count"]
                == len(overlap["quarantined_broader_unit_ids"])
            ),
            "no_meaningful_broader_heldout_overlap": (
                overlap["post_quarantine_audit"]["status"] == "pass"
            ),
            "no_long_s_in_v8_text": True,
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    return (
        "# Expanded Sonnet Corpus V8 Correction\n\n"
        "V8 is a logical overlay on the frozen canonical V1 corpus and V7 split. "
        "It changes no V1/V7 file and copies no large source shard.\n\n"
        f"It reviews source-backed authorship for {report['reviewed_authorship_count']} active anthology sonnets. "
        f"It resolves {report['resolved_authorship_count']} and quarantines {report['unresolved_authorship_count']} unresolved abbreviations. It "
        f"repairs {report['repaired_sonnet_count']} mixed-TEI line renderings, and excludes two anthology residual records from broader training.\n\n"
        f"The V8 training rendering replaces {report['long_s_normalization']['replacement_count']:,} U+017F characters in "
        f"{report['long_s_normalization']['affected_unit_count']:,} units while preserving each V1 source reference and hash.\n\n"
        f"The initial normalized eight-word-shingle audit compared {report['overlap_audit']['pre_quarantine_audit']['broader_training_unit_count']:,} broader training records with "
        f"{report['overlap_audit']['pre_quarantine_audit']['heldout_sonnet_count']:,} held-out sonnets. It found {report['overlap_audit']['pre_quarantine_audit']['meaningful_pair_count']} meaningful pairs and quarantined "
        f"{report['overlap_audit']['quarantined_broader_unit_count']} broader records. The final audit found {report['overlap_audit']['post_quarantine_audit']['meaningful_pair_count']} meaningful pairs.\n\n"
        "All included sonnets have exactly 14 source lines. Corrected anthology attributions are non-generic and trace to pinned local TEI structure.\n"
    )


def _author_group_id(key: str) -> str:
    return f"author:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"


def _v8_split_group_id(author_group: str, work_group: str) -> str:
    value = f"{author_group}\0{work_group}"
    return f"v8split:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _portable(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))
