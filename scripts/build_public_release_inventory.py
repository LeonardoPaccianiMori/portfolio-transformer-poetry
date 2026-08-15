#!/usr/bin/env python3
"""Build and validate fail-closed current-tree and published-history inventories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from functools import cache
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release"
CURRENT_MANIFEST = RELEASE_DIR / "artifact_rights_manifest.csv"
HISTORY_MANIFEST = RELEASE_DIR / "history_rights_manifest.csv"
HISTORY_TARGET = RELEASE_DIR / "history_review_target.txt"
SELF_GENERATED = "self_generated"
PENDING = "pending_review"

FIELDS = [
    "repository_relative_path", "scope", "git_blob_oid", "sha256", "artifact_class",
    "source", "source_revision", "rights_evidence_reference", "modifications_made_by_project",
    "required_notices", "current_tree_disposition", "historical_retention_disposition",
    "privacy_security_disposition", "decision_authority_role", "decision_record_id", "review_date",
]

RESOLVED_CURRENT = {"approved_for_public_tree", "remove_from_current_tree", "not_applicable"}
RESOLVED_HISTORY = {"approved_for_public_history", "remove_from_history", "not_applicable"}
RESOLVED_PRIVACY = {"approved", "sanitized", "remove_from_history", "not_applicable"}
REQUIRED_EVIDENCE_FIELDS = {
    "artifact_class", "source", "source_revision", "rights_evidence_reference",
    "modifications_made_by_project", "required_notices",
}
UNRESOLVED_VALUES = {
    "", PENDING, "pending", "unresolved", "needs_review",
    "mixed_or_third_party_source_needs_review", "recorded_in_project_metadata",
}


def run_git(*args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=text)
    return result.stdout


def default_published_refs() -> list[str]:
    target = HISTORY_TARGET.read_text(encoding="utf-8").strip()
    if not target:
        raise ValueError(f"{HISTORY_TARGET}: empty history review target")
    return [target]


def tracked_paths() -> list[str]:
    return sorted(indexed_entries())


@cache
def indexed_entries() -> dict[str, str]:
    output = run_git("ls-files", "-s", "-z", text=False)
    entries: dict[str, str] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        _mode, oid, stage = metadata.split()
        if stage != b"0":
            raise ValueError("unmerged index entry")
        entries[raw_path.decode("utf-8", "surrogateescape")] = oid.decode("ascii")
    return entries


@cache
def unstaged_paths() -> set[str]:
    output = run_git("diff-files", "--name-only", "-z", text=False)
    return {part.decode("utf-8", "surrogateescape") for part in output.split(b"\0") if part}


def published_history_pairs(refs: Iterable[str]) -> list[tuple[str, str]]:
    output = run_git(
        "log", "-m", "--root", "--raw", "--no-abbrev", "--no-renames",
        "--format=", "--no-color", "-z", *refs, text=False,
    )
    return parse_raw_history_pairs(output)


def parse_raw_history_pairs(output: bytes) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    zero_oid = "0" * 40
    fields = output.split(b"\0")
    for offset in range(0, len(fields) - 1, 2):
        metadata = fields[offset].lstrip(b"\n").decode("ascii")
        if not metadata.startswith(":"):
            continue
        path = fields[offset + 1].decode("utf-8", "surrogateescape")
        parts = metadata.split()
        if len(parts) < 5:
            continue
        old_mode, new_mode = parts[0].removeprefix(":"), parts[1]
        old_oid, new_oid = parts[-3], parts[-2]
        for oid, mode in ((old_oid, old_mode), (new_oid, new_mode)):
            if oid != zero_oid and mode != "160000":
                pairs.add((path, oid))
    return sorted(pairs)


def indexed_blob_identity(path: str) -> tuple[str, str]:
    generated_manifests = {
        CURRENT_MANIFEST.relative_to(ROOT).as_posix(),
        HISTORY_MANIFEST.relative_to(ROOT).as_posix(),
    }
    if path in generated_manifests:
        return SELF_GENERATED, SELF_GENERATED
    blob_oid = indexed_entries()[path]
    if path in unstaged_paths():
        payload = run_git("cat-file", "blob", blob_oid, text=False)
    else:
        payload = (ROOT / path).read_bytes()
    return blob_oid, hashlib.sha256(payload).hexdigest()


@cache
def historical_blob_sha256(oid: str) -> str:
    return hashlib.sha256(run_git("cat-file", "blob", oid, text=False)).hexdigest()


def artifact_class(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if path.startswith("LICENSE") or path in {"NOTICE", "CITATION.cff"}:
        return "license_or_notice"
    if path.startswith("tests/"):
        return "test"
    if path.startswith(("src/", "scripts/")):
        return "software"
    if path.startswith("configs/") or path in {"pyproject.toml", "requirements.txt"} or path.startswith("requirements/"):
        return "configuration"
    if path.startswith("reports/"):
        return "aggregate_report" if "public/" in path else "report"
    if path.startswith("data/processed/") and suffix in {".txt", ".jsonl"}:
        return "processed_corpus"
    if path.startswith("data/"):
        return "data_metadata_or_artifact"
    if path.startswith("demo/"):
        return "demo"
    if path.startswith(("docs/", "release/")) or suffix == ".md":
        return "documentation"
    return "repository_support"


def default_source(path: str, oid: str) -> tuple[str, str, str]:
    kind = artifact_class(path)
    if path in {CURRENT_MANIFEST.relative_to(ROOT).as_posix(), HISTORY_MANIFEST.relative_to(ROOT).as_posix()}:
        return "generated_by_release_inventory_tool", SELF_GENERATED, "scripts/build_public_release_inventory.py"
    if kind in {"processed_corpus", "data_metadata_or_artifact"}:
        return "mixed_or_third_party_source_needs_review", "recorded_in_project_metadata", "DATA_SOURCES_AND_ATTRIBUTION.md"
    return "Leonardo_project_work_with_AI_assistance", oid, "AI_CONTRIBUTIONS.md"


def load_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["repository_relative_path"], row["scope"], row["git_blob_oid"]): row for row in rows}


def make_row(path: str, scope: str, oid: str, sha256: str, existing: dict[str, str] | None) -> dict[str, str]:
    source, revision, evidence = default_source(path, oid)
    row = {
        "repository_relative_path": path, "scope": scope, "git_blob_oid": oid, "sha256": sha256,
        "artifact_class": artifact_class(path), "source": source, "source_revision": revision,
        "rights_evidence_reference": evidence, "modifications_made_by_project": "needs_review",
        "required_notices": "needs_review", "current_tree_disposition": PENDING,
        "historical_retention_disposition": PENDING, "privacy_security_disposition": PENDING,
        "decision_authority_role": PENDING, "decision_record_id": PENDING, "review_date": PENDING,
    }
    if existing:
        for field in FIELDS[4:]:
            if existing.get(field):
                row[field] = existing[field]
    return row


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(refs: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    old_current = load_rows(CURRENT_MANIFEST)
    old_history = load_rows(HISTORY_MANIFEST)
    history_rows = []
    for path, oid in published_history_pairs(refs):
        key = (path, "published_history", oid)
        history_rows.append(make_row(path, "published_history", oid, historical_blob_sha256(oid), old_history.get(key)))
    write_rows(HISTORY_MANIFEST, history_rows)
    current_rows = []
    for path in tracked_paths():
        oid, sha256 = indexed_blob_identity(path)
        key = (path, "current_tree", oid)
        current_rows.append(make_row(path, "current_tree", oid, sha256, old_current.get(key)))
    write_rows(CURRENT_MANIFEST, current_rows)
    return current_rows, history_rows


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"{path}: unexpected columns")
        return list(reader)


def structural_errors(refs: list[str]) -> list[str]:
    errors: list[str] = []
    current = read_manifest(CURRENT_MANIFEST)
    history = read_manifest(HISTORY_MANIFEST)
    expected_current = set(tracked_paths())
    actual_current = {row["repository_relative_path"] for row in current}
    if expected_current != actual_current:
        errors.append("current-tree manifest coverage differs from git index")
    expected_history = set(published_history_pairs(refs))
    actual_history = {(row["repository_relative_path"], row["git_blob_oid"]) for row in history}
    if expected_history != actual_history:
        errors.append("history manifest coverage differs from published refs")
    if len(actual_current) != len(current) or len(actual_history) != len(history):
        errors.append("duplicate manifest row")
    for row in current:
        if row["scope"] != "current_tree":
            errors.append(f"unexpected current scope: {row['repository_relative_path']}")
        expected_oid, expected_sha = indexed_blob_identity(row["repository_relative_path"])
        if (row["git_blob_oid"], row["sha256"]) != (expected_oid, expected_sha):
            errors.append(f"stale current identity: {row['repository_relative_path']}")
    for row in history:
        if row["scope"] != "published_history":
            errors.append(f"unexpected historical scope: {row['repository_relative_path']}")
        if row["sha256"] != historical_blob_sha256(row["git_blob_oid"]):
            errors.append(f"stale historical identity: {row['repository_relative_path']}")
    return errors


def clearance_errors(rows: Iterable[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        path = row["repository_relative_path"]
        for field in REQUIRED_EVIDENCE_FIELDS:
            if row[field].strip().lower() in UNRESOLVED_VALUES:
                errors.append(f"{path}: unresolved {field}")
        if row["current_tree_disposition"] not in RESOLVED_CURRENT:
            errors.append(f"{path}: unresolved current-tree disposition")
        if row["historical_retention_disposition"] not in RESOLVED_HISTORY:
            errors.append(f"{path}: unresolved historical disposition")
        if row["privacy_security_disposition"] not in RESOLVED_PRIVACY:
            errors.append(f"{path}: unresolved privacy/security disposition")
        if row["scope"] == "current_tree" and row["current_tree_disposition"] == "remove_from_current_tree":
            errors.append(f"{path}: marked for removal but still present in current tree")
        if (
            row["historical_retention_disposition"] == "remove_from_history"
            or row["privacy_security_disposition"] == "remove_from_history"
        ):
            errors.append(f"{path}: history removal requires a separate destructive-action plan")
        for field in ("decision_authority_role", "decision_record_id", "review_date"):
            if not row[field] or row[field] in {PENDING, "pending", "unresolved"}:
                errors.append(f"{path}: unresolved {field}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published-ref", action="append", default=[])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-cleared", action="store_true")
    args = parser.parse_args()
    refs = args.published_ref or default_published_refs()
    if not args.check:
        build(refs)
    errors = structural_errors(refs)
    if args.require_cleared:
        errors.extend(clearance_errors(read_manifest(CURRENT_MANIFEST)))
        errors.extend(clearance_errors(read_manifest(HISTORY_MANIFEST)))
    for error in errors:
        print(f"release-inventory | ERROR | {error}")
    if errors:
        return 1
    print(f"release-inventory | OK | current={len(read_manifest(CURRENT_MANIFEST))} history={len(read_manifest(HISTORY_MANIFEST))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
