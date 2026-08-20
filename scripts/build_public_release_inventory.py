#!/usr/bin/env python3
"""Build and validate fail-closed current-tree and published-history inventories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from functools import cache
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release"
CURRENT_MANIFEST = RELEASE_DIR / "artifact_rights_manifest.csv"
HISTORY_MANIFEST = RELEASE_DIR / "history_rights_manifest.csv"
HISTORY_TARGET = RELEASE_DIR / "history_review_target.txt"
DECISION_POLICY = RELEASE_DIR / "public_release_decision_policy.json"
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
RESOLVED_PRIVACY = {
    "approved", "approved_historical_exception", "sanitized",
    "remove_from_history", "not_applicable",
}
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


@cache
def decision_policy() -> dict:
    policy = json.loads(DECISION_POLICY.read_text(encoding="utf-8"))
    required = {
        "schema_version", "review_date", "decision_authority_role",
        "decision_record_id", "historical_privacy_exceptions", "rules",
    }
    missing = required - set(policy)
    if missing:
        raise ValueError(f"{DECISION_POLICY}: missing fields {sorted(missing)}")
    return policy


def matching_rule(path: str) -> dict | None:
    for rule in decision_policy()["rules"]:
        if any(fnmatchcase(path, pattern) for pattern in rule["path_globs"]):
            return rule
    return None


def historical_privacy_exception(path: str, oid: str) -> dict | None:
    for exception in decision_policy()["historical_privacy_exceptions"]:
        if (
            exception["repository_relative_path"] == path
            and exception["git_blob_oid"] == oid
        ):
            return exception
    return None


def make_row(path: str, scope: str, oid: str, sha256: str) -> dict[str, str]:
    rule = matching_rule(path)
    if rule is None:
        source = revision = evidence = PENDING
        artifact = artifact_class(path)
        modifications = notices = PENDING
        current_disposition = historical_disposition = privacy_disposition = PENDING
    else:
        source = rule["source"]
        revision = rule["source_revision"]
        if revision == "git_blob_oid":
            revision = oid
        elif revision == SELF_GENERATED and scope == "published_history":
            revision = oid
        evidence = rule["rights_evidence_reference"]
        artifact = rule["artifact_class"]
        modifications = rule["modifications_made_by_project"]
        notices = rule["required_notices"]
        current_disposition = (
            rule["current_tree_disposition"] if scope == "current_tree" else "not_applicable"
        )
        historical_disposition = (
            rule["historical_retention_disposition"]
            if scope == "published_history"
            else "not_applicable"
        )
        privacy_disposition = rule["privacy_security_disposition"]
        if scope == "published_history" and historical_privacy_exception(path, oid):
            privacy_disposition = "approved_historical_exception"
    row = {
        "repository_relative_path": path, "scope": scope, "git_blob_oid": oid, "sha256": sha256,
        "artifact_class": artifact, "source": source, "source_revision": revision,
        "rights_evidence_reference": evidence, "modifications_made_by_project": modifications,
        "required_notices": notices, "current_tree_disposition": current_disposition,
        "historical_retention_disposition": historical_disposition,
        "privacy_security_disposition": privacy_disposition,
        "decision_authority_role": decision_policy()["decision_authority_role"] if rule else PENDING,
        "decision_record_id": decision_policy()["decision_record_id"] if rule else PENDING,
        "review_date": decision_policy()["review_date"] if rule else PENDING,
    }
    return row


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(refs: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    history_rows = []
    for path, oid in published_history_pairs(refs):
        history_rows.append(make_row(path, "published_history", oid, historical_blob_sha256(oid)))
    write_rows(HISTORY_MANIFEST, history_rows)
    current_rows = []
    for path in tracked_paths():
        oid, sha256 = indexed_blob_identity(path)
        current_rows.append(make_row(path, "current_tree", oid, sha256))
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


def policy_errors(refs: list[str]) -> list[str]:
    errors: list[str] = []
    policy = decision_policy()
    exception_keys = {
        (item["repository_relative_path"], item["git_blob_oid"])
        for item in policy["historical_privacy_exceptions"]
    }
    history_pairs = set(published_history_pairs(refs))
    for path, oid in sorted(exception_keys - history_pairs):
        errors.append(f"historical privacy exception is not in reviewed history: {path} {oid}")
    for path in tracked_paths():
        if matching_rule(path) is None:
            errors.append(f"no release decision rule for current path: {path}")
    for path, _oid in history_pairs:
        if matching_rule(path) is None:
            errors.append(f"no release decision rule for historical path: {path}")
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
    errors.extend(policy_errors(refs))
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
