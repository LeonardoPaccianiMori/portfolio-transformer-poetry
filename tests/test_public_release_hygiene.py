from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNED_SUFFIXES = {
    ".py", ".js", ".css", ".html", ".md", ".json", ".toml", ".yml",
    ".yaml", ".cff", ".csv", ".tsv", ".txt", ".sh", ".ini", ".cfg",
    ".conf",
}
SCANNED_ROOTS = [
    ROOT / "src", ROOT / "scripts", ROOT / "tests", ROOT / "configs",
    ROOT / "docs", ROOT / "reports", ROOT / "demo", ROOT / ".github",
    ROOT / "release", ROOT / "environments",
]
CURRENT_MANIFEST = ROOT / "release/artifact_rights_manifest.csv"
HISTORY_MANIFEST = ROOT / "release/history_rights_manifest.csv"
DECISION_POLICY = ROOT / "release/public_release_decision_policy.json"
ROOT_FILES = [ROOT / name for name in (
    "README.md", "MODEL_CARD.md", "DATA_SOURCES_AND_ATTRIBUTION.md",
    "PROJECT_SPEC.md", "AI_CONTRIBUTIONS.md", "REPRODUCIBILITY.md",
    "CHANGELOG.md", "CITATION.cff", "LICENSE.md", "NOTICE",
    "pyproject.toml", "requirements.txt",
)]
BLOCKERS = {
    "absolute_home_path": re.compile(r"/(?:home|Users)/[^\s`\"']+"),
    "file_uri": re.compile(r"file://", re.IGNORECASE),
    "private_key": re.compile(r"BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY"),
    "ssh_endpoint": re.compile(r"(?:ssh|scp)\s+(?:-[^\s]+\s+)*[^\s@]+@[^\s]+", re.IGNORECASE),
    "operational_geography": re.compile(r"\b(?:Czech|Canadian)\b", re.IGNORECASE),
    "provider_identifier": re.compile(r"\bvast\.ai\b|\binstance\s+(?:id\s*)?47607076\b", re.IGNORECASE),
    "private_agent_artifact": re.compile(r"(?:^|[\s`/])\.(?:codex|agents)(?:/|[\s`])", re.IGNORECASE),
}
ALLOWLIST = {
    ("tests/test_biblioteca_italiana.py", "file_uri"),
    ("tests/test_minerva_v7_composition.py", "file_uri"),
}


def reviewed_hygiene_allowlist() -> set[tuple[str, str, str]]:
    policy = json.loads(DECISION_POLICY.read_text(encoding="utf-8"))
    return {
        (item["repository_relative_path"], item["git_blob_oid"], item["finding"])
        for item in policy["historical_hygiene_allowlist"]
    }


def approved_manifest_files():
    if not CURRENT_MANIFEST.exists():
        return
    with CURRENT_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["current_tree_disposition"] != "approved_for_public_tree":
                continue
            if row["repository_relative_path"].startswith("data/processed/"):
                continue
            path = ROOT / row["repository_relative_path"]
            if path.is_file() and path.suffix.lower() in SCANNED_SUFFIXES:
                yield path


def scanned_files():
    for root in SCANNED_ROOTS:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in SCANNED_SUFFIXES:
                    yield path
    for path in ROOT_FILES:
        if path.exists():
            yield path
    yield from approved_manifest_files()


def test_public_material_has_no_machine_or_provider_identifiers():
    findings = []
    exact_allowlist = reviewed_hygiene_allowlist()
    for path in sorted(set(scanned_files())):
        relative = path.relative_to(ROOT).as_posix()
        if path == Path(__file__):
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in BLOCKERS.items():
            if not pattern.search(content) or (relative, label) in ALLOWLIST:
                continue
            oid = subprocess.run(
                ["git", "hash-object", path],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if (relative, oid, label) not in exact_allowlist:
                findings.append(f"{relative}: {label}")
    assert findings == []


def test_public_reports_do_not_embed_raw_evaluation_prompts_or_outputs():
    markers = (
        "### Generated Text",
        "- Prompt text:",
        '"generated_text":',
        '"opening_text":',
        "| Blind ID |",
        "| Output | Grammar |",
    )
    findings = []
    for path in sorted((ROOT / "reports").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".json"}:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in content for marker in markers):
            findings.append(path.relative_to(ROOT).as_posix())
    assert findings == []


def test_reviewed_history_has_only_exact_approved_operational_exceptions():
    policy = json.loads(DECISION_POLICY.read_text(encoding="utf-8"))
    expected_exceptions = {
        (item["repository_relative_path"], item["git_blob_oid"], "absolute_home_path")
        for item in policy["historical_privacy_exceptions"]
    }
    expected_allowlist = reviewed_hygiene_allowlist()
    seen_exceptions = set()
    seen_allowlist = set()
    findings = []
    historical_blockers = {
        key: value for key, value in BLOCKERS.items()
        if key != "private_agent_artifact"
    }
    with HISTORY_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        relative = row["repository_relative_path"]
        path = Path(relative)
        if relative.startswith("data/processed/"):
            continue
        if path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        content = subprocess.run(
            ["git", "cat-file", "blob", row["git_blob_oid"]],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
        ).stdout
        for label, pattern in historical_blockers.items():
            if not pattern.search(content):
                continue
            key = (relative, row["git_blob_oid"], label)
            if key in expected_exceptions:
                seen_exceptions.add(key)
            elif key in expected_allowlist:
                seen_allowlist.add(key)
            elif (relative, label) not in ALLOWLIST:
                findings.append(f"{relative} {row['git_blob_oid']}: {label}")
    assert findings == []
    assert seen_exceptions == expected_exceptions
    assert seen_allowlist == expected_allowlist
