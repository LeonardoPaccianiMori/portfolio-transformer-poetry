from __future__ import annotations

import csv
import re
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


def approved_manifest_files():
    if not CURRENT_MANIFEST.exists():
        return
    with CURRENT_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["current_tree_disposition"] != "approved_for_public_tree":
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
    for path in sorted(set(scanned_files())):
        relative = path.relative_to(ROOT).as_posix()
        if path == Path(__file__):
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in BLOCKERS.items():
            if pattern.search(content) and (relative, label) not in ALLOWLIST:
                findings.append(f"{relative}: {label}")
    assert findings == []
