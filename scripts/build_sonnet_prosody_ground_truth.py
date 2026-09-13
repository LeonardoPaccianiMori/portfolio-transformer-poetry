#!/usr/bin/env python3
"""Build the frozen sonnet-prosody ground truth from the V6 sonnet corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

AUTHOR_ORDER = ("Dante Alighieri", "Francesco Petrarca")
POEMS_PER_AUTHOR = 15
LONG_S = "\u017f"
SCHEMES_PATH = ROOT / "data/metadata/sonnet_prosody_ground_truth_schemes_v1.json"
OUTPUT_PATH = ROOT / "data/metadata/sonnet_prosody_ground_truth_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemes", type=Path, default=SCHEMES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def eligible(row: dict[str, str], excluded: set[str]) -> bool:
    if row["poem_id"] in excluded:
        return False
    lines = clean_lines(ROOT / row["clean_text_path"])
    if len(lines) != 14:
        return False
    text = "\n".join(lines)
    return "[" not in text and "]" not in text


def select(rows: list[dict[str, str]], author: str, excluded: set[str]) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row["author"] == author
        and row["split_expanded_with_petrarch"] == "train"
        and eligible(row, excluded)
    ]
    last = len(candidates) - 1
    positions = sorted({round(i * last / (POEMS_PER_AUTHOR - 1)) for i in range(POEMS_PER_AUTHOR)})
    if len(positions) != POEMS_PER_AUTHOR:
        raise ValueError(f"even spacing collapsed for {author}")
    return [candidates[position] for position in positions]


def canonical_scheme(scheme: str) -> str:
    mapping: dict[str, str] = {}
    result: list[str] = []
    for letter in scheme.replace(" ", "").upper():
        if letter not in mapping:
            mapping[letter] = chr(ord("A") + len(mapping))
        result.append(mapping[letter])
    return "".join(result)


def poem_sha256(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    annotation = json.loads(args.schemes.read_text(encoding="utf-8"))
    scheme_rows = annotation["schemes"]
    excluded = {entry["poem_id"] for entry in annotation.get("excluded_poems", [])}
    manifest_path = ROOT / annotation["corpus_manifest"]
    rows = load_manifest(manifest_path)

    poems: list[dict[str, object]] = []
    for author in AUTHOR_ORDER:
        expected_ids = [row["poem_id"] for row in scheme_rows if row["author"] == author]
        selected = select(rows, author, excluded)
        selected_ids = [row["poem_id"] for row in selected]
        if selected_ids != expected_ids:
            raise ValueError(
                f"frozen selection mismatch for {author}:\n"
                f"selected: {selected_ids}\nexpected: {expected_ids}"
            )
        for row, scheme_row in zip(selected, [r for r in scheme_rows if r["author"] == author], strict=True):
            raw_lines = clean_lines(ROOT / row["clean_text_path"])
            long_s_total = 0
            lines: list[str] = []
            for raw_line in raw_lines:
                long_s_total += raw_line.count(LONG_S)
                lines.append(raw_line.replace(LONG_S, "s"))
            scheme = canonical_scheme(scheme_row["scheme"])
            if len(scheme) != len(lines):
                raise ValueError(f"scheme length mismatch for {row['poem_id']}")
            poems.append(
                {
                    "poem_id": row["poem_id"],
                    "author": author,
                    "title_or_first_line": row["title_or_first_line"],
                    "source_archive": row["source_archive"],
                    "source_url": row["source_url"],
                    "license_notes": row["license_notes"],
                    "clean_text_path": row["clean_text_path"],
                    "long_s_replacement_count": long_s_total,
                    "poem_sha256": poem_sha256(lines),
                    "expected_scheme": scheme,
                    "lines": [
                        {
                            "line_number": index + 1,
                            "text": line,
                            "expected_metre": "hendecasyllable",
                            "rhyme_class": scheme[index],
                        }
                        for index, line in enumerate(lines)
                    ],
                }
            )

    payload = {
        "schema_version": 1,
        "selection_rule": annotation["selection_rule"],
        "annotation_method": annotation["annotation_method"],
        "source_manifest": annotation["corpus_manifest"],
        "source_split": annotation["source_split"],
        "excluded_poems": annotation.get("excluded_poems", []),
        "poem_count": len(poems),
        "line_count": sum(len(poem["lines"]) for poem in poems),
        "poems": poems,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote ground truth: {args.output}")
    print(f"poems: {payload['poem_count']} lines: {payload['line_count']}")


if __name__ == "__main__":
    main()
