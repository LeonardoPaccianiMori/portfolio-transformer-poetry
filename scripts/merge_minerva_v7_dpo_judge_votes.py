#!/usr/bin/env python3
"""Validate and merge completed blinded DPO judge-vote chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_dpo_preferences import validate_vote


def merge_vote_chunks(*, assignment_manifest: dict, vote_paths: list[Path]) -> list[dict]:
    """Return votes only when every frozen assignment appears exactly once."""

    expected = {
        (str(row["pair"]["pair_id"]), str(row["judge_id"]))
        for row in assignment_manifest.get("assignments", [])
    }
    if len(expected) != int(assignment_manifest.get("assignment_count", -1)):
        raise ValueError("frozen assignment manifest is inconsistent")
    rows = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(vote_paths):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = validate_vote(json.loads(line))
            identity = (str(row["pair_id"]), str(row["judge_id"]))
            if identity not in expected:
                raise ValueError("vote references an assignment outside the freeze")
            if identity in seen:
                raise ValueError("duplicate vote across judge chunks")
            seen.add(identity)
            rows.append(row)
    if seen != expected:
        raise ValueError(
            f"incomplete blinded vote set: found {len(seen)} of {len(expected)}"
        )
    return sorted(rows, key=lambda row: (str(row["pair_id"]), str(row["judge_id"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path(
            "artifacts/local/minerva_7b_v7_dpo/review/authoritative/"
            "judge_assignments.json"
        ),
    )
    parser.add_argument(
        "--votes-dir",
        type=Path,
        default=Path(
            "artifacts/local/minerva_7b_v7_dpo/review/authoritative/votes"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/local/minerva_7b_v7_dpo/review/authoritative/"
            "ai_votes.frozen.jsonl"
        ),
    )
    args = parser.parse_args()
    manifest = json.loads(args.assignments.read_text(encoding="utf-8"))
    vote_paths = sorted(args.votes_dir.glob("*.votes.jsonl"))
    print(
        "minerva-v7-dpo | start job=merge_blinded_votes device=cpu "
        f"total_steps={manifest['assignment_count']} progress_interval=final-only",
        flush=True,
    )
    rows = merge_vote_chunks(
        assignment_manifest=manifest,
        vote_paths=vote_paths,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        "minerva-v7-dpo | complete "
        f"votes={len(rows)} pairs={manifest['pair_count']} judges=3 "
        "v7_test_accessed=False training_performed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
