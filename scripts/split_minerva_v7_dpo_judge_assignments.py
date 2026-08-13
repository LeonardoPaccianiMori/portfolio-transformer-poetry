#!/usr/bin/env python3
"""Split frozen DPO judge assignments into deterministic bounded packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def split_assignments(payload: dict, *, chunk_size: int) -> list[dict]:
    """Return judge-specific packets without changing pair or judge identities."""

    if chunk_size <= 0:
        raise ValueError("judge chunk size must be positive")
    judge_ids = tuple(str(value) for value in payload.get("judge_ids", []))
    if len(judge_ids) != 3 or len(set(judge_ids)) != 3:
        raise ValueError("assignment manifest must name three distinct judges")
    assignments = list(payload.get("assignments", []))
    if len(assignments) != int(payload.get("assignment_count", -1)):
        raise ValueError("assignment manifest count mismatch")
    packets = []
    for judge_id in judge_ids:
        rows = sorted(
            (
                row for row in assignments
                if str(row.get("judge_id")) == judge_id
            ),
            key=lambda row: str(row["pair"]["pair_id"]),
        )
        if len(rows) != int(payload.get("pair_count", -1)):
            raise ValueError("judge does not have exactly one assignment per pair")
        for start in range(0, len(rows), chunk_size):
            selected = rows[start : start + chunk_size]
            packets.append(
                {
                    "preference_version": payload["preference_version"],
                    "judge_id": judge_id,
                    "chunk_index": start // chunk_size,
                    "assignment_count": len(selected),
                    "assignments": selected,
                    "blind_to_candidate_identity_and_recipe": True,
                    "v7_test_accessed": False,
                }
            )
    return packets


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
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/local/minerva_7b_v7_dpo/review/authoritative/judge_packets"
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=30)
    args = parser.parse_args()
    payload = json.loads(args.assignments.read_text(encoding="utf-8"))
    packets = split_assignments(payload, chunk_size=args.chunk_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "minerva-v7-dpo | start job=split_judge_assignments device=cpu "
        f"total_steps={len(packets)} progress_interval=one_packet",
        flush=True,
    )
    for packet in packets:
        path = args.output_dir / (
            f"{packet['judge_id']}_chunk_{packet['chunk_index']:03d}.json"
        )
        path.write_text(
            json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "minerva-v7-dpo | packet "
            f"judge={packet['judge_id']} chunk={packet['chunk_index']} "
            f"assignments={packet['assignment_count']}",
            flush=True,
        )
    print(
        "minerva-v7-dpo | complete "
        f"packets={len(packets)} assignments={payload['assignment_count']} "
        "v7_test_accessed=False training_performed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
