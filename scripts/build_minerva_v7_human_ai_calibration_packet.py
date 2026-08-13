#!/usr/bin/env python3
"""Freeze a validation-only 20-pair human--AI calibration packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_dpo_preferences import write_json_atomic
from scripts.prepare_minerva_v7_dpo_review import RUBRIC, _calibration_markdown


VERSION = "minerva_7b_v7_human_ai_calibration_validation_v1"
SELECTION_SEED = 10_957
PAIR_COUNT = 20
COMPLETION_CONTRAST_COUNT = 10
EXPECTED_ANALYSIS_SHA256 = (
    "35e84bfb31caa3c5d85a78563811ac7f2447f2efcf5606c31d9799cec0111d8b"
)
EXPECTED_PROMPT_SHA256 = (
    "2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7"
)
CELLS = ("no_labels_balanced", "no_labels_creative")


def build_validation_calibration_packet(
    *, analysis_path: Path, prompt_path: Path
) -> tuple[dict, dict]:
    """Select balanced/creative matched pairs without opening private ratings."""

    if _sha256(analysis_path) != EXPECTED_ANALYSIS_SHA256:
        raise ValueError("human-AI calibration analysis hash mismatch")
    if _sha256(prompt_path) != EXPECTED_PROMPT_SHA256:
        raise ValueError("human-AI calibration prompt hash mismatch")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    prompts = {
        str(row["id"]): row
        for row in json.loads(prompt_path.read_text(encoding="utf-8"))["prompts"]
    }
    if analysis.get("v7_test_accessed") is not False or analysis.get(
        "training_performed"
    ) is not False:
        raise ValueError("human-AI calibration input crosses authorization boundary")
    grouped = {}
    for row in analysis.get("rows", []):
        if row.get("cell_id") in CELLS:
            grouped.setdefault((str(row["prompt_id"]), int(row["seed"])), {})[
                str(row["cell_id"])
            ] = row
    candidates = []
    for (prompt_id, seed), cells in grouped.items():
        if set(cells) != set(CELLS):
            continue
        prompt = prompts[prompt_id]
        balanced = cells["no_labels_balanced"]
        creative = cells["no_labels_creative"]
        if not (_otherwise_safe(balanced) and _otherwise_safe(creative)):
            continue
        completion_difference = bool(
            balanced["ends_with_terminal_punctuation"]
            != creative["ends_with_terminal_punctuation"]
        )
        pair_type = (
            "terminal_completion_contrast"
            if completion_difference
            else "matched_literary_comparison"
        )
        candidates.append(
            {
                "prompt_id": prompt_id,
                "seed": seed,
                "period": str(prompt["period"]),
                "author_key": str(prompt["author_key"]),
                "work_key": str(prompt["work_key"]),
                "pair_type": pair_type,
                "balanced": balanced,
                "creative": creative,
            }
        )
    selected = _select(candidates)
    public_rows = []
    private_rows = []
    for row in selected:
        digest = hashlib.sha256(
            f"{VERSION}|{row['prompt_id']}|{row['seed']}".encode()
        ).hexdigest()
        balanced_is_a = random.Random(
            int(digest[:16], 16) ^ SELECTION_SEED
        ).randrange(2) == 0
        a = row["balanced"] if balanced_is_a else row["creative"]
        b = row["creative"] if balanced_is_a else row["balanced"]
        pair_id = f"calibration_pair_{digest[:16]}"
        public_rows.append(
            {
                "pair_id": pair_id,
                "pair_type": row["pair_type"],
                "opening_line": str(a["text"].splitlines()[0]),
                "candidate_a": str(a["text"]),
                "candidate_b": str(b["text"]),
            }
        )
        private_rows.append(
            {
                "pair_id": pair_id,
                "prompt_id": row["prompt_id"],
                "seed": row["seed"],
                "period": row["period"],
                "author_key": row["author_key"],
                "work_key": row["work_key"],
                "candidate_a_cell": str(a["cell_id"]),
                "candidate_b_cell": str(b["cell_id"]),
                "candidate_a_terminal_punctuation": bool(
                    a["ends_with_terminal_punctuation"]
                ),
                "candidate_b_terminal_punctuation": bool(
                    b["ends_with_terminal_punctuation"]
                ),
                "automatic_completion_is_not_a_preference_label": True,
            }
        )
    common = {
        "calibration_version": VERSION,
        "source_role": "validation_only_human_ai_calibration_not_dpo_training",
        "analysis_sha256": EXPECTED_ANALYSIS_SHA256,
        "prompt_manifest_sha256": EXPECTED_PROMPT_SHA256,
        "selection_seed": SELECTION_SEED,
        "pair_count": len(public_rows),
        "pair_type_counts": dict(
            sorted(Counter(row["pair_type"] for row in public_rows).items())
        ),
        "period_counts": dict(
            sorted(Counter(row["period"] for row in private_rows).items())
        ),
        "v7_test_accessed": False,
        "eligible_for_dpo_training": False,
    }
    return ({**common, "pairs": public_rows}, {**common, "mapping": private_rows})


def _select(candidates: list[dict]) -> list[dict]:
    completion = [
        row for row in candidates
        if row["pair_type"] == "terminal_completion_contrast"
    ]
    literary = [
        row for row in candidates
        if row["pair_type"] == "matched_literary_comparison"
    ]
    selected = _balanced_pick(completion, COMPLETION_CONTRAST_COUNT, set(), set())
    authors = {row["author_key"] for row in selected}
    works = {row["work_key"] for row in selected}
    selected.extend(
        _balanced_pick(
            literary, PAIR_COUNT - COMPLETION_CONTRAST_COUNT, authors, works
        )
    )
    if len(selected) != PAIR_COUNT:
        raise ValueError("insufficient balanced human-AI calibration pairs")
    return sorted(selected, key=lambda row: (row["pair_type"], row["period"], row["prompt_id"]))


def _balanced_pick(
    pool: list[dict], count: int, used_authors: set[str], used_works: set[str]
) -> list[dict]:
    period_counts: Counter[str] = Counter()
    selected = []
    remaining = list(pool)
    while len(selected) < count:
        if not remaining:
            raise ValueError("calibration selection pool exhausted")
        row = min(
            remaining,
            key=lambda item: (
                period_counts[item["period"]],
                item["work_key"] in used_works,
                item["author_key"] in used_authors,
                hashlib.sha256(
                    f"{SELECTION_SEED}|{item['prompt_id']}|{item['seed']}".encode()
                ).hexdigest(),
            ),
        )
        remaining.remove(row)
        selected.append(row)
        period_counts[row["period"]] += 1
        used_authors.add(row["author_key"])
        used_works.add(row["work_key"])
    return selected


def _otherwise_safe(row: dict) -> bool:
    return bool(
        row["fourteen_line"]
        and row["meta_text_free"]
        and row["no_line_at_or_above_120_characters"]
        and row["below_035_repetition_ratio"]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", type=Path,
        default=Path(
            "artifacts/local/minerva_7b_v7_stage_3_no_labels_creative/analysis/analysis.json"
        ),
    )
    parser.add_argument(
        "--prompts", type=Path,
        default=Path("configs/minerva_7b_v7_exploratory_prompts.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/human_ai_calibration"),
    )
    args = parser.parse_args()
    print(
        "minerva-v7-dpo | start job=build_human_ai_calibration device=cpu "
        "total_steps=20 progress_interval=final-only",
        flush=True,
    )
    packet, mapping = build_validation_calibration_packet(
        analysis_path=args.analysis, prompt_path=args.prompts
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet_hash = write_json_atomic(args.output_dir / "packet.json", packet)
    mapping_hash = write_json_atomic(
        args.output_dir / "mapping.private.json", mapping
    )
    (args.output_dir / "packet.md").write_text(
        _calibration_markdown(packet), encoding="utf-8"
    )
    (args.output_dir / "rubric.md").write_text(
        f"# Human--AI Calibration Rubric\n\n{RUBRIC}\n",
        encoding="utf-8",
    )
    print(
        "minerva-v7-dpo | complete pairs=20 "
        f"packet_sha256={packet_hash} mapping_sha256={mapping_hash} "
        "source_split=validation eligible_for_dpo_training=False "
        "v7_test_accessed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
