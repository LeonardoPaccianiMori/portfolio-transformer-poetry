#!/usr/bin/env python3
"""Parse and score the completed validation-only human calibration Markdown."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FIELDS = {
    "Grammar": "grammar",
    "Coherence/progression": "coherence",
    "Historical register": "historical_register",
    "Poetic force": "poetic_force",
    "Sonnet form": "form",
    "Volta/closure": "volta_closure",
}


def parse_completed_packet(text: str) -> list[dict]:
    """Parse exactly 20 complete human decisions from the frozen template."""

    chunks = re.split(r"(?=^## Pair \d+:)", text, flags=re.MULTILINE)[1:]
    if len(chunks) != 20:
        raise ValueError("completed human packet must contain exactly 20 pairs")
    rows = []
    for chunk in chunks:
        identity = re.search(r"^## Pair (\d+): `([^`]+)`", chunk, re.MULTILINE)
        if identity is None:
            raise ValueError("human packet pair identity is malformed")
        pair_id = identity.group(2)
        scores = {}
        completion = {}
        for side in ("A", "B"):
            start = f"Scores for {side} (1--5):"
            end = "Scores for B (1--5):" if side == "A" else "Preference (A/B/tie):"
            section = chunk.split(start, 1)[1].split(end, 1)[0]
            values = dict(
                re.findall(r"^- ([^:]+):\s*(.*?)\s*$", section, re.MULTILINE)
            )
            if set(values) != set(FIELDS) | {"Terminal syntax genuinely complete (yes/no)"}:
                raise ValueError(f"human score schema mismatch: {pair_id} {side}")
            scores[side] = {FIELDS[key]: int(values[key]) for key in FIELDS}
            if any(not 1 <= value <= 5 for value in scores[side].values()):
                raise ValueError("human score outside 1--5")
            answer = values["Terminal syntax genuinely complete (yes/no)"].casefold()
            if answer not in {"yes", "no"}:
                raise ValueError("human completion answer must be yes or no")
            completion[side] = answer == "yes"
        preference_match = re.search(
            r"^Preference \(A/B/tie\):\s*(A|B|tie)\s*$", chunk, re.MULTILINE
        )
        reason_match = re.search(r"^Reason:\s*(.+?)\s*$", chunk, re.MULTILINE)
        if preference_match is None or reason_match is None:
            raise ValueError(f"human preference or reason missing: {pair_id}")
        rows.append(
            {
                "pair_id": pair_id,
                "preference": preference_match.group(1),
                "scores": scores,
                "terminal_syntax_complete": completion,
                "reason": reason_match.group(1),
            }
        )
    if len({row["pair_id"] for row in rows}) != 20:
        raise ValueError("human packet contains duplicate pair identities")
    return rows


def compare_human_ai(*, human_rows: list[dict], aggregation: dict) -> dict:
    """Compare frozen human preferences against already-frozen AI majorities."""

    human = {row["pair_id"]: row for row in human_rows}
    decisions = {row["pair_id"]: row for row in aggregation.get("decisions", [])}
    if set(human) != set(decisions):
        raise ValueError("human and AI calibration pair identities differ")
    rows = []
    for pair_id in sorted(human):
        ai = decisions[pair_id]
        majority = str(ai["majority_preference"])
        comparable = majority in {"A", "B"}
        rows.append(
            {
                "pair_id": pair_id,
                "pair_type": ai["pair_type"],
                "human_preference": human[pair_id]["preference"],
                "ai_majority_preference": majority,
                "ai_vote_counts": ai["vote_counts"],
                "comparable": comparable,
                "agreement": comparable and human[pair_id]["preference"] == majority,
                "human_scores": human[pair_id]["scores"],
                "human_terminal_syntax_complete": human[pair_id]["terminal_syntax_complete"],
                "human_reason": human[pair_id]["reason"],
            }
        )
    comparable = [row for row in rows if row["comparable"]]
    agreement_count = sum(row["agreement"] for row in comparable)
    by_type = {}
    for pair_type in sorted({row["pair_type"] for row in rows}):
        selected = [row for row in comparable if row["pair_type"] == pair_type]
        by_type[pair_type] = {
            "comparable_pair_count": len(selected),
            "agreement_count": sum(row["agreement"] for row in selected),
            "agreement_rate": sum(row["agreement"] for row in selected) / len(selected),
        }
    return {
        "calibration_version": aggregation["calibration_version"],
        "source_role": "validation_only_human_ai_calibration_not_dpo_training",
        "pair_count": len(rows),
        "comparable_decisive_pair_count": len(comparable),
        "agreement_count": agreement_count,
        "agreement_rate": agreement_count / len(comparable),
        "agreement_gate_at_least_080": agreement_count / len(comparable) >= 0.80,
        "agreement_by_pair_type": by_type,
        "comparisons": rows,
        "eligible_for_dpo_training": False,
        "v7_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet-md", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/human_ai_calibration/packet.md"),
    )
    parser.add_argument(
        "--ai-aggregation", type=Path,
        default=Path(
            "artifacts/local/minerva_7b_v7_dpo/human_ai_calibration/"
            "ai_aggregation.frozen.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "artifacts/local/minerva_7b_v7_dpo/human_ai_calibration/"
            "human_ai_comparison.frozen.json"
        ),
    )
    args = parser.parse_args()
    print(
        "minerva-v7-dpo | start job=score_human_ai_calibration device=cpu "
        "total_steps=20 progress_interval=final-only",
        flush=True,
    )
    human = parse_completed_packet(args.packet_md.read_text(encoding="utf-8"))
    ai = json.loads(args.ai_aggregation.read_text(encoding="utf-8"))
    result = compare_human_ai(human_rows=human, aggregation=ai)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "minerva-v7-dpo | complete "
        f"pairs={result['pair_count']} agreement={result['agreement_rate']:.3f} "
        f"gate_passed={result['agreement_gate_at_least_080']} "
        "eligible_for_dpo_training=False v7_test_accessed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
