"""Blinded review and comparison reports for Minerva 7B recovery outputs."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sonnet_evaluation.metrics import resolve_generated_path
from sonnet_evaluation.minerva_7b_quality_recovery import (
    RECOVERY_CONDITIONS,
    RECOVERY_OUTPUT_COUNT,
    RECOVERY_VERSION,
)
from sonnet_evaluation.qualitative import fenced_text_block
from sonnet_evaluation.task_acceptance import (
    score_task_format_acceptance_directory,
)


REVIEW_TITLE = "# Minerva 7B Quality Recovery: Blinded Review"
REVIEW_PLACEHOLDER = "TODO"


def build_recovery_review_artifacts(
    *,
    output_root: Path,
    mapping_path: Path,
    review_path: Path,
    automatic_report_path: Path,
    result_report_path: Path,
) -> dict[str, Any]:
    """Write scaffolds, then produce the final comparison once review is complete."""
    summary = _load_summary(output_root)
    automatic_rows, mapping = _collect_outputs(output_root, summary)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    if mapping_path.is_file():
        existing = json.loads(mapping_path.read_text(encoding="utf-8"))
        if existing != mapping:
            raise ValueError("quality-recovery blind mapping changed")
    else:
        mapping_path.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    automatic_report_path.parent.mkdir(parents=True, exist_ok=True)
    automatic_report_path.write_text(
        _build_automatic_report(automatic_rows), encoding="utf-8"
    )
    if not review_path.is_file():
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(
            _build_review_scaffold(mapping), encoding="utf-8"
        )
        return {
            "review_complete": False,
            "output_count": len(mapping),
            "mapping_path": str(mapping_path),
            "review_path": str(review_path),
        }

    judgments = parse_recovery_review(review_path, allow_incomplete=True)
    if judgments is None:
        return {
            "review_complete": False,
            "output_count": len(mapping),
            "mapping_path": str(mapping_path),
            "review_path": str(review_path),
        }
    condition_results = _aggregate_results(
        automatic_rows=automatic_rows,
        mapping=mapping,
        judgments=judgments,
    )
    result_report_path.parent.mkdir(parents=True, exist_ok=True)
    result_report_path.write_text(
        _build_result_report(condition_results), encoding="utf-8"
    )
    return {
        "review_complete": True,
        "output_count": len(mapping),
        "condition_results": condition_results,
        "result_report_path": str(result_report_path),
    }


def parse_recovery_review(
    path: Path, *, allow_incomplete: bool = False
) -> dict[str, dict[str, Any]] | None:
    """Parse the fixed yes/no review table without consulting the mapping."""
    judgments = {}
    incomplete = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            continue
        blind_id = cells[0].strip("`")
        labels = cells[1:4]
        evidence = cells[4]
        if any(label == REVIEW_PLACEHOLDER for label in labels) or evidence == REVIEW_PLACEHOLDER:
            incomplete = True
            continue
        if any(label not in {"yes", "no"} for label in labels):
            raise ValueError(f"invalid recovery judgment label: {blind_id}")
        if blind_id in judgments:
            raise ValueError(f"duplicate recovery judgment: {blind_id}")
        judgments[blind_id] = {
            "grammar": labels[0] == "yes",
            "topic": labels[1] == "yes",
            "collapse": labels[2] == "yes",
            "evidence": evidence,
        }
    if incomplete:
        if allow_incomplete:
            return None
        raise ValueError("quality-recovery review still contains TODO values")
    if len(judgments) != RECOVERY_OUTPUT_COUNT:
        raise ValueError(
            f"quality-recovery review requires {RECOVERY_OUTPUT_COUNT} judgments"
        )
    return judgments


def _load_summary(output_root: Path) -> dict[str, Any]:
    path = output_root / "recovery_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "recovery_version": RECOVERY_VERSION,
        "condition_count": len(RECOVERY_CONDITIONS),
        "output_count": RECOVERY_OUTPUT_COUNT,
        "final_test_used": False,
        "training_used": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"quality-recovery summary mismatch: {key}")
    expected_conditions = [
        (condition["condition_id"], condition["model_state"])
        for condition in RECOVERY_CONDITIONS
    ]
    actual_conditions = [
        (condition.get("condition_id"), condition.get("model_state"))
        for condition in summary.get("conditions", [])
    ]
    if actual_conditions != expected_conditions:
        raise ValueError("quality-recovery summary conditions changed")
    if any(
        condition.get("output_count") != 12
        for condition in summary["conditions"]
    ):
        raise ValueError("quality-recovery summary condition is incomplete")
    return summary


def _collect_outputs(
    output_root: Path, summary: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    mapping = {}
    for condition in summary["conditions"]:
        condition_id = condition["condition_id"]
        generation_dir = output_root / condition_id
        metadata_path = generation_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        controls = score_task_format_acceptance_directory(generation_dir)
        if len(controls) != 12:
            raise ValueError("each recovery condition must contain 12 outputs")
        controls_by_prompt = {row["prompt_id"]: row for row in controls}
        for generated in metadata["generated_files"]:
            prompt_id = generated["prompt_id"]
            control = controls_by_prompt[prompt_id]
            blind_id = hashlib.sha256(
                f"{RECOVERY_VERSION}|{condition_id}|{prompt_id}".encode("utf-8")
            ).hexdigest()[:12]
            if blind_id in mapping:
                raise ValueError("quality-recovery blind ID collision")
            output_path = resolve_generated_path(
                generated["path"], metadata_path
            )
            text = output_path.read_text(encoding="utf-8")
            mapping[blind_id] = {
                "condition_id": condition_id,
                "model_state": condition["model_state"],
                "prompt_id": prompt_id,
                "source_prompt_id": generated["source_prompt_id"],
                "opening_line": generated["opening_line"],
                "path": str(output_path),
                "text": text,
            }
            rows.append({
                "blind_id": blind_id,
                "condition_id": condition_id,
                "automatic_control_pass": control["automatic_control_pass"],
                "repetition_ratio": control["repetition_ratio"],
                "non_empty_line_count": control["non_empty_line_count"],
            })
    if len(mapping) != RECOVERY_OUTPUT_COUNT:
        raise ValueError("quality-recovery output mapping is incomplete")
    return rows, dict(sorted(mapping.items()))


def _build_review_scaffold(mapping: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        REVIEW_TITLE,
        "",
        "Do not open the local blind mapping before completing every row. "
        "`Grammar` means generally grammatical Italian; historical spelling "
        "alone is not an error. `Topic` requires one recognizable topic for at "
        "least seven generated lines. `Collapse` means severe repetition or "
        "generation degeneration. Replace every `TODO`.",
        "",
        "| Blind ID | Grammar | Topic | Collapse | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for blind_id in mapping:
        lines.append(
            f"| `{blind_id}` | TODO | TODO | TODO | TODO |"
        )
    for blind_id, row in mapping.items():
        display_text = "\n".join(
            line.rstrip() for line in str(row["text"]).splitlines()
        )
        lines.extend([
            "",
            f"## Output `{blind_id}`",
            "",
            f"Opening line: `{row['opening_line']}`",
            "",
            fenced_text_block(display_text),
        ])
    return "\n".join(lines) + "\n"


def _build_automatic_report(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Minerva 7B Quality Recovery: Automatic Controls",
        "",
        "These measurements are diagnostic. They do not establish grammar, "
        "topic continuity, metre, rhyme, or literary quality.",
        "",
        "| Condition | Controlled form | Mean repetition | Mean lines |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition in RECOVERY_CONDITIONS:
        condition_rows = [
            row for row in rows if row["condition_id"] == condition["condition_id"]
        ]
        lines.append(
            f"| `{condition['condition_id']}` | "
            f"{sum(row['automatic_control_pass'] for row in condition_rows)}/12 | "
            f"{statistics.fmean(row['repetition_ratio'] for row in condition_rows):.4f} | "
            f"{statistics.fmean(row['non_empty_line_count'] for row in condition_rows):.2f} |"
        )
    lines.extend(["", "Final-test material used: **no**.", ""])
    return "\n".join(lines)


def _aggregate_results(
    *,
    automatic_rows: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Mapping[str, Any]],
    judgments: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for condition in RECOVERY_CONDITIONS:
        condition_id = condition["condition_id"]
        blind_ids = [
            blind_id
            for blind_id, row in mapping.items()
            if row["condition_id"] == condition_id
        ]
        rows = [
            row for row in automatic_rows if row["condition_id"] == condition_id
        ]
        results.append({
            "condition_id": condition_id,
            "model_state": condition["model_state"],
            "grammar_count": sum(judgments[key]["grammar"] for key in blind_ids),
            "topic_count": sum(judgments[key]["topic"] for key in blind_ids),
            "collapse_count": sum(judgments[key]["collapse"] for key in blind_ids),
            "controlled_form_count": sum(row["automatic_control_pass"] for row in rows),
            "mean_repetition_ratio": statistics.fmean(
                row["repetition_ratio"] for row in rows
            ),
        })
    return results


def _build_result_report(results: Sequence[Mapping[str, Any]]) -> str:
    rows_by_id = {row["condition_id"]: row for row in results}
    lineage_ids = (
        "untouched_control",
        "stage_a_control",
        "stage_b_control",
    )
    decoding_ids = tuple(
        condition["condition_id"]
        for condition in RECOVERY_CONDITIONS
        if condition["model_state"] == "stage_b"
    )
    lineage_ranking = _rank([rows_by_id[key] for key in lineage_ids])
    decoding_ranking = _rank([rows_by_id[key] for key in decoding_ids])
    lines = [
        "# Minerva 7B Quality-Recovery Diagnostic Result",
        "",
        "## Condition Results",
        "",
        "| Condition | Grammar | Topic | Collapse | Controlled form | Mean repetition |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| `{row['condition_id']}` | {row['grammar_count']}/12 | "
            f"{row['topic_count']}/12 | {row['collapse_count']}/12 | "
            f"{row['controlled_form_count']}/12 | "
            f"{row['mean_repetition_ratio']:.4f} |"
        )
    lines.extend([
        "",
        "## Predeclared Rankings",
        "",
        "Ranking order is grammar descending, collapse ascending, topic "
        "descending, then repeated-character 4-gram ratio ascending.",
        "",
        "Lineage: " + " > ".join(f"`{row['condition_id']}`" for row in lineage_ranking),
        "",
        "Stage B decoding: "
        + " > ".join(f"`{row['condition_id']}`" for row in decoding_ranking),
        "",
        "This validation-only result does not authorize training and does not "
        "replace the completed final-test evaluation.",
        "",
    ])
    return "\n".join(lines)


def _rank(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -int(row["grammar_count"]),
            int(row["collapse_count"]),
            -int(row["topic_count"]),
            float(row["mean_repetition_ratio"]),
            str(row["condition_id"]),
        ),
    )
