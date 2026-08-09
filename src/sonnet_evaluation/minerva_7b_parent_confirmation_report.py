"""Blinded review and gates for the Minerva 7B parent confirmation."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sonnet_evaluation.memorization import (
    load_training_records,
    score_generation_memorization,
)
from sonnet_evaluation.metrics import resolve_generated_path
from sonnet_evaluation.minerva_7b_parent_confirmation import (
    CONFIRMATION_CONDITIONS,
    CONFIRMATION_OUTPUT_COUNT,
    CONFIRMATION_PROMPT_COUNT,
    CONFIRMATION_THRESHOLDS,
    CONFIRMATION_VERSION,
)
from sonnet_evaluation.qualitative import fenced_text_block
from sonnet_evaluation.task_acceptance import (
    score_task_format_acceptance_directory,
)


REVIEW_TITLE = "# Minerva 7B Parent-Decoding Confirmation: Blinded Review"
REVIEW_PLACEHOLDER = "TODO"


def build_parent_confirmation_review_artifacts(
    *,
    output_root: Path,
    mapping_path: Path,
    review_path: Path,
    automatic_report_path: Path,
    result_report_path: Path,
    repo_root: Path,
    manifest_path: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build blind controls, then aggregate conditions after review."""
    summary = _load_summary(output_root)
    _report(progress, "loading V6 training poems for overlap checks")
    training_records = load_training_records(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset="expanded_with_petrarch",
        split="train",
    )
    rows, mapping = _collect_outputs(
        output_root=output_root,
        summary=summary,
        training_records=training_records,
        progress=progress,
    )
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    if mapping_path.is_file():
        existing = json.loads(mapping_path.read_text(encoding="utf-8"))
        if existing != mapping:
            raise ValueError("parent confirmation blind mapping changed")
    else:
        mapping_path.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    automatic_report_path.parent.mkdir(parents=True, exist_ok=True)
    automatic_report_path.write_text(
        _build_automatic_report(rows), encoding="utf-8"
    )
    if not review_path.is_file():
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(_build_review_scaffold(mapping), encoding="utf-8")
        return {
            "review_complete": False,
            "output_count": len(mapping),
            "mapping_path": str(mapping_path),
            "review_path": str(review_path),
        }

    judgments = parse_parent_confirmation_review(
        review_path, allow_incomplete=True
    )
    if judgments is None:
        return {
            "review_complete": False,
            "output_count": len(mapping),
            "mapping_path": str(mapping_path),
            "review_path": str(review_path),
        }

    condition_results = _aggregate_results(
        automatic_rows=rows,
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


def parse_parent_confirmation_review(
    path: Path, *, allow_incomplete: bool = False
) -> dict[str, dict[str, Any]] | None:
    """Parse all fixed blind labels without consulting condition identity."""
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
        if (
            any(label == REVIEW_PLACEHOLDER for label in labels)
            or evidence == REVIEW_PLACEHOLDER
        ):
            incomplete = True
            continue
        if any(label not in {"yes", "no"} for label in labels):
            raise ValueError(f"invalid parent confirmation label: {blind_id}")
        if blind_id in judgments:
            raise ValueError(f"duplicate parent confirmation judgment: {blind_id}")
        judgments[blind_id] = {
            "grammar": labels[0] == "yes",
            "topic": labels[1] == "yes",
            "collapse": labels[2] == "yes",
            "evidence": evidence,
        }
    if incomplete:
        if allow_incomplete:
            return None
        raise ValueError("parent confirmation review still contains TODO values")
    if len(judgments) != CONFIRMATION_OUTPUT_COUNT:
        raise ValueError(
            "parent confirmation review requires "
            f"{CONFIRMATION_OUTPUT_COUNT} judgments"
        )
    return judgments


def _load_summary(output_root: Path) -> dict[str, Any]:
    summary = json.loads(
        (output_root / "confirmation_summary.json").read_text(encoding="utf-8")
    )
    expected = {
        "confirmation_version": CONFIRMATION_VERSION,
        "condition_count": len(CONFIRMATION_CONDITIONS),
        "output_count": CONFIRMATION_OUTPUT_COUNT,
        "final_test_used": False,
        "training_used": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"parent confirmation summary mismatch: {key}")
    expected_conditions = [
        (condition["condition_id"], condition["model_state"])
        for condition in CONFIRMATION_CONDITIONS
    ]
    actual_conditions = [
        (condition.get("condition_id"), condition.get("model_state"))
        for condition in summary.get("conditions", [])
    ]
    if actual_conditions != expected_conditions:
        raise ValueError("parent confirmation summary conditions changed")
    if any(
        condition.get("output_count") != CONFIRMATION_PROMPT_COUNT
        for condition in summary["conditions"]
    ):
        raise ValueError("parent confirmation summary condition is incomplete")
    return summary


def _collect_outputs(
    *,
    output_root: Path,
    summary: Mapping[str, Any],
    training_records: list[dict[str, str]],
    progress: Callable[[str], None] | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    mapping = {}
    for index, condition in enumerate(summary["conditions"], start=1):
        condition_id = condition["condition_id"]
        generation_dir = output_root / condition_id
        metadata_path = generation_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        controls = score_task_format_acceptance_directory(generation_dir)
        memorization = score_generation_memorization(
            generation_dir=generation_dir,
            training_records=training_records,
        )
        if (
            len(controls) != CONFIRMATION_PROMPT_COUNT
            or len(memorization) != CONFIRMATION_PROMPT_COUNT
        ):
            raise ValueError("each parent confirmation condition needs 24 outputs")
        controls_by_prompt = {row["prompt_id"]: row for row in controls}
        memory_by_prompt = {row["prompt_id"]: row for row in memorization}
        for generated in metadata["generated_files"]:
            prompt_id = generated["prompt_id"]
            control = controls_by_prompt[prompt_id]
            memory = memory_by_prompt[prompt_id]
            blind_id = hashlib.sha256(
                f"{CONFIRMATION_VERSION}|{condition_id}|{prompt_id}".encode(
                    "utf-8"
                )
            ).hexdigest()[:12]
            if blind_id in mapping:
                raise ValueError("parent confirmation blind ID collision")
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
                "memorization_risk": memory["risk_level"],
            })
        _report(
            progress,
            f"scored condition {index}/{len(CONFIRMATION_CONDITIONS)}: "
            f"{condition_id}",
        )
    if len(mapping) != CONFIRMATION_OUTPUT_COUNT:
        raise ValueError("parent confirmation output mapping is incomplete")
    return rows, dict(sorted(mapping.items()))


def _build_review_scaffold(mapping: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        REVIEW_TITLE,
        "",
        "Do not open the local blind mapping before completing every row. "
        "`Grammar` means generally grammatical Italian; historical spelling "
        "and poetic inversion alone are not errors. `Topic` requires one "
        "recognizable topic for at least seven generated lines. `Collapse` "
        "means severe repetition or generation degeneration. Replace every "
        "`TODO`.",
        "",
        "| Blind ID | Grammar | Topic | Collapse | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for blind_id in mapping:
        lines.append(f"| `{blind_id}` | TODO | TODO | TODO | TODO |")
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
        "# Minerva 7B Parent-Decoding Confirmation: Automatic Controls",
        "",
        "These measurements do not establish grammar, topic continuity, metre, "
        "rhyme, or literary quality.",
        "",
        "| Condition | Controlled form | High-risk overlap | Mean repetition | Mean lines |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONFIRMATION_CONDITIONS:
        condition_rows = [
            row
            for row in rows
            if row["condition_id"] == condition["condition_id"]
        ]
        lines.append(
            f"| `{condition['condition_id']}` | "
            f"{sum(row['automatic_control_pass'] for row in condition_rows)}/24 | "
            f"{sum(row['memorization_risk'] == 'high' for row in condition_rows)}/24 | "
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
    for condition in CONFIRMATION_CONDITIONS:
        condition_id = condition["condition_id"]
        blind_ids = [
            blind_id
            for blind_id, row in mapping.items()
            if row["condition_id"] == condition_id
        ]
        rows = [
            row for row in automatic_rows if row["condition_id"] == condition_id
        ]
        result = {
            "condition_id": condition_id,
            "model_state": condition["model_state"],
            "grammar_count": sum(judgments[key]["grammar"] for key in blind_ids),
            "topic_count": sum(judgments[key]["topic"] for key in blind_ids),
            "collapse_count": sum(judgments[key]["collapse"] for key in blind_ids),
            "controlled_form_count": sum(
                row["automatic_control_pass"] for row in rows
            ),
            "high_risk_memorization_count": sum(
                row["memorization_risk"] == "high" for row in rows
            ),
            "mean_repetition_ratio": statistics.fmean(
                row["repetition_ratio"] for row in rows
            ),
        }
        result["gate_passed"] = _condition_passes(result)
        results.append(result)
    return results


def _condition_passes(row: Mapping[str, Any]) -> bool:
    return bool(
        row["controlled_form_count"]
        >= CONFIRMATION_THRESHOLDS["controlled_form_min"]
        and row["grammar_count"] >= CONFIRMATION_THRESHOLDS["grammar_min"]
        and row["topic_count"] >= CONFIRMATION_THRESHOLDS["topic_min"]
        and row["collapse_count"] <= CONFIRMATION_THRESHOLDS["collapse_max"]
        and row["high_risk_memorization_count"]
        <= CONFIRMATION_THRESHOLDS["high_risk_memorization_max"]
    )


def _build_result_report(results: Sequence[Mapping[str, Any]]) -> str:
    ranking = sorted(
        results,
        key=lambda row: (
            -int(row["gate_passed"]),
            -int(row["grammar_count"]),
            int(row["collapse_count"]),
            -int(row["topic_count"]),
            -int(row["controlled_form_count"]),
            float(row["mean_repetition_ratio"]),
            str(row["condition_id"]),
        ),
    )
    passing = [row for row in ranking if row["gate_passed"]]
    if passing:
        decision = (
            "Validation candidate selected: "
            f"`{passing[0]['condition_id']}`. A separately frozen final "
            "confirmation is required before replacing the published model."
        )
    else:
        decision = (
            "No condition passed. Design of a mixed-corpus full-weight "
            "calibration is permitted, but training remains unauthorized until "
            "that protocol is approved and frozen."
        )
    lines = [
        "# Minerva 7B Parent-Decoding Confirmation Result",
        "",
        "| Condition | Grammar | Topic | Collapse | Form | High-risk | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            f"| `{row['condition_id']}` | {row['grammar_count']}/24 | "
            f"{row['topic_count']}/24 | {row['collapse_count']}/24 | "
            f"{row['controlled_form_count']}/24 | "
            f"{row['high_risk_memorization_count']}/24 | "
            f"{'pass' if row['gate_passed'] else 'fail'} |"
        )
    lines.extend([
        "",
        "Ranking: " + " > ".join(
            f"`{row['condition_id']}`" for row in ranking
        ),
        "",
        decision,
        "",
        "Final-test material used: **no**.",
        "",
    ])
    return "\n".join(lines)


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
