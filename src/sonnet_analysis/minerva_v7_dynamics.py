"""Parse and validate Minerva V7 training dynamics without plotting dependencies."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


DYNAMICS_VERSION = "minerva_7b_v7_training_dynamics_v1"
TELEMETRY_FIELDS = (
    "stage_id", "stage_update", "global_update", "mean_training_loss",
    "preclip_global_gradient_norm", "learning_rate", "tokens_per_second",
    "elapsed_seconds", "eta_seconds", "cumulative_cost_usd",
    "first_window_index", "next_window_index", "window_identity_sha256",
)
EVALUATION_FIELDS = (
    "stage_id", "update", "historical_general_bridge_token_weighted_loss",
    "historical_non_sonnet_poetry_loss", "v7_sonnet_validation_loss",
    "modern_validation_loss", "instruction_validation_loss",
    "passes_all_gates", "is_current_selected_candidate",
)


def build_dynamics_report(*, run_dir: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    stage_config = protocol.get("stages") or protocol.get("training", {}).get("stages")
    if not isinstance(stage_config, list):
        raise ValueError("protocol has no stage configuration")
    stages = {row["stage_id"]: row for row in stage_config}
    telemetry, telemetry_tail = read_jsonl_tolerant(run_dir / "telemetry.jsonl")
    evaluations, evaluation_tail = read_jsonl_tolerant(run_dir / "evaluations.jsonl")
    sparse, sparse_tail = read_jsonl_tolerant(run_dir / "sparse_layerwise_summaries.jsonl")
    telemetry = _deduplicate(telemetry, ("stage_id", "stage_update"), "telemetry")
    evaluations = _deduplicate(evaluations, ("stage_id", "update"), "evaluations")
    sparse = _deduplicate(sparse, ("stage_id", "update"), "sparse summaries")

    issues: list[str] = []
    for label, rows in (("telemetry", telemetry), ("evaluation", evaluations), ("sparse", sparse)):
        unknown = sorted({str(row.get("stage_id")) for row in rows} - stages.keys())
        if unknown:
            issues.append(f"{label} contains unknown stages: {', '.join(unknown)}")

    stage_rows = []
    telemetry_by_stage = _group(telemetry)
    evaluation_by_stage = _group(evaluations, update_key="update")
    sparse_by_stage = _group(sparse, update_key="update")
    preceding_complete = True
    for stage_id, stage in stages.items():
        rows = telemetry_by_stage.get(stage_id, [])
        observed = [int(row["stage_update"]) for row in rows]
        expected_prefix = list(range(1, len(observed) + 1))
        if observed != expected_prefix:
            issues.append(f"{stage_id} telemetry is not a unique contiguous prefix")
        maximum = int(stage["optimizer_updates"])
        if observed and observed[-1] > maximum:
            issues.append(f"{stage_id} telemetry exceeds the frozen update count")
        complete = observed == list(range(1, maximum + 1))
        if rows and not preceding_complete:
            issues.append(f"{stage_id} starts before its preceding stage is complete")
        if not complete:
            preceding_complete = False
        validation_updates = [int(row["update"]) for row in evaluation_by_stage.get(stage_id, [])]
        invalid_eval = [value for value in validation_updates if value <= 0 or value > maximum]
        if invalid_eval:
            issues.append(f"{stage_id} has evaluation updates outside its frozen range")
        selected = [
            int(row["update"]) for row in evaluation_by_stage.get(stage_id, [])
            if row.get("is_current_selected_candidate")
        ]
        stage_rows.append(
            {
                "stage_id": stage_id,
                "expected_updates": maximum,
                "observed_updates": len(rows),
                "latest_update": observed[-1] if observed else 0,
                "complete": complete,
                "evaluation_events": len(validation_updates),
                "latest_selected_candidate_update": selected[-1] if selected else None,
                "sparse_summary_events": len(sparse_by_stage.get(stage_id, [])),
                "mean_tokens_per_second": _mean(rows, "tokens_per_second"),
                "final_elapsed_seconds": float(rows[-1]["elapsed_seconds"]) if rows else None,
                "final_cumulative_cost_usd": float(rows[-1]["cumulative_cost_usd"]) if rows else None,
            }
        )

    return {
        "dynamics_version": DYNAMICS_VERSION,
        "run_dir": str(run_dir),
        "protocol_sha256": protocol.get("protocol_sha256") or _sha256(protocol_path),
        "status": "valid_complete" if not issues and all(row["complete"] for row in stage_rows) else (
            "valid_in_progress" if not issues else "invalid"
        ),
        "issues": issues,
        "ignored_incomplete_final_jsonl": {
            "telemetry": telemetry_tail,
            "evaluations": evaluation_tail,
            "sparse_summaries": sparse_tail,
        },
        "stages": stage_rows,
        "telemetry": telemetry,
        "evaluations": evaluations,
        "sparse_summaries": sparse,
    }


def read_jsonl_tolerant(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read JSONL and tolerate only an incomplete final line from an active writer."""

    if not path.is_file():
        return [], False
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = []
    ignored_tail = False
    for index, encoded in enumerate(lines):
        complete_line = encoded.endswith((b"\n", b"\r"))
        try:
            row = json.loads(encoded)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not complete_line:
                ignored_tail = True
                continue
            raise ValueError(f"invalid JSONL row {index + 1}: {path}") from None
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {index + 1} is not an object: {path}")
        rows.append(row)
    return rows, ignored_tail


def write_dynamics_exports(report: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_dir / "dynamics_report.json",
        "telemetry": output_dir / "telemetry.csv",
        "evaluations": output_dir / "evaluations.csv",
        "stages": output_dir / "stage_summary.csv",
    }
    paths["report"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(paths["telemetry"], report["telemetry"], TELEMETRY_FIELDS)
    _write_csv(paths["evaluations"], report["evaluations"], EVALUATION_FIELDS)
    stage_fields = tuple(report["stages"][0]) if report["stages"] else ("stage_id",)
    _write_csv(paths["stages"], report["stages"], stage_fields)
    return {key: str(path) for key, path in paths.items()}


def _deduplicate(rows: Iterable[dict[str, Any]], keys: tuple[str, ...], label: str) -> list[dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        identity = tuple(row.get(key) for key in keys)
        if None in identity:
            raise ValueError(f"{label} row lacks identity fields")
        previous = result.get(identity)
        if previous is not None and previous != row:
            raise ValueError(f"conflicting duplicate {label} row: {identity}")
        result[identity] = row
    return sorted(result.values(), key=lambda row: tuple(row[key] for key in keys))


def _group(rows: Iterable[dict[str, Any]], update_key: str = "stage_update") -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stage_id"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row[update_key]))
    return dict(grouped)


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else None


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
