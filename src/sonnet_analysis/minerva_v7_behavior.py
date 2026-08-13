"""Automatic and blinded behavioral comparison for matched V7 generations."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from sonnet_evaluation.metrics import score_generated_text
from sonnet_analysis.minerva_v7_memorization import score_texts_against_reference
from sonnet_analysis.minerva_v7_registry import MODEL_STATES


BEHAVIOR_VERSION = "minerva_7b_v7_behavior_analysis_v1"
REVIEW_PLACEHOLDER = "TODO"


def analyze_matched_generations(
    *,
    state_directories: Mapping[str, Path],
    confirmatory_seed: int,
    memorization_records: Sequence[Mapping[str, str]] | None = None,
    progress: Any | None = None,
    authoritative: bool = True,
    expected_state_identities: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Score identical prompt/seed grids and separate confirmatory/replication rows."""

    rows = []
    grids = {}
    if authoritative and set(state_directories) != {state.state_id for state in MODEL_STATES}:
        raise ValueError("authoritative behavior analysis requires all seven frozen states")
    for state_id, directory in state_directories.items():
        completion = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
        if completion.get("v7_test_accessed") is not False:
            raise ValueError("matched generation does not prove test isolation")
        if expected_state_identities is not None and (
            completion.get("state_identity_sha256") != expected_state_identities.get(state_id)
        ):
            raise ValueError("matched generation completion state identity mismatch")
        if authoritative and (
            completion.get("prompt_count") != 24
            or completion.get("seeds") != [4099, 4100, 4101]
            or completion.get("output_count") != 72
            or len(completion.get("outputs", [])) != 72
        ):
            raise ValueError("authoritative behavior analysis requires the frozen 24 x 3 grid")
        grid = set()
        for output in completion["outputs"]:
            path = Path(str(output["path"]))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("matched generation completion contains an unsafe path")
            path = directory / path
            try:
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != output.get("sha256"):
                    raise ValueError("matched generation output hash mismatch")
                payload = json.loads(raw)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"matched generation output is malformed: {path}") from error
            if payload.get("state_id") != state_id or payload.get("v7_test_accessed") is not False:
                raise ValueError("matched generation state lineage mismatch")
            prompt_id = str(payload["prompt"]["id"])
            seed = int(payload["seed"])
            grid.add((prompt_id, seed))
            metrics = score_generated_text(payload["text"], payload["opening_line"])
            rows.append(
                {
                    "state_id": state_id,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "analysis_role": "confirmatory" if seed == confirmatory_seed else "exploratory_replication",
                    "author": payload["prompt"].get("author", ""),
                    "period": payload["prompt"].get("period", ""),
                    "text": payload["text"],
                    "memorization": None,
                    **metrics,
                }
            )
        grids[state_id] = grid
    if len({frozenset(grid) for grid in grids.values()}) != 1:
        raise ValueError("states do not contain an identical prompt/seed grid")
    if authoritative and any(len(grid) != 72 for grid in grids.values()):
        raise ValueError("authoritative behavior grid contains duplicate prompt/seed outputs")
    if memorization_records:
        memorization_rows = score_texts_against_reference(
            [str(row["text"]) for row in rows], memorization_records,
            progress=progress,
        )
        for row, memorization in zip(rows, memorization_rows):
            row["memorization"] = memorization
    summaries = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["state_id"], row["analysis_role"])].append(row)
    for (state_id, role), values in sorted(grouped.items()):
        summaries.append(
            {
                "state_id": state_id,
                "analysis_role": role,
                "outputs": len(values),
                "fourteen_line_rate": statistics.fmean(row["non_empty_line_count"] == 14 for row in values),
                "prompt_preservation_rate": statistics.fmean(row["prompt_preserved"] for row in values),
                "mean_repetition_ratio": statistics.fmean(row["repetition_ratio"] for row in values),
                "mean_unique_character_ratio": statistics.fmean(row["unique_character_ratio"] for row in values),
                "mean_character_count": statistics.fmean(row["character_count"] for row in values),
                "high_memorization_risk_count": sum(
                    row["memorization"] is not None
                    and row["memorization"]["risk_level"] == "high"
                    for row in values
                ),
            }
        )
    return {
        "behavior_version": BEHAVIOR_VERSION,
        "confirmatory_seed": confirmatory_seed,
        "states": list(state_directories),
        "rows": rows,
        "summaries": summaries,
        "memorization_scored": memorization_records is not None,
        "v7_test_accessed": False,
    }


def build_blinded_review(
    *, behavior_report: Mapping[str, Any], mapping_path: Path, review_path: Path
) -> dict[str, Any]:
    """Freeze deterministic blind IDs and a grammar/register/poetry/form rubric."""

    mapping = {}
    for row in behavior_report["rows"]:
        blind_id = hashlib.sha256(
            f"{BEHAVIOR_VERSION}|{row['state_id']}|{row['prompt_id']}|{row['seed']}".encode("utf-8")
        ).hexdigest()[:14]
        if blind_id in mapping:
            raise ValueError("blinded behavioral review ID collision")
        mapping[blind_id] = {
            "state_id": row["state_id"],
            "prompt_id": row["prompt_id"],
            "seed": row["seed"],
            "analysis_role": row["analysis_role"],
            "text": row["text"],
        }
    if mapping_path.is_file():
        if json.loads(mapping_path.read_text(encoding="utf-8")) != mapping:
            raise ValueError("existing blind mapping differs")
    else:
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not review_path.exists():
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(_review_markdown(mapping), encoding="utf-8")
    return {"mapping_path": str(mapping_path), "review_path": str(review_path), "output_count": len(mapping)}


def _review_markdown(mapping: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "# Minerva V7 Model-Change Blinded Review",
        "",
        "Score without consulting the private state mapping. Use 1 (poor) through 5 (strong). Form means coherence as a sonnet-like whole; line count alone is decoder-enforced.",
        "",
        "| Blind ID | Grammar | Historical Register | Poetic Quality | Sonnet/Form Coherence | Volta/Argument | Collapse | Evidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for blind_id in sorted(mapping):
        lines.append(
            f"| `{blind_id}` | {REVIEW_PLACEHOLDER} | {REVIEW_PLACEHOLDER} | {REVIEW_PLACEHOLDER} | "
            f"{REVIEW_PLACEHOLDER} | {REVIEW_PLACEHOLDER} | {REVIEW_PLACEHOLDER} | {REVIEW_PLACEHOLDER} |"
        )
    lines.extend(["", "## Outputs", ""])
    for blind_id, row in sorted(mapping.items()):
        lines.extend([f"### `{blind_id}`", "", "```text", str(row["text"]), "```", ""])
    return "\n".join(lines)
