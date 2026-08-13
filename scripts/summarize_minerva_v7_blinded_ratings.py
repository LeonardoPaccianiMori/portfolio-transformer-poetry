#!/usr/bin/env python3
"""Join frozen Minerva V7 blind ratings to identities and summarize them."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCORE_FIELDS = (
    "grammar",
    "historical_register",
    "poetic_quality",
    "sonnet_form_coherence",
    "volta_argument",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group-fields",
        nargs="+",
        required=True,
        help="Mapping fields that identify the comparison cell.",
    )
    parser.add_argument("--paired-left", help="Baseline system for a paired comparison.")
    parser.add_argument("--paired-right", help="Comparison system for a paired comparison.")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=17211)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ratings(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["blind_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("ratings contain duplicate blind IDs")
    for row in rows:
        for field in SCORE_FIELDS:
            value = row.get(field)
            if not isinstance(value, int) or not 1 <= value <= 5:
                raise ValueError(f"invalid {field} for blind ID {row.get('blind_id')}")
        for field in ("meta_text", "truncation"):
            if row.get(field) not in {"yes", "no"}:
                raise ValueError(f"invalid {field} for blind ID {row.get('blind_id')}")
    return rows


def qualifies_moderate(row: dict[str, Any]) -> bool:
    return (
        row["meta_text"] == "no"
        and row["truncation"] == "no"
        and not bool(row.get("collapse", False))
        and min(int(row[field]) for field in SCORE_FIELDS) >= 3
    )


def qualifies_strict(row: dict[str, Any]) -> bool:
    return (
        row["meta_text"] == "no"
        and row["truncation"] == "no"
        and not bool(row.get("collapse", False))
        and int(row["grammar"]) >= 4
        and int(row["historical_register"]) >= 3
        and int(row["poetic_quality"]) >= 4
        and int(row["sonnet_form_coherence"]) >= 4
        and int(row["volta_argument"]) >= 3
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": len(rows)}
    for field in SCORE_FIELDS:
        summary[field] = {
            "mean": statistics.fmean(float(row[field]) for row in rows),
            "counts": dict(sorted(Counter(str(row[field]) for row in rows).items())),
        }
    for field in ("meta_text", "truncation"):
        summary[f"{field}_counts"] = dict(sorted(Counter(str(row[field]) for row in rows).items()))
    if any("collapse" in row for row in rows):
        summary["collapse_counts"] = dict(
            sorted(Counter(str(bool(row.get("collapse", False))).lower() for row in rows).items())
        )
    summary["moderate_clean_count"] = sum(qualifies_moderate(row) for row in rows)
    summary["strict_good_count"] = sum(qualifies_strict(row) for row in rows)
    return summary


def paired_comparison(
    rows: list[dict[str, Any]],
    *,
    left: str,
    right: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare two systems after matching each rating by prompt and seed."""
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    lookup = {
        (str(row["system_id"]), str(row["prompt_id"]), int(row["seed"])): row
        for row in rows
    }
    pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in rows:
        if str(row["system_id"]) != left:
            continue
        prompt_id, row_seed = str(row["prompt_id"]), int(row["seed"])
        counterpart = lookup.get((right, prompt_id, row_seed))
        if counterpart is None:
            raise ValueError(f"missing paired rating for prompt {prompt_id} seed {row_seed}")
        if prompt_id in pairs:
            raise ValueError(f"multiple paired seeds for prompt {prompt_id}")
        pairs[prompt_id] = (row, counterpart)
    expected_right = sum(str(row["system_id"]) == right for row in rows)
    if len(pairs) != expected_right or not pairs:
        raise ValueError("paired systems do not have equal complete coverage")

    metric_functions = {
        **{field: lambda row, current=field: float(row[current]) for field in SCORE_FIELDS},
        "meta_text_free": lambda row: float(row["meta_text"] == "no"),
        "complete": lambda row: float(row["truncation"] == "no"),
        "collapse_free": lambda row: float(not bool(row.get("collapse", False))),
        "moderate_clean": lambda row: float(qualifies_moderate(row)),
        "strict_good": lambda row: float(qualifies_strict(row)),
    }
    result: dict[str, Any] = {
        "left_system_id": left,
        "right_system_id": right,
        "paired_prompts": len(pairs),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }
    prompt_ids = sorted(pairs)
    for index, (metric, value) in enumerate(metric_functions.items()):
        differences = {
            prompt_id: value(right_row) - value(left_row)
            for prompt_id, (left_row, right_row) in pairs.items()
        }
        rng = random.Random(seed + index)
        distribution = sorted(
            statistics.fmean(differences[rng.choice(prompt_ids)] for _ in prompt_ids)
            for _ in range(resamples)
        )
        result[metric] = {
            "mean_paired_change": statistics.fmean(differences.values()),
            "ci_low": distribution[int(0.025 * (resamples - 1))],
            "ci_high": distribution[int(0.975 * (resamples - 1))],
        }
    return result


def main() -> None:
    args = parse_args()
    ratings = load_ratings(args.ratings)
    mapping_document = json.loads(args.mapping.read_text(encoding="utf-8"))
    mapping = mapping_document.get("mapping")
    if not isinstance(mapping, list):
        raise ValueError("mapping document has no mapping list")
    by_id = {str(row["blind_id"]): row for row in mapping}
    if len(by_id) != len(mapping):
        raise ValueError("mapping contains duplicate blind IDs")
    rating_ids = {str(row["blind_id"]) for row in ratings}
    if rating_ids != set(by_id):
        raise ValueError("rating and mapping blind IDs differ")

    joined = []
    for rating in ratings:
        identity = by_id[str(rating["blind_id"])]
        missing = [field for field in args.group_fields if field not in identity]
        if missing:
            raise ValueError(f"mapping lacks group fields: {missing}")
        joined.append(
            {
                **rating,
                **{field: identity[field] for field in args.group_fields},
                **{
                    field: identity[field]
                    for field in ("prompt_id", "seed", "system_id")
                    if field in identity
                },
            }
        )

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        groups[tuple(str(row[field]) for field in args.group_fields)].append(row)
    group_summaries = []
    for key, rows in sorted(groups.items()):
        group_summaries.append(
            {
                **dict(zip(args.group_fields, key, strict=True)),
                **summarize(rows),
            }
        )

    document = {
        "analysis_version": "minerva_v7_blinded_rating_summary_v1",
        "ratings_sha256": sha256(args.ratings),
        "mapping_sha256": sha256(args.mapping),
        "reviewer_type": "assistant_ai_analyst",
        "ratings_frozen_before_unblinding": True,
        "group_fields": args.group_fields,
        "overall": summarize(joined),
        "groups": group_summaries,
        "qualified_blind_ids": {
            "moderate_clean": sorted(str(row["blind_id"]) for row in joined if qualifies_moderate(row)),
            "strict_good": sorted(str(row["blind_id"]) for row in joined if qualifies_strict(row)),
        },
        "v7_test_accessed": bool(mapping_document.get("v7_test_accessed", False)),
        "retuning_after_test_forbidden": bool(
            mapping_document.get("retuning_after_test_forbidden", False)
        ),
    }
    if bool(args.paired_left) != bool(args.paired_right):
        raise ValueError("paired-left and paired-right must be provided together")
    if args.paired_left and args.paired_right:
        document["paired_comparison"] = paired_comparison(
            joined,
            left=args.paired_left,
            right=args.paired_right,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"minerva-v7-research | complete blinded_rows={len(joined)} "
        f"groups={len(group_summaries)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
