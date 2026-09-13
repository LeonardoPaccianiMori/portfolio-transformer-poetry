"""Retroactive prose/rhyme scoring of the sealed Stage-3 and DPO outputs.

The module reads the frozen one-time final-test generation directory, scores
every output with the rule-based prosody checker, and compares the two systems
on paired outputs.  It only reads the sealed artifacts; it does not regenerate
or modify them.  The checker is provisional until the owner reviews the A2
packet, so these results are provisional and form-only.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

from sonnet_evaluation.sonnet_prosody import analyse_sonnet

SYSTEM_IDS = ("stage_3", "dpo")
QUATRAIN_PATTERNS = frozenset({"ABBAABBA", "ABABABAB"})
TERCET_PATTERNS = frozenset({"ABCABC", "ABCACB", "ABABAB"})


def load_sealed_outputs(
    generation_dir: Path, *, verify_hashes: bool = True
) -> list[dict[str, Any]]:
    complete = json.loads((generation_dir / "complete.json").read_text(encoding="utf-8"))
    outputs = []
    for row in complete["outputs"]:
        path = generation_dir / row["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("system_id") != row["system_id"]:
            raise ValueError(f"system mismatch in {row['path']}")
        if verify_hashes:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"sealed output hash mismatch: {row['path']}")
        outputs.append(
            {
                "path": row["path"],
                "prompt_id": row["prompt_id"],
                "seed": int(row["seed"]),
                "system_id": row["system_id"],
                "opening_line": payload["opening_line"],
                "text": payload["text"],
            }
        )
    if len(outputs) != int(complete["completed_output_count"]):
        raise ValueError("sealed output count does not match complete.json")
    return outputs


def canonical_scheme(letters: str) -> str:
    mapping: dict[str, str] = {}
    result: list[str] = []
    for letter in letters:
        if letter == "?":
            result.append("?")
            continue
        if letter not in mapping:
            mapping[letter] = chr(ord("A") + len(mapping))
        result.append(mapping[letter])
    return "".join(result)


def quatrain_scheme(scheme: str) -> str | None:
    return scheme[:8] if len(scheme) >= 8 else None


def tercet_scheme(scheme: str) -> str | None:
    return canonical_scheme(scheme[8:14]) if len(scheme) >= 14 else None


def score_output(text: str) -> dict[str, Any]:
    analysis = analyse_sonnet(text)
    lines = analysis["lines"]
    valid = sum(1 for line in lines if line["is_hendecasyllable"] is True)
    failed = sum(1 for line in lines if line["is_hendecasyllable"] is False)
    uncertain = sum(1 for line in lines if line["is_hendecasyllable"] is None)
    scheme = analysis["rhyme_scheme"]
    quatrain = quatrain_scheme(scheme)
    tercet = tercet_scheme(scheme)
    pair_count = len(lines) * (len(lines) - 1) // 2
    return {
        "line_count": analysis["line_count"],
        "structure_ok": analysis["structure_ok"],
        "stanza_pattern": list(analysis["stanza_pattern"]),
        "stanza_pattern_ok": analysis["stanza_pattern_ok"],
        "hendecasyllable_lines": valid,
        "failed_lines": failed,
        "uncertain_lines": uncertain,
        "all_lines_valid": analysis["line_count"] == 14 and valid == 14,
        "rhyme_scheme": scheme,
        "quatrain_scheme": quatrain,
        "quatrain_ok": quatrain in QUATRAIN_PATTERNS,
        "tercet_scheme": tercet,
        "tercet_ok": tercet in TERCET_PATTERNS,
        "perfect_rhyme_pairs": analysis["perfect_rhyme_pairs"],
        "soft_rhyme_pairs": analysis["soft_rhyme_pairs"],
        "rhyme_score": analysis["rhyme_score"],
        "line_pair_count": pair_count,
    }


def score_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for record in records:
        metrics = score_output(str(record["text"]))
        scored.append({**record, **metrics})
        del scored[-1]["text"]
    return scored


def pair_records(
    scored_records: Iterable[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for record in scored_records:
        key = (str(record["prompt_id"]), int(record["seed"]))
        grouped.setdefault(key, {})[str(record["system_id"])] = record
    pairs = []
    for system_map in grouped.values():
        if all(system in system_map for system in SYSTEM_IDS):
            pairs.append((system_map["stage_3"], system_map["dpo"]))
    return pairs


def aggregate(
    scored_records: Iterable[Mapping[str, Any]], system_id: str
) -> dict[str, Any]:
    rows = [row for row in scored_records if row["system_id"] == system_id]
    if not rows:
        raise ValueError(f"no records for system {system_id}")
    count = len(rows)

    def mean(field: str) -> float:
        return sum(float(row[field]) for row in rows) / count

    def rate(field: str) -> float:
        return sum(1 for row in rows if row[field]) / count

    tercets: dict[str, int] = {}
    for row in rows:
        key = row["tercet_scheme"] or "?"
        tercets[key] = tercets.get(key, 0) + 1
    return {
        "system_id": system_id,
        "output_count": count,
        "structure_ok_rate": rate("structure_ok"),
        "stanza_pattern_ok_rate": rate("stanza_pattern_ok"),
        "all_lines_valid_rate": rate("all_lines_valid"),
        "definite_valid_rate": rate("all_lines_valid"),
        "no_definite_failure_rate": sum(
            1 for row in rows if row["failed_lines"] == 0
        )
        / count,
        "quatrain_ok_rate": rate("quatrain_ok"),
        "tercet_ok_rate": rate("tercet_ok"),
        "mean_hendecasyllable_lines": mean("hendecasyllable_lines"),
        "mean_failed_lines": mean("failed_lines"),
        "mean_uncertain_lines": mean("uncertain_lines"),
        "mean_perfect_rhyme_pairs": mean("perfect_rhyme_pairs"),
        "mean_soft_rhyme_pairs": mean("soft_rhyme_pairs"),
        "mean_rhyme_score": mean("rhyme_score"),
        "tercet_scheme_distribution": dict(
            sorted(tercets.items(), key=lambda item: -item[1])
        ),
    }


def paired_comparison(
    pairs: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    field: str,
) -> dict[str, Any]:
    differences = [
        float(stage_3[field]) - float(dpo[field]) for stage_3, dpo in pairs
    ]
    count = len(differences)
    if count == 0:
        raise ValueError("no paired outputs")
    mean = sum(differences) / count
    if count > 1:
        variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
        standard_error = math.sqrt(variance / count)
        half_width = 1.96 * standard_error
    else:
        half_width = float("nan")
    positive = sum(1 for value in differences if value > 0)
    negative = sum(1 for value in differences if value < 0)
    equal = count - positive - negative
    return {
        "field": field,
        "pair_count": count,
        "mean_difference_stage_3_minus_dpo": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "positive_pairs": positive,
        "negative_pairs": negative,
        "equal_pairs": equal,
    }


def mcnemar(
    pairs: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    field: str,
) -> dict[str, Any]:
    stage_3_only = 0
    dpo_only = 0
    both = 0
    neither = 0
    for stage_3, dpo in pairs:
        left = bool(stage_3[field])
        right = bool(dpo[field])
        if left and right:
            both += 1
        elif left:
            stage_3_only += 1
        elif right:
            dpo_only += 1
        else:
            neither += 1
    discordant = stage_3_only + dpo_only
    if discordant:
        statistic = (stage_3_only - dpo_only) / math.sqrt(discordant)
        p_value = math.erfc(abs(statistic) / math.sqrt(2))
    else:
        statistic = 0.0
        p_value = 1.0
    return {
        "field": field,
        "stage_3_only": stage_3_only,
        "dpo_only": dpo_only,
        "both": both,
        "neither": neither,
        "discordant": discordant,
        "statistic": statistic,
        "p_value_normal_approximation": p_value,
    }


def format_report(
    *,
    output_count: int,
    pair_count: int,
    protocol_sha256: str,
    adapter_sha256: str,
    stage_3: Mapping[str, Any],
    dpo: Mapping[str, Any],
    comparisons: list[Mapping[str, Any]],
    mcnemar_results: list[Mapping[str, Any]],
) -> str:
    lines = [
        "# Retroactive Prosody Scoring of the Sealed Stage-3 and DPO Outputs v1",
        "",
        "Date: 2026-09-13",
        "",
        "This report scores the frozen one-time final-test outputs with the",
        "rule-based `sonnet_prosody` checker. It only reads the sealed",
        "artifacts. The checker flags were reviewed on 2026-09-13; see",
        "`reports/sonnet_prosody_review_outcome_v1.md`. The results below",
        "measure form only, not literary quality. Selection effects are not",
        "implied.",
        "",
        f"- Sealed outputs: {output_count}",
        f"- Paired outputs: {pair_count}",
        f"- Final protocol SHA-256: `{protocol_sha256}`",
        f"- DPO adapter SHA-256: `{adapter_sha256}`",
        "",
        "## Per-system form metrics",
        "",
        "| Metric | Stage 3 | DPO |",
        "|---|---:|---:|",
        f"| Outputs | {stage_3['output_count']} | {dpo['output_count']} |",
        f"| Structure OK rate | {stage_3['structure_ok_rate']:.4f} | {dpo['structure_ok_rate']:.4f} |",
        f"| Stanza pattern OK rate | {stage_3['stanza_pattern_ok_rate']:.4f} | {dpo['stanza_pattern_ok_rate']:.4f} |",
        f"| All 14 lines valid rate | {stage_3['all_lines_valid_rate']:.4f} | {dpo['all_lines_valid_rate']:.4f} |",
        f"| No definite metre failure rate | {stage_3['no_definite_failure_rate']:.4f} | {dpo['no_definite_failure_rate']:.4f} |",
        f"| Quatrain scheme OK rate | {stage_3['quatrain_ok_rate']:.4f} | {dpo['quatrain_ok_rate']:.4f} |",
        f"| Tercet scheme OK rate | {stage_3['tercet_ok_rate']:.4f} | {dpo['tercet_ok_rate']:.4f} |",
        f"| Mean hendecasyllable lines | {stage_3['mean_hendecasyllable_lines']:.3f} | {dpo['mean_hendecasyllable_lines']:.3f} |",
        f"| Mean failed lines | {stage_3['mean_failed_lines']:.3f} | {dpo['mean_failed_lines']:.3f} |",
        f"| Mean uncertain lines | {stage_3['mean_uncertain_lines']:.3f} | {dpo['mean_uncertain_lines']:.3f} |",
        f"| Mean perfect rhyme pairs | {stage_3['mean_perfect_rhyme_pairs']:.3f} | {dpo['mean_perfect_rhyme_pairs']:.3f} |",
        f"| Mean soft rhyme pairs | {stage_3['mean_soft_rhyme_pairs']:.3f} | {dpo['mean_soft_rhyme_pairs']:.3f} |",
        f"| Mean rhyme score | {stage_3['mean_rhyme_score']:.4f} | {dpo['mean_rhyme_score']:.4f} |",
        "",
        "## Paired comparisons (Stage 3 minus DPO)",
        "",
        "| Field | Mean difference | 95% CI | + pairs | - pairs | = pairs |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['field']} | {row['mean_difference_stage_3_minus_dpo']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
            f"{row['positive_pairs']} | {row['negative_pairs']} | {row['equal_pairs']} |"
        )
    lines.extend(
        [
            "",
            "## Discordant binary outcomes (McNemar normal approximation)",
            "",
            "| Field | Stage 3 only | DPO only | Both | Neither | p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mcnemar_results:
        lines.append(
            f"| {row['field']} | {row['stage_3_only']} | {row['dpo_only']} | "
            f"{row['both']} | {row['neither']} | {row['p_value_normal_approximation']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Tercet pattern distribution (top patterns)",
            "",
            f"- Stage 3: {stage_3['tercet_scheme_distribution']}",
            f"- DPO: {dpo['tercet_scheme_distribution']}",
            "",
            "## Verification",
            "",
            "- Score: `python3 scripts/score_sealed_sonnet_prosody.py`",
            "- Checker validation: `reports/sonnet_prosody_validation_v1.md`",
            "",
        ]
    )
    return "\n".join(lines)
