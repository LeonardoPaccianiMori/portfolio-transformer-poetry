"""Ground-truth validation for the rule-based sonnet prosody checker.

The validation scores the frozen Dante and Petrarch ground truth, reports the
line-level metre errors and poem-level rhyme errors, runs the checker over the
full V6 sonnet corpus, and writes the owner review packet.  A passing gate is
necessary but not sufficient: the checker stays provisional until the owner
reviews the packet.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from sonnet_evaluation.sonnet_prosody import (
    analyse_sonnet,
    soft_rhyme_key,
)

MIN_METRE_ACCURACY = 0.95
MIN_METRE_COVERAGE = 0.90
MIN_RHYME_AGREEMENT = 0.90
REVIEW_LIMIT = 100


def load_ground_truth(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported ground-truth schema version")
    return payload


def score_ground_truth(
    ground_truth: Mapping[str, Any],
    *,
    stress_overrides: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for poem in ground_truth["poems"]:
        expected_lines = poem["lines"]
        text = "\n".join(line["text"] for line in expected_lines)
        analysis = analyse_sonnet(text, stress_overrides=stress_overrides)
        if len(analysis["lines"]) != len(expected_lines):
            raise ValueError(f"checker line count mismatch for {poem['poem_id']}")
        expected_classes = [line["rhyme_class"] for line in expected_lines]
        observed_classes = list(analysis["rhyme_scheme"])
        observed_soft_classes = [
            soft_rhyme_key(line["rhyme_key"]) for line in analysis["lines"]
        ]
        observed_soft_scheme = labeled_scheme(observed_soft_classes)
        agreement = pairwise_agreement(expected_classes, observed_classes)
        results.append(
            {
                "poem_id": poem["poem_id"],
                "author": poem["author"],
                "expected_scheme": poem["expected_scheme"],
                "observed_scheme": analysis["rhyme_scheme"],
                "scheme_match": analysis["rhyme_scheme"] == poem["expected_scheme"],
                "observed_soft_scheme": observed_soft_scheme,
                "soft_scheme_match": observed_soft_scheme == poem["expected_scheme"],
                "pairwise_rhyme_agreement": agreement,
                "metre": [
                    {
                        "line_number": expected["line_number"],
                        "text": expected["text"],
                        "expected": expected["expected_metre"],
                        "observed": observed["is_hendecasyllable"],
                        "syllable_count": observed["syllable_count"],
                        "verse_type": observed["verse_type"],
                        "uncertain": observed["uncertain"],
                        "uncertainty_reasons": list(observed["uncertainty_reasons"]),
                    }
                    for expected, observed in zip(
                        expected_lines, analysis["lines"], strict=True
                    )
                ],
            }
        )
    return results


def labeled_scheme(keys: Iterable[str | None]) -> str:
    mapping: dict[str, str] = {}
    letters: list[str] = []
    for key in keys:
        if key is None:
            letters.append("?")
            continue
        if key not in mapping:
            mapping[key] = chr(ord("A") + len(mapping))
        letters.append(mapping[key])
    return "".join(letters)


def pairwise_agreement(
    expected: Iterable[str], observed: Iterable[str]
) -> float | None:
    expected_list = list(expected)
    observed_list = list(observed)
    if len(expected_list) != len(observed_list):
        raise ValueError("expected and observed labels differ in length")
    total = 0
    same = 0
    for left in range(len(expected_list)):
        for right in range(left + 1, len(expected_list)):
            total += 1
            expected_same = expected_list[left] == expected_list[right]
            observed_same = observed_list[left] == observed_list[right]
            if expected_same == observed_same:
                same += 1
    return same / total if total else None


def metre_metrics(poem_results: list[dict[str, Any]]) -> dict[str, Any]:
    lines = [line for poem in poem_results for line in poem["metre"]]
    total = len(lines)
    correct = sum(1 for line in lines if line["observed"] is True)
    incorrect = sum(1 for line in lines if line["observed"] is False)
    uncertain = sum(1 for line in lines if line["observed"] is None)
    definite = correct + incorrect
    accuracy = correct / definite if definite else None
    coverage = definite / total if total else None
    gate_met = accuracy is not None and accuracy >= MIN_METRE_ACCURACY
    coverage_warning = coverage is not None and coverage < MIN_METRE_COVERAGE
    reasons = Counter(
        reason
        for line in lines
        for reason in line["uncertainty_reasons"]
    )
    return {
        "line_count": total,
        "correct": correct,
        "incorrect": incorrect,
        "uncertain": uncertain,
        "definite": definite,
        "accuracy": accuracy,
        "coverage": coverage,
        "gate_met": gate_met,
        "coverage_warning": coverage_warning,
        "uncertainty_reasons": dict(reasons),
    }


def rhyme_metrics(poem_results: list[dict[str, Any]]) -> dict[str, Any]:
    poem_count = len(poem_results)
    exact = sum(1 for poem in poem_results if poem["scheme_match"])
    soft_exact = sum(1 for poem in poem_results if poem["soft_scheme_match"])
    agreements = [
        poem["pairwise_rhyme_agreement"]
        for poem in poem_results
        if poem["pairwise_rhyme_agreement"] is not None
    ]
    mean_agreement = sum(agreements) / len(agreements) if agreements else None
    scheme_accuracy = exact / poem_count if poem_count else None
    soft_scheme_accuracy = soft_exact / poem_count if poem_count else None
    gate_met = (
        mean_agreement is not None
        and mean_agreement >= MIN_RHYME_AGREEMENT
        and scheme_accuracy is not None
        and scheme_accuracy >= MIN_RHYME_AGREEMENT
    )
    return {
        "poem_count": poem_count,
        "exact_scheme_matches": exact,
        "scheme_accuracy": scheme_accuracy,
        "soft_exact_scheme_matches": soft_exact,
        "soft_scheme_accuracy": soft_scheme_accuracy,
        "mean_pairwise_agreement": mean_agreement,
        "gate_met": gate_met,
        "mismatched_poems": [
            {
                "poem_id": poem["poem_id"],
                "expected_scheme": poem["expected_scheme"],
                "observed_scheme": poem["observed_scheme"],
                "pairwise_rhyme_agreement": poem["pairwise_rhyme_agreement"],
            }
            for poem in poem_results
            if not poem["scheme_match"]
        ],
    }


def select_review_lines(
    poem_results: list[dict[str, Any]], *, limit: int = REVIEW_LIMIT
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    def add(poem: dict[str, Any], line: dict[str, Any], category: str) -> None:
        key = (poem["poem_id"], line["line_number"])
        if key in seen or len(selected) >= limit:
            return
        seen.add(key)
        selected.append(
            {
                "category": category,
                "poem_id": poem["poem_id"],
                "author": poem["author"],
                "line_number": line["line_number"],
                "text": line["text"],
                "syllable_count": line["syllable_count"],
                "verse_type": line["verse_type"],
                "observed": line["observed"],
                "uncertain": line["uncertain"],
                "uncertainty_reasons": "|".join(line["uncertainty_reasons"]),
            }
        )

    for poem in poem_results:
        for line in poem["metre"]:
            if line["observed"] is not True:
                add(poem, line, "metre_flagged")
    for poem in poem_results:
        if poem["scheme_match"]:
            continue
        for line in poem["metre"]:
            add(poem, line, "scheme_mismatch")
    for poem in poem_results:
        for line in poem["metre"]:
            add(poem, line, "sample")
    return selected


def corpus_statistics(
    manifest_path: Path, *, root: Path
) -> dict[str, Any]:
    poems = 0
    lines_scored = 0
    correct = 0
    incorrect = 0
    uncertain = 0
    all_valid_poems = 0
    verse_types: Counter[str] = Counter()
    uncertainty_reasons: Counter[str] = Counter()
    failure_examples: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        text = (root / row["clean_text_path"]).read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 14:
            continue
        poems += 1
        analysis = analyse_sonnet("\n".join(line.replace("\u017f", "s") for line in lines))
        poem_valid = True
        for line_result in analysis["lines"]:
            lines_scored += 1
            observed = line_result["is_hendecasyllable"]
            if observed is True:
                correct += 1
                if line_result["verse_type"]:
                    verse_types[line_result["verse_type"]] += 1
            elif observed is False:
                incorrect += 1
                poem_valid = False
                if len(failure_examples) < 10:
                    failure_examples.append(
                        {
                            "poem_id": row["poem_id"],
                            "text": line_result["text"],
                            "syllable_count": line_result["syllable_count"],
                            "verse_type": line_result["verse_type"],
                        }
                    )
            else:
                uncertain += 1
                poem_valid = False
                for reason in line_result["uncertainty_reasons"]:
                    uncertainty_reasons[reason] += 1
        if poem_valid:
            all_valid_poems += 1
    return {
        "poem_count": poems,
        "line_count": lines_scored,
        "correct": correct,
        "incorrect": incorrect,
        "uncertain": uncertain,
        "all_valid_poems": all_valid_poems,
        "verse_types": dict(verse_types),
        "uncertainty_reasons": dict(uncertainty_reasons),
        "failure_examples": failure_examples,
    }


def format_report(
    ground_truth: Mapping[str, Any],
    metre: Mapping[str, Any],
    rhyme: Mapping[str, Any],
    corpus: Mapping[str, Any],
    review_packet_path: Path,
) -> str:
    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.2f}%"

    lines = [
        "# Sonnet Prosody Checker Validation v1",
        "",
        "Date: 2026-09-13",
        "",
        "This report validates the rule-based `sonnet_prosody` checker against the",
        f"frozen ground truth of {ground_truth['poem_count']} sonnets "
        f"({ground_truth['line_count']} lines).",
        "The checker stays provisional until the owner reviews the packet.",
        "",
        "## Ground truth",
        "",
        f"- Selection rule: {ground_truth['selection_rule']}",
        f"- Annotation method: {ground_truth['annotation_method']}",
        "",
        "## Metre",
        "",
        f"- Lines scored: {metre['line_count']}",
        f"- Correct: {metre['correct']}",
        f"- Incorrect: {metre['incorrect']}",
        f"- Uncertain: {metre['uncertain']}",
        f"- Accuracy among definite lines: {percent(metre['accuracy'])}",
        f"- Definite coverage: {percent(metre['coverage'])}",
        f"- Coverage warning (below 90%): "
        f"{'YES' if metre['coverage_warning'] else 'NO'}",
        f"- Gate (accuracy >= 95% on definite lines): "
        f"{'PASS' if metre['gate_met'] else 'FAIL'}",
        "",
        "Ambiguous lines are excluded from the definite accuracy and reported",
        "separately in the uncertainty reasons and the review packet.",
        "",
        "## Rhyme",
        "",
        f"- Poems: {rhyme['poem_count']}",
        f"- Exact scheme matches: {rhyme['exact_scheme_matches']} "
        f"({percent(rhyme['scheme_accuracy'])})",
        f"- Soft (Sicilian) scheme matches: {rhyme['soft_exact_scheme_matches']} "
        f"({percent(rhyme['soft_scheme_accuracy'])})",
        f"- Mean pairwise agreement: {percent(rhyme['mean_pairwise_agreement'])}",
        f"- Gate (both >= 90%): {'PASS' if rhyme['gate_met'] else 'FAIL'}",
        "",
    ]
    if rhyme["mismatched_poems"]:
        lines.append("### Mismatched poems")
        lines.append("")
        for poem in rhyme["mismatched_poems"]:
            lines.append(
                f"- {poem['poem_id']}: expected {poem['expected_scheme']}, "
                f"observed {poem['observed_scheme']} "
                f"(pairwise {percent(poem['pairwise_rhyme_agreement'])})"
            )
        lines.append("")
    lines.extend(
        [
            "## Full-corpus statistics (V6)",
            "",
            f"- Poems: {corpus['poem_count']}",
            f"- Lines: {corpus['line_count']}",
            f"- Correct: {corpus['correct']}",
            f"- Incorrect: {corpus['incorrect']}",
            f"- Uncertain: {corpus['uncertain']}",
            f"- Poems with all 14 lines valid: {corpus['all_valid_poems']}",
            f"- Verse types: {corpus['verse_types']}",
            f"- Uncertainty reasons: {corpus['uncertainty_reasons']}",
            "",
            "## Manual review",
            "",
            f"- Packet: `{review_packet_path}`",
            f"- Selection rule: all metre-flagged lines, then scheme-mismatch lines,",
            f"  then a deterministic sample, up to {REVIEW_LIMIT} lines.",
            "- Status: pending owner review. The checker must not be used as a",
            "  metric, selector, label source, or reward before the review.",
            "",
            "## Verification",
            "",
            "- Rebuild: `python3 scripts/build_sonnet_prosody_ground_truth.py`",
            "- Validate: `python3 scripts/validate_sonnet_prosody.py`",
            "",
        ]
    )
    return "\n".join(lines)


def write_review_packet(path: Path, review_lines: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "poem_id",
        "author",
        "line_number",
        "text",
        "syllable_count",
        "verse_type",
        "observed",
        "uncertain",
        "uncertainty_reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(review_lines)
