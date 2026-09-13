#!/usr/bin/env python3
"""Validate the sonnet prosody checker against the frozen ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.sonnet_prosody_validation import (
    corpus_statistics,
    format_report,
    load_ground_truth,
    metre_metrics,
    rhyme_metrics,
    score_ground_truth,
    select_review_lines,
    write_review_packet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=ROOT / "data/metadata/sonnet_prosody_ground_truth_v1.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/metadata/sonnets_expanded_v6_manifest.csv",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=ROOT / "reports/sonnet_prosody_validation_v1.md",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "reports/sonnet_prosody_validation_v1.json",
    )
    parser.add_argument(
        "--review-packet",
        type=Path,
        default=ROOT / "reports/sonnet_prosody_review_packet_v1.csv",
    )
    parser.add_argument("--review-limit", type=int, default=100)
    return parser.parse_args()


def repository_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    ground_truth = load_ground_truth(args.ground_truth)
    poem_results = score_ground_truth(ground_truth)
    metre = metre_metrics(poem_results)
    rhyme = rhyme_metrics(poem_results)
    corpus = corpus_statistics(args.manifest, root=ROOT)
    review_lines = select_review_lines(poem_results, limit=args.review_limit)
    write_review_packet(args.review_packet, review_lines)

    report_md = format_report(
        ground_truth,
        metre,
        rhyme,
        corpus,
        Path(repository_relative(args.review_packet)),
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(report_md, encoding="utf-8")

    report = {
        "schema_version": 1,
        "ground_truth": {
            "path": repository_relative(args.ground_truth),
            "poem_count": ground_truth["poem_count"],
            "line_count": ground_truth["line_count"],
            "selection_rule": ground_truth["selection_rule"],
            "annotation_method": ground_truth["annotation_method"],
        },
        "metre": metre,
        "rhyme": rhyme,
        "corpus": corpus,
        "review_packet": {
            "path": repository_relative(args.review_packet),
            "line_count": len(review_lines),
            "categories": dict(Counter(line["category"] for line in review_lines)),
        },
        "poems": poem_results,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"metre accuracy: {metre['accuracy']}")
    print(f"metre coverage: {metre['coverage']}")
    print(f"metre gate: {'PASS' if metre['gate_met'] else 'FAIL'}")
    print(f"exact scheme matches: {rhyme['exact_scheme_matches']}/{rhyme['poem_count']}")
    print(f"mean pairwise rhyme agreement: {rhyme['mean_pairwise_agreement']}")
    print(f"rhyme gate: {'PASS' if rhyme['gate_met'] else 'FAIL'}")
    print(f"review packet lines: {len(review_lines)}")
    print(f"wrote report: {repository_relative(args.report_md)}")


if __name__ == "__main__":
    main()
