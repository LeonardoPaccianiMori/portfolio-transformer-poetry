#!/usr/bin/env python3
"""Write automatic, memorization, and qualitative reports for task-format output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.memorization import write_memorization_report
from sonnet_evaluation.metrics import write_generation_metrics_report
from sonnet_evaluation.qualitative import write_qualitative_review_report
from sonnet_evaluation.task_acceptance import write_task_format_acceptance_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ROOT / "data" / "metadata" / "sonnets_expanded_v5_manifest.csv",
    )
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument("--training-split", default="train")
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--memorization-output", type=Path, required=True)
    parser.add_argument("--qualitative-output", type=Path, required=True)
    parser.add_argument("--acceptance-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("task-evaluation | writing generation metrics", flush=True)
    metric_rows = write_generation_metrics_report(
        generation_dir=args.generation_dir,
        output_path=args.metrics_output,
    )
    print(
        f"task-evaluation | metrics complete outputs={len(metric_rows)}",
        flush=True,
    )

    print("task-evaluation | checking memorization against training poems", flush=True)
    memorization_rows = write_memorization_report(
        generation_dir=args.generation_dir,
        manifest_path=args.manifest_path,
        repo_root=ROOT,
        dataset=args.dataset,
        split=args.training_split,
        output_path=args.memorization_output,
    )
    print(
        f"task-evaluation | memorization complete outputs={len(memorization_rows)}",
        flush=True,
    )

    print("task-evaluation | writing automatic acceptance controls", flush=True)
    acceptance_rows = write_task_format_acceptance_report(
        generation_dir=args.generation_dir,
        output_path=args.acceptance_output,
    )
    print(
        f"task-evaluation | acceptance controls complete outputs={len(acceptance_rows)}",
        flush=True,
    )

    print("task-evaluation | writing qualitative review template", flush=True)
    review_rows = write_qualitative_review_report(
        generation_dir=args.generation_dir,
        output_path=args.qualitative_output,
        review_context="task_format_sonnet",
    )
    print(
        "task-evaluation | complete "
        f"outputs={len(review_rows)} acceptance_report={args.acceptance_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
