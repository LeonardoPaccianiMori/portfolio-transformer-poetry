#!/usr/bin/env python3
"""Select the best available checkpoint from a fine-tuning run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.finetuning_selection import write_finetuning_checkpoint_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.run_dir / "selected_checkpoint.json"
    selection = write_finetuning_checkpoint_selection(
        repo_root=ROOT,
        run_dir=args.run_dir,
        output_path=output_path,
    )
    print(f"wrote selection: {output_path}")
    print(
        "selected checkpoint: "
        f"step {selection['selected_checkpoint_step']} "
        f"for best measured validation step {selection['best_validation_step']}"
    )


if __name__ == "__main__":
    main()
