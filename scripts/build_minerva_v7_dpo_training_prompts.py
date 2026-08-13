#!/usr/bin/env python3
"""Build the hash-pinned, training-only 512-opening DPO prompt manifest."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_dpo_preferences import (
    build_training_prompt_manifest,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/training_prompts.json"),
    )
    args = parser.parse_args()
    started = time.monotonic()
    print(
        "minerva-v7-dpo | start job=build_training_prompts device=cpu "
        "total_steps=512 progress_interval=final-only",
        flush=True,
    )
    manifest = build_training_prompt_manifest(
        document_index_path=ROOT / "data/local/minerva_7b_v7/encoded/sonnets_train.documents.jsonl",
        reference_manifest_path=ROOT / "artifacts/local/minerva_7b_v7_analysis/memorization_reference/manifest.json",
        validation_prompt_path=ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
    )
    identity = write_json_atomic(ROOT / args.output, manifest)
    print(
        "minerva-v7-dpo | complete prompts=512 source_split=train "
        f"sha256={identity} elapsed={time.monotonic() - started:.1f}s "
        "v7_test_accessed=False training_performed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
