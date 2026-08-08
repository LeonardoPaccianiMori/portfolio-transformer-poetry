#!/usr/bin/env python3
"""Build the deterministic V6 correction of the V5 sonnet corpus."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.sonnet_v6_correction import build_sonnets_expanded_v6


def main() -> None:
    started_at = time.monotonic()

    def progress(message: str) -> None:
        elapsed = round(time.monotonic() - started_at)
        print(f"sonnet-v6-build | {message} | elapsed={elapsed}s", flush=True)

    print(
        "sonnet-v6-build | start source=sonnets_expanded_v5 "
        "expected_output=1868 expected_removals=7",
        flush=True,
    )
    report = build_sonnets_expanded_v6(
        repo_root=ROOT,
        source_manifest_path=Path(
            "data/metadata/sonnets_expanded_v5_manifest.csv"
        ),
        source_attribution_path=Path(
            "data/metadata/sonnets_expanded_v5_attribution.md"
        ),
        validation_prompt_path=Path(
            "configs/minerva_3b_validation_sanity_prompts.json"
        ),
        final_test_prompt_path=Path("configs/task_format_acceptance_prompts.json"),
        progress=progress,
    )
    print(
        "sonnet-v6-build | complete poems={poems} removals={removals} "
        "splits={splits} manifest_sha256={sha}".format(
            poems=report["output_poem_count"],
            removals=report["removed_poem_count"],
            splits=report["split_counts"],
            sha=report["output_manifest_sha256"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
