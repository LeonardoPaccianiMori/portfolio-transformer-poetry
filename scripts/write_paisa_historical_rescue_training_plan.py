#!/usr/bin/env python3
"""Write the fixed PAISA-to-historical rescue training-plan reports."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.paisa_historical_rescue import (
    RESCUE_TRAINING_PLAN_JSON_PATH,
)
from sonnet_training.paisa_historical_rescue import (
    RESCUE_TRAINING_PLAN_MARKDOWN_PATH,
)
from sonnet_training.paisa_historical_rescue import write_rescue_training_plan


def main() -> None:
    plan = write_rescue_training_plan(ROOT)
    print(
        "rescue-plan | complete stages={stages} updates={updates:,} "
        "raw_estimate_hours={hours:.1f}".format(
            stages=len(plan.stages),
            updates=sum(stage.train_steps for stage in plan.stages),
            hours=plan.estimated_raw_training_hours,
        ),
        flush=True,
    )
    print(f"rescue-plan | wrote JSON: {RESCUE_TRAINING_PLAN_JSON_PATH}", flush=True)
    print(
        "rescue-plan | wrote Markdown: "
        f"{RESCUE_TRAINING_PLAN_MARKDOWN_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
