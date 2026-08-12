#!/usr/bin/env python3
"""Run the bounded single-H100 qualification without authorizing training."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    os.environ["V7_QUALIFICATION_CONFIG"] = (
        "configs/minerva_7b_v7_single_h100_qualification.json"
    )
    runpy.run_path(
        ROOT / "scripts/qualify_minerva_7b_v7_dual_a6000.py", run_name="__main__"
    )
