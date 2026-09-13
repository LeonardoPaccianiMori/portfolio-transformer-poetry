"""Verifier-labelled form DPO configuration and preference loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sonnet_evaluation.verifier_preferences import (
    PREFERENCE_VERSION,
    load_verifier_preferences,
)
from sonnet_training.minerva_v7_ai_dpo import (
    DPOExample,
    PARENT_IDENTITY,
    PARENT_STATE_ID,
    TARGET_MODULES,
)

EXPERIMENT_VERSION = "minerva_7b_v7_verifier_form_dpo_v1"
SCOPE = "exploratory_verifier_labelled_form_dpo"
CHECKPOINT_PREFIX = "minerva_v7_verifier_dpo"


def load_verifier_dpo_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_version": EXPERIMENT_VERSION,
        "scope": SCOPE,
        "parent_state_id": PARENT_STATE_ID,
        "parent_state_identity_sha256": PARENT_IDENTITY,
        "preference_version": PREFERENCE_VERSION,
        "context_length": 1024,
        "epochs": 1,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5,
        "minimum_learning_rate": 1e-6,
        "warmup_fraction": 0.1,
        "weight_decay": 0.0,
        "maximum_gradient_norm": 1.0,
        "dpo_beta": 0.1,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": list(TARGET_MODULES),
        "split_seed": 11411,
        "validation_fraction": 0.1,
        "training_seed": 11413,
        "progress_interval": 5,
        "checkpoint_interval": 15,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"verifier DPO configuration mismatch: {key}")
    authorization = config.get("authorization", {})
    if (
        authorization.get("verifier_labelled_dpo_authorized") is not True
        or authorization.get("human_calibrated_claim_authorized") is not False
        or authorization.get("validation_calibration_pairs_eligible_for_training")
        is not False
        or authorization.get("v7_test_access_authorized") is not False
        or authorization.get("autonomous_completion_workflow_active") is not True
    ):
        raise PermissionError("verifier DPO authorization contract changed")
    if float(config.get("hourly_rate_usd", 0)) <= 0:
        raise ValueError("verifier DPO must record a positive hourly rate")
    return config


def load_verifier_examples(path: Path) -> list[DPOExample]:
    payload = load_verifier_preferences(path)
    rows = []
    for raw in payload["pairs"]:
        rows.append(
            DPOExample(
                pair_id=str(raw["pair_id"]),
                pair_type=str(raw["pair_type"]),
                prompt_id=str(raw["prompt_id"]),
                opening_line=str(raw["opening_line"]),
                chosen=str(raw["chosen"]),
                rejected=str(raw["rejected"]),
                vote_counts={},
            )
        )
    if not rows or len({row.pair_id for row in rows}) != len(rows):
        raise ValueError("verifier DPO examples are empty or duplicated")
    return rows
