import hashlib
import json

import pytest
import torch

from sonnet_evaluation.minerva_7b_sonnet_final import (
    SELECTION_VERSION,
    validate_frozen_selection,
)
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)
from sonnet_training.minerva_7b_sonnet_lora import (
    SELECTED_STAGE_A_SHA256,
    SONNET_RUN_VERSION,
    SONNET_TASK_FORMAT_VERSION,
    V6_MANIFEST_SHA256,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_path = run_dir / "checkpoints" / "adapter_epoch_03.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({
        "checkpoint_type": "minerva_7b_v6_sonnet_lora_adapter",
        "run_version": SONNET_RUN_VERSION,
        "model_id": MINERVA_7B_INSTRUCT_MODEL_ID,
        "revision": MINERVA_7B_INSTRUCT_REVISION,
        "task_format_version": SONNET_TASK_FORMAT_VERSION,
        "selected_stage_a_sha256": SELECTED_STAGE_A_SHA256,
        "manifest_sha256": V6_MANIFEST_SHA256,
        "row": {"epoch": 3, "preservation_gate_passed": True},
    }, checkpoint_path)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"conditions": [{"epoch": 3}]}))
    selection = {
        "selection_version": SELECTION_VERSION,
        "stage_b_run_version": SONNET_RUN_VERSION,
        "selection_frozen": True,
        "final_test_used": False,
        "candidate_summary_sha256": _sha(summary_path),
        "selected_epoch": 3,
        "selected_checkpoint_sha256": _sha(checkpoint_path),
    }
    return run_dir, summary_path, selection


def test_final_test_requires_hash_pinned_validation_selection(tmp_path):
    run_dir, summary_path, selection = _fixture(tmp_path)

    epoch, checkpoint = validate_frozen_selection(
        selection=selection,
        run_dir=run_dir,
        candidate_summary_path=summary_path,
    )

    assert epoch == 3
    assert checkpoint.name == "adapter_epoch_03.pt"


def test_final_test_rejects_unfrozen_or_changed_selection(tmp_path):
    run_dir, summary_path, selection = _fixture(tmp_path)
    selection["selection_frozen"] = False

    with pytest.raises(ValueError, match="selection_frozen"):
        validate_frozen_selection(
            selection=selection,
            run_dir=run_dir,
            candidate_summary_path=summary_path,
        )
