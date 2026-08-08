import pytest

from sonnet_evaluation.minerva_7b_sonnet_candidates import (
    build_sonnet_candidate_prompt,
    validate_candidate_checkpoint,
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


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<user>{messages[0]['content']}<assistant>"


def _checkpoint(*, gate=True, epoch=3):
    return {
        "checkpoint_type": "minerva_7b_v6_sonnet_lora_adapter",
        "run_version": SONNET_RUN_VERSION,
        "model_id": MINERVA_7B_INSTRUCT_MODEL_ID,
        "revision": MINERVA_7B_INSTRUCT_REVISION,
        "task_format_version": SONNET_TASK_FORMAT_VERSION,
        "selected_stage_a_sha256": SELECTED_STAGE_A_SHA256,
        "manifest_sha256": V6_MANIFEST_SHA256,
        "row": {"epoch": epoch, "preservation_gate_passed": gate},
    }


def test_candidate_prompt_uses_training_instruction_and_prefills_opening():
    prompt = build_sonnet_candidate_prompt(FakeTokenizer(), "Prima linea")

    assert "esattamente quattordici versi" in prompt
    assert prompt.endswith("Prima linea\n")


def test_candidate_checkpoint_requires_matching_lineage_and_gate():
    validate_candidate_checkpoint(_checkpoint(), expected_epoch=3)

    with pytest.raises(ValueError, match="preservation"):
        validate_candidate_checkpoint(_checkpoint(gate=False), expected_epoch=3)
    with pytest.raises(ValueError, match="epoch"):
        validate_candidate_checkpoint(_checkpoint(), expected_epoch=2)
