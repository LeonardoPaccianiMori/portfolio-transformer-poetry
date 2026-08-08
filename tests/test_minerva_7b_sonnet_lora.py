from dataclasses import replace

import pytest

from sonnet_corpus.task_format import IGNORE_INDEX, SonnetContinuationExample
from sonnet_training.minerva_7b_sonnet_lora import (
    Minerva7BSonnetLoRAConfig,
    build_sonnet_training_plan,
    build_sonnet_user_message,
    collate_sonnet_examples,
    select_top_qualifying_candidates,
    sonnet_learning_rate,
    tokenize_sonnet_chat_example,
    validate_sonnet_config,
)


class FakeChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        prompt = "<user>" + messages[0]["content"] + "<assistant>"
        rendered = prompt
        if len(messages) == 2:
            rendered += messages[1]["content"] + "<eos>"
        return [ord(character) for character in rendered]


def _example() -> SonnetContinuationExample:
    return SonnetContinuationExample(
        poem_id="poem",
        split="train",
        opening_line="Prima linea",
        continuation_text="\n".join(f"Linea {index}" for index in range(2, 15)),
    )


def test_sonnet_chat_example_masks_prompt_and_supervises_complete_sonnet():
    tokenized = tokenize_sonnet_chat_example(
        example=_example(), tokenizer=FakeChatTokenizer(), context_length=2048
    )

    assert tokenized.labels[:tokenized.response_start] == (
        IGNORE_INDEX,
    ) * tokenized.response_start
    assert tokenized.labels[tokenized.response_start] == ord("P")
    assert "esattamente quattordici versi" in build_sonnet_user_message("Prima linea")


def test_sonnet_collation_masks_padding():
    first = tokenize_sonnet_chat_example(
        example=_example(), tokenizer=FakeChatTokenizer(), context_length=2048
    )
    shorter = replace(
        first,
        poem_id="short",
        input_ids=first.input_ids[:-4],
        labels=first.labels[:-4],
    )

    batch = collate_sonnet_examples(examples=[shorter, first], pad_token_id=0)

    assert batch.attention_mask[0, -1].item() == 0
    assert batch.labels[0, -1].item() == IGNORE_INDEX
    assert batch.target_count > 0


def test_stage_b_plan_and_learning_rate_are_locked_to_ten_epochs():
    config = Minerva7BSonnetLoRAConfig()
    plan = build_sonnet_training_plan(
        config=config, train_count=1481, validation_count=190
    )

    assert plan.updates_per_epoch == 186
    assert plan.planned_updates == 1860
    assert plan.warmup_steps == 93
    assert sonnet_learning_rate(config=config, plan=plan, step=93) == config.learning_rate
    assert sonnet_learning_rate(config=config, plan=plan, step=1860) == pytest.approx(
        config.min_learning_rate
    )


def test_top_candidates_use_raw_validation_loss_and_require_preservation():
    rows = [
        {"epoch": 1, "validation_loss": 2.0, "preservation_gate_passed": True},
        {"epoch": 2, "validation_loss": 1.9, "preservation_gate_passed": False},
        {"epoch": 3, "validation_loss": 1.95, "preservation_gate_passed": True},
        {"epoch": 4, "validation_loss": 1.97, "preservation_gate_passed": True},
        {"epoch": 5, "validation_loss": 1.96, "preservation_gate_passed": True},
    ]

    selected = select_top_qualifying_candidates(rows)

    assert [row["epoch"] for row in selected] == [3, 5, 4]


def test_stage_b_rejects_recipe_drift():
    with pytest.raises(ValueError, match="locked"):
        validate_sonnet_config(
            replace(Minerva7BSonnetLoRAConfig(), learning_rate=2e-5)
        )
