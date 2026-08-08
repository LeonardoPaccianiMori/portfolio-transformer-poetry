from dataclasses import replace

import pytest
import torch

from sonnet_corpus.task_format import SonnetContinuationExample
from sonnet_training.minerva_qlora_finetuning import IGNORE_INDEX
from sonnet_training.minerva_qlora_finetuning import MinervaQLoRAFineTuningConfig
from sonnet_training.minerva_qlora_finetuning import build_minerva_continuation_prompt
from sonnet_training.minerva_qlora_finetuning import build_training_plan
from sonnet_training.minerva_qlora_finetuning import collate_minerva_continuation_examples
from sonnet_training.minerva_qlora_finetuning import learning_rate_for_qlora_step
from sonnet_training.minerva_qlora_finetuning import _load_resume_checkpoint
from sonnet_training.minerva_qlora_finetuning import _resume_recipe_config
from sonnet_training.minerva_qlora_finetuning import _save_adapter_checkpoint
from sonnet_training.minerva_qlora_finetuning import tokenize_minerva_continuation_example
from sonnet_training.minerva_qlora_finetuning import validate_finetuning_config


class CharacterTokenizer:
    eos_token_id = 0

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def _example(*, poem_id="example", opening="Prima riga", continuation="Seconda\nTerza"):
    return SonnetContinuationExample(
        poem_id=poem_id,
        split="train",
        opening_line=opening,
        continuation_text=continuation,
    )


def test_minerva_prompt_is_the_visible_opening_line_and_newline():
    assert build_minerva_continuation_prompt("Prima riga") == "Prima riga\n"

    with pytest.raises(ValueError, match="exactly one line"):
        build_minerva_continuation_prompt("Prima\nriga")


def test_minerva_tokenization_masks_the_full_prompt_and_keeps_continuation():
    tokenized = tokenize_minerva_continuation_example(
        example=_example(),
        tokenizer=CharacterTokenizer(),
        context_length=512,
    )
    prompt_length = len("Prima riga\n")

    assert tokenized.input_ids[-1] == 0
    assert tokenized.labels[:prompt_length] == (IGNORE_INDEX,) * prompt_length
    assert tokenized.labels[prompt_length] == ord("S")
    assert tokenized.continuation_target_start == prompt_length


def test_minerva_collation_right_pads_and_preserves_masked_labels():
    tokenizer = CharacterTokenizer()
    first = tokenize_minerva_continuation_example(
        example=_example(opening="Uno", continuation="Due"),
        tokenizer=tokenizer,
        context_length=512,
    )
    second = tokenize_minerva_continuation_example(
        example=_example(opening="Uno", continuation="Due\nTre"),
        tokenizer=tokenizer,
        context_length=512,
    )

    batch = collate_minerva_continuation_examples(
        examples=[first, second],
        pad_token_id=0,
    )

    assert batch.input_ids.shape == (2, len(second.input_ids))
    assert batch.attention_mask[0, -1].item() == 0
    assert batch.labels[0, -1].item() == IGNORE_INDEX
    assert batch.supervised_target_count == sum(
        label != IGNORE_INDEX
        for example in (first, second)
        for label in example.labels
    )
    assert batch.input_ids.dtype == torch.long


def test_fixed_minerva_training_plan_uses_full_document_passes():
    config = MinervaQLoRAFineTuningConfig()

    plan = build_training_plan(config=config, train_example_count=1486)

    assert plan.updates_per_epoch == 186
    assert plan.planned_updates == 3720
    assert plan.warmup_steps == 186
    assert learning_rate_for_qlora_step(config=config, plan=plan, step=1) == pytest.approx(
        1e-4 / 186
    )
    assert learning_rate_for_qlora_step(
        config=config,
        plan=plan,
        step=plan.planned_updates,
    ) == pytest.approx(1e-5)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("context_length", 1024),
        ("batch_size", 2),
        ("gradient_accumulation_steps", 4),
        ("max_epochs", 10),
        ("lora_rank", 32),
    ],
)
def test_minerva_finetuning_rejects_recipe_drift(field_name, value):
    config = replace(MinervaQLoRAFineTuningConfig(), **{field_name: value})

    with pytest.raises(ValueError, match="locked"):
        validate_finetuning_config(config)


def test_resume_recipe_identity_excludes_the_local_checkpoint_path():
    initial = MinervaQLoRAFineTuningConfig()
    resumed = replace(
        initial,
        resume_from_checkpoint="runs/minerva_qlora/resume.pt",
    )

    assert _resume_recipe_config(initial) == _resume_recipe_config(resumed)


def test_resume_checkpoint_restores_adapter_optimizer_and_shuffle_state(tmp_path):
    class AdapterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.tensor([1.0]))

    def get_adapter_state(model):
        return {"adapter": model.adapter}

    def set_adapter_state(model, state):
        model.adapter.data.copy_(state["adapter"])

    model = AdapterModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.adapter.sum().backward()
    optimizer.step()
    config = MinervaQLoRAFineTuningConfig()
    plan = build_training_plan(config=config, train_example_count=1486)
    generator_state = torch.Generator().manual_seed(1337).get_state()
    checkpoint_path = tmp_path / "resume.pt"
    dependencies = {
        "get_peft_model_state_dict": get_adapter_state,
        "set_peft_model_state_dict": set_adapter_state,
    }

    _save_adapter_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        dependencies=dependencies,
        config=config,
        manifest_sha256="manifest",
        epoch=2,
        step=372,
        best_validation_row={"epoch": 2, "step": 372, "validation_loss": 1.5},
        optimizer=optimizer,
        history=[{"epoch": 1}, {"epoch": 2}],
        non_improving_evaluations=1,
        generator_state=generator_state,
        include_optimizer_state=True,
    )
    expected_adapter = model.adapter.detach().clone()
    model.adapter.data.zero_()

    restored = _load_resume_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        dependencies=dependencies,
        config=replace(config, resume_from_checkpoint=str(checkpoint_path)),
        manifest_sha256="manifest",
        plan=plan,
    )

    epoch, step, history, best_row, non_improving_evaluations, restored_generator = restored

    assert epoch == 2
    assert step == 372
    assert history == [{"epoch": 1}, {"epoch": 2}]
    assert best_row == {"epoch": 2, "step": 372, "validation_loss": 1.5}
    assert non_improving_evaluations == 1
    assert torch.equal(restored_generator, generator_state)
    assert torch.equal(model.adapter.detach(), expected_adapter)
