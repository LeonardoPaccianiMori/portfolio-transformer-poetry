from types import SimpleNamespace

import pytest
import torch

from sonnet_evaluation.minerva_generation import (
    MINERVA_BASE_VARIANT,
    _validate_adapter_checkpoint,
    generate_minerva_continuation,
    generate_minerva_variant_for_prompts,
)


class CharacterTensorTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    all_special_ids = [0]
    all_special_tokens = ["<eos>"]

    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert add_special_tokens is False
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[ord(char) for char in text]])}

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id) for token_id in token_ids if token_id != 0)


class ScriptedHuggingFaceModel:
    def __init__(self, token_ids):
        self.token_ids = iter(token_ids)
        self.training = True

    def eval(self):
        self.training = False
        return self

    def __call__(self, *, input_ids, attention_mask, past_key_values, use_cache):
        assert input_ids.ndim == 2
        assert attention_mask.ndim == 2
        assert use_cache is True
        next_token_id = next(self.token_ids)
        vocabulary_size = max(256, next_token_id + 1)
        logits = torch.full((1, 1, vocabulary_size), -100.0)
        logits[0, 0, next_token_id] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def test_minerva_generation_preserves_opening_and_stops_after_target_line():
    tokenizer = CharacterTensorTokenizer()
    continuation = "Seconda linea\n"
    model = ScriptedHuggingFaceModel(ord(char) for char in continuation)

    result = generate_minerva_continuation(
        model=model,
        tokenizer=tokenizer,
        opening_line="Prima linea",
        max_new_tokens=100,
        device="cpu",
        seed=1337,
        top_k=50,
        continuation_line_target=1,
    )

    assert result["text"] == "Prima linea\nSeconda linea\n"
    assert result["stop_reason"] == "target_lines"
    assert result["completed_continuation_lines"] == 1
    assert result["generated_new_tokens"] == len(continuation)
    assert model.training is False


def test_minerva_variant_writes_task_format_compatible_metadata(tmp_path):
    tokenizer = CharacterTensorTokenizer()
    model = ScriptedHuggingFaceModel(ord(char) for char in "Seconda\n")
    output_dir = tmp_path / "base"

    metadata = generate_minerva_variant_for_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=[{
            "id": "first",
            "poem_id": "poem",
            "author": "Autore",
            "opening_line": "Prima",
        }],
        output_dir=output_dir,
        model_variant=MINERVA_BASE_VARIANT,
        max_new_tokens=20,
        seeds=[1337],
        device="cpu",
        continuation_line_target=1,
    )

    assert metadata["generation_format"] == "task_format_opening_line_continuation"
    assert metadata["total_line_target"] == 2
    assert metadata["generated_files"][0]["source_prompt_id"] == "first"
    assert (output_dir / "first__seed_1337.txt").read_text(encoding="utf-8") == (
        "Prima\nSeconda\n"
    )


def test_minerva_generation_rejects_non_selected_adapter_checkpoint():
    checkpoint = {
        "checkpoint_type": "minerva_qlora_adapter",
        "model_id": "sapienzanlp/Minerva-3B-base-v1.0",
        "revision": "129ae5366bae3611a1c9f8c68606c38b7de8b055",
        "adapter_state_dict": {"adapter": torch.tensor([1.0])},
        "recipe_config": {
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ("q_proj",),
        },
        "epoch": 6,
        "step": 1116,
        "best_validation_row": {"epoch": 3, "step": 558},
    }

    with pytest.raises(ValueError, match="selected best"):
        _validate_adapter_checkpoint(checkpoint)
