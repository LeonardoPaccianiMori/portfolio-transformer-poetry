import pytest
import torch
import torch.nn.functional as F

import sonnet_training.task_format_steps as task_format_steps
from sonnet_corpus.task_format import (
    IGNORE_INDEX,
    EncodedSonnetContinuationExample,
    SonnetContinuationExample,
)
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.task_format_steps import (
    SonnetContinuationBatch,
    collate_sonnet_continuation_examples,
    estimate_sonnet_continuation_loss,
    sample_sonnet_continuation_batch,
    train_sonnet_continuation_step,
)


def make_encoded_example(
    poem_id: str,
    input_values: list[int],
    target_values: list[int],
) -> EncodedSonnetContinuationExample:
    return EncodedSonnetContinuationExample(
        example=SonnetContinuationExample(
            poem_id=poem_id,
            split="train",
            opening_line="opening",
            continuation_text="continuation",
        ),
        input_ids=torch.tensor(input_values, dtype=torch.long),
        target_ids=torch.tensor(target_values, dtype=torch.long),
        continuation_target_start=1,
    )


def build_tiny_transformer(vocab_size: int = 8) -> CausalTransformerLanguageModel:
    return CausalTransformerLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        max_context_length=8,
    )


def test_collate_right_pads_inputs_and_ignores_padding_labels():
    short = make_encoded_example("short", [1, 2], [IGNORE_INDEX, 3])
    long = make_encoded_example("long", [4, 5, 6, 7], [IGNORE_INDEX, 1, 2, 3])

    batch = collate_sonnet_continuation_examples(
        examples=[short, long],
        pad_token_id=0,
        max_context_length=8,
    )

    assert torch.equal(
        batch.input_ids,
        torch.tensor([[1, 2, 0, 0], [4, 5, 6, 7]], dtype=torch.long),
    )
    assert torch.equal(
        batch.target_ids,
        torch.tensor(
            [[IGNORE_INDEX, 3, IGNORE_INDEX, IGNORE_INDEX], [IGNORE_INDEX, 1, 2, 3]],
            dtype=torch.long,
        ),
    )
    assert batch.supervised_target_count == 4


def test_collate_rejects_examples_that_exceed_model_context():
    example = make_encoded_example(
        "long",
        [1, 2, 3],
        [IGNORE_INDEX, 2, 3],
    )

    with pytest.raises(ValueError, match="max_context_length"):
        collate_sonnet_continuation_examples(
            examples=[example],
            pad_token_id=0,
            max_context_length=2,
        )


def test_sample_uniformly_selects_document_indices(monkeypatch: pytest.MonkeyPatch):
    examples = [
        make_encoded_example("first", [1, 2], [IGNORE_INDEX, 3]),
        make_encoded_example("second", [4, 5], [IGNORE_INDEX, 6]),
        make_encoded_example("third", [2, 3], [IGNORE_INDEX, 4]),
    ]

    def fixed_randint(**kwargs):
        assert kwargs["low"] == 0
        assert kwargs["high"] == 3
        assert kwargs["size"] == (2,)
        return torch.tensor([2, 0])

    monkeypatch.setattr(task_format_steps.torch, "randint", fixed_randint)
    batch = sample_sonnet_continuation_batch(
        examples=examples,
        batch_size=2,
        pad_token_id=0,
        max_context_length=8,
    )

    assert torch.equal(batch.input_ids, torch.tensor([[2, 3], [1, 2]]))


def test_train_step_weights_gradient_accumulation_by_supervised_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    class WeightedLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, target_ids):
            del input_ids
            values = target_ids[target_ids != IGNORE_INDEX].float()
            return torch.empty(0), self.weight * values.mean()

    first_batch = SonnetContinuationBatch(
        input_ids=torch.tensor([[1]], dtype=torch.long),
        target_ids=torch.tensor([[1]], dtype=torch.long),
        supervised_target_count=1,
    )
    second_batch = SonnetContinuationBatch(
        input_ids=torch.tensor([[2, 2, 2]], dtype=torch.long),
        target_ids=torch.tensor([[3, 3, 3]], dtype=torch.long),
        supervised_target_count=3,
    )
    batches = iter([first_batch, second_batch])
    model = WeightedLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    monkeypatch.setattr(
        task_format_steps,
        "sample_sonnet_continuation_batch",
        lambda **kwargs: next(batches),
    )

    loss = train_sonnet_continuation_step(
        model=model,
        optimizer=optimizer,
        examples=[],
        batch_size=1,
        pad_token_id=0,
        max_context_length=8,
        device=torch.device("cpu"),
        gradient_accumulation_steps=2,
    )

    assert loss == pytest.approx(2.5)
    assert model.weight.item() == pytest.approx(0.75)


def test_validation_visits_every_example_and_weights_loss_by_target_count():
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_first_tokens: list[int] = []

        def forward(self, input_ids, target_ids):
            self.seen_first_tokens.extend(input_ids[:, 0].tolist())
            values = target_ids[target_ids != IGNORE_INDEX].float()
            return torch.empty(0), values.mean()

    examples = [
        make_encoded_example("first", [1], [2]),
        make_encoded_example("second", [3, 3, 3], [4, 4, 4]),
        make_encoded_example("third", [5, 5], [6, 6]),
    ]
    model = RecordingModel()

    loss = estimate_sonnet_continuation_loss(
        model=model,
        examples=examples,
        batch_size=2,
        pad_token_id=0,
        max_context_length=8,
        device=torch.device("cpu"),
    )

    assert loss == pytest.approx((2 + 4 * 3 + 6 * 2) / 6)
    assert model.seen_first_tokens == [1, 3, 5]
    assert model.training


def test_transformer_cross_entropy_ignores_masked_targets_explicitly():
    torch.manual_seed(0)
    model = build_tiny_transformer()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    target_ids = torch.tensor([[IGNORE_INDEX, 2, IGNORE_INDEX, 4]], dtype=torch.long)

    logits, loss = model(input_ids, target_ids)

    expected_loss = F.cross_entropy(
        logits[0, [1, 3]],
        torch.tensor([2, 4]),
        ignore_index=IGNORE_INDEX,
    )
    assert loss is not None
    assert torch.allclose(loss, expected_loss)
