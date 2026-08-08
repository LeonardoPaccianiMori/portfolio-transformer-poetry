# Minerva 3B Validation Sanity Audit

Decision approved: 2026-08-08.

## Purpose

The first fixed Minerva comparison found a large QLoRA task-adaptation effect,
but the selected epoch-3 adapter still failed the predeclared final-test quality
gate. Before using untouched Minerva as a DPO or GRPO judge, this bounded audit
tests three plausible explanations:

- the bare opening-line prompt understates the untouched Base model's ability;
- the full-strength epoch-3 adapter overwhelms useful Base behavior;
- the epoch-6 overfitting contrast is qualitatively worse than epoch 3.

The original final-test result remains valid. This audit uses validation poems
only and cannot redefine or erase that failure.

## Frozen Validation Set

[`../configs/minerva_3b_validation_sanity_prompts.json`](../configs/minerva_3b_validation_sanity_prompts.json)
contains eight V5 validation poems from eight distinct authors. It covers the
thirteenth, fourteenth, sixteenth, seventeenth, and eighteenth centuries. The
runner validates every opening against the V5 manifest and rejects overlap with
the fixed final-test prompt file.

The decoding configuration is fixed before generation:

- seed: `4242`;
- temperature: `0.8`;
- top-k: `50`;
- generated-token ceiling: `512`;
- target: thirteen continuation lines after the exact opening line;
- total: 56 outputs, eight for each of seven conditions.

## Conditions

| Condition | Checkpoint | Prompt | Adapter scale | Selection role |
| --- | --- | --- | ---: | --- |
| `base_raw` | untouched Base | opening line only | 0.00 | diagnostic |
| `base_instructed` | untouched Base | explicit Italian sonnet instruction | 0.00 | diagnostic |
| `best_scale_025` | selected epoch 3 | opening line only | 0.25 | eligible |
| `best_scale_050` | selected epoch 3 | opening line only | 0.50 | eligible |
| `best_scale_075` | selected epoch 3 | opening line only | 0.75 | eligible |
| `best_scale_100` | selected epoch 3 | opening line only | 1.00 | eligible |
| `final_scale_100` | overfitted epoch 6 | opening line only | 1.00 | diagnostic |

The instruction used for `base_instructed` requests exactly fourteen lines,
coherent subject matter, grammatical syntax, and no repetition. It ends with
the exact opening line. The instruction itself is hidden from the visible
output. The Base conditions cannot be selected as QLoRA configurations, and
the epoch-6 checkpoint cannot be selected because its validation loss is worse.

Epochs 1 and 2 were not retained by the completed training run. Reconstructing
them would require a new training run, so they are outside this bounded audit.

## Blinded Review And Selection

The CPU evaluator computes form, length, token-ceiling, and repetition
diagnostics. It separately randomizes outputs behind stable hashed identifiers
and creates a review without condition names. Each output receives three
binary human judgments:

- generally grammatical Italian;
- one topic or argument sustained for at least seven generated lines;
- severe repetition or generation collapse.

An eligible epoch-3 scale qualifies for one final-test rerun only if it has:

- at least 7/8 exact-opening controlled 14-line outputs;
- at least 5/8 generally grammatical outputs;
- at least 5/8 seven-line topic continuations;
- no more than 1/8 severe-collapse outputs.

Qualifying conditions are ranked by grammatical count, then fewer collapses,
then topic count, then lower adapter scale. Automatic repetition cannot replace
human judgment. The condition mapping remains unopened until every blinded
judgment is fixed.

If no epoch-3 scale qualifies, the QLoRA branch remains failed and no new final
test is run. If one qualifies, it receives exactly one separately reported
rerun on the existing final-test protocol. The original scale-1.0 result stays
in the report as the primary fixed-recipe result.

## Execution And Interruption

[`../scripts/generate_minerva_sanity_audit.py`](../scripts/generate_minerva_sanity_audit.py)
loads Minerva once and reports progress, elapsed time, and ETA after every
output. A rerun reuses each condition whose complete metadata and output files
match the protocol, limiting interruption loss to the current condition.

After GPU generation, the assistant runs
[`../scripts/evaluate_minerva_sanity_audit.py`](../scripts/evaluate_minerva_sanity_audit.py)
on CPU to create the automatic evidence and blinded review. DPO and GRPO judge
development was paused until this audit was resolved.

## Completed Result

The audit completed on 2026-08-08. None of the selectable epoch-3 adapter
strengths qualified: their grammatical counts were 0/8, 0/8, 0/8, and 1/8 for
scales 0.25, 0.50, 0.75, and 1.00. Raw Base retained grammatical modern prose
in 5/8 outputs but did not reliably perform the historical-sonnet task; the
explicit instruction reduced that count to 3/8 and increased collapse.

No additional final-test rerun is authorized. The complete result is recorded
in
[`../reports/minerva_3b_validation_sanity_evaluation.md`](../reports/minerva_3b_validation_sanity_evaluation.md).
The Minerva judge-validation gate is now the current checkpoint again.
