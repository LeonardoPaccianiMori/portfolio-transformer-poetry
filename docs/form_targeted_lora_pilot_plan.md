# Form-Targeted LoRA Pilot Plan

## Status and scope

Leonardo approved this pilot on 2026-09-13 as the first Phase B experiment.
The A3 retroactive scoring found no fully form-valid Stage-3 or DPO output, so
the approved best-of-N pilot was skipped with recorded evidence and the plan's
decision gate moved to form-targeted measurement.

This plan covers one LoRA supervised fine-tuning arm and its frozen
evaluation. It does not approve a model release, a full-weight retraining, a
checkpoint publication, or any other Phase B item.

## Question

Does one small form-targeted LoRA SFT step on the corrected 4+4+3+3 V8 corpus
increase formal sonnet validity beyond the Stage-3 baseline?

## Data

- Source: `data/processed/sonnets_expanded_v8/derived_sonnets/part-0001.txt`
  with byte ranges from
  `data/processed/sonnets_expanded_v8/sonnets_manifest.csv`.
- Selection: rows with `v8_split == "train"` only. Quarantine, validation,
  and test rows are excluded.
- Size: 16,298 sonnets, 228,172 lines, 8,753,204 characters (about 2.19M
  tokens). The exact encoded token count is recorded at encode time.
- Rendering: the V8 derived 4+4+3+3 view, including the stanza blank lines.
- Tokenizer: the frozen Minerva tokenizer from the verified Stage-3 package.

## Method

- Base: the published Stage-3 BF16 model, loaded from the verified local
  package copy.
- Adapter: PEFT LoRA, rank 16, alpha 32, dropout 0.05, all linear target
  modules, BF16, no quantization.
- Optimization: causal language-model loss, one epoch, AdamW, learning rate
  1e-4 with 3% warmup and cosine decay, gradient clipping 1.0. The batch size
  is set to fit one H100 80 GB and recorded in the run log.
- Stop rule: stop after the planned token budget; stop on non-finite loss or
  a failed checkpoint save.
- The adapter stays local. No merge and no release.

## Frozen evaluation

- Openings: the 120 tracked validation prompts in
  `configs/minerva_7b_v7_exploratory_prompts.json`, SHA-256
  `2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7`,
  `source_split: sonnets_validation`, `v7_test_accessed: false`.
- Seeds: 6200 and 6201.
- Recipe: temperature 0.85, `top_p` 0.95, `no_repeat_ngram_size` 4, at most
  512 new tokens, stop at 13 continuation lines, the frozen Italian
  instruction prompt, and the opening line force-appended. The sealed test
  set is not touched.
- Systems: baseline Stage-3 with the adapter disabled and candidate with the
  adapter enabled, 240 outputs each, paired by opening and seed.
- Scoring: the reviewed `sonnet_prosody` checker with the same metrics as A3
  (full-form valid rate, accepted hendecasyllable lines, failed lines,
  quatrain and tercet scheme rates, rhyme score) plus paired 95% intervals.

## Pre-registered reading

- Signal: the candidate produces full-form valid outputs where the baseline
  has none, or the paired increase in accepted lines excludes zero and
  reaches at least the sealed DPO effect of 0.23 lines.
- Null: no paired difference outside the interval and no valid output. Then
  stop and reframe toward full-weight V8 retraining or plan-then-poem.
- Guard: form only. No literary-quality claim, no selection-for-quality
  claim, and no public wording before a separate review.

## Implementation files

- `configs/form_targeted_lora_pilot.json`
- `src/sonnet_training/form_targeted_data.py`
- `src/sonnet_training/form_targeted_lora.py`
- `src/sonnet_analysis/form_targeted_validation.py`
- `scripts/encode_form_targeted_v8_data.py`
- `scripts/train_form_targeted_lora.py`
- `scripts/generate_form_targeted_validation.py`
- `scripts/score_form_targeted_validation.py`
- `tests/test_form_targeted_data.py` and related test files
- `reports/form_targeted_lora_pilot_v1.md` and `.json`

## Budget

- One H100 80 GB session: encoding, LoRA training, 480 generations, scoring.
  Estimated 1 to 1.5 hours, about $2 to $5.
- The encoded V8 train set is small (int32 tokens, well under 1 GB). The
  Stage-3 package is already on the prepared instance.

## Risks and limits

- One arm can underpower the test by design.
- The 4+4+3+3 rendering is a new training policy; the shift can help or hurt.
- V8 has never been trained on; this is the first run on it.
- The checker's definite coverage limit (87.1% on the ground truth) applies.
- Training-data decisions for V8 inherit the recorded publication-risk
  decision; this plan releases no adapter.

## Status on 2026-09-13

The pilot ran on one H100. Encoding produced 2,901,061 tokens from 16,298
train sonnets. The LoRA arm completed 178 of 178 steps with a final training
loss of 0.0267. On 240 matched validation pairs the candidate lost 4.44
accepted hendecasyllable lines per output (95% CI -4.75 to -4.13), failed
lines rose by 6.35, the 4+4+3+3 stanza pattern fell from 8 occurrences to 0,
and the candidate text is degenerate. No full-form valid output exists in
either system. The pre-registered reading is NULL. This training
configuration is not supported, and the adapter is preserved locally as
evidence. See `reports/form_targeted_lora_pilot_v1.md`.
