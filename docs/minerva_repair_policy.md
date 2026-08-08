# Minerva Repair Policy

Decision approved: 2026-08-08.

## Purpose

The completed Minerva 3B experiment rejected one fixed QLoRA recipe, not the
entire pretrained-model branch. That recipe learned reliable line production
and topic persistence but damaged grammatical composition. The project will
therefore run one bounded repair programme before returning to Minerva-guided
DPO and GRPO for the from-scratch model.

The untouched-Minerva judge gate and both dependent post-training branches are
paused. They are not cancelled, and none of their final-test isolation rules
change.

## Model Roles

### Minerva 7B Instruct

Use `sapienzanlp/Minerva-7B-instruct-v1.0` at revision
`d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d` as the strongest Minerva-family
quality baseline. It is an Apache-2.0, 7.4-billion-parameter model derived from
Minerva 7B Base through supervised instruction tuning and online DPO.

The earlier project decision excluded this checkpoint to keep the external
comparison limited to base-model adaptation. That restriction is superseded:
output quality is now the purpose of this repair programme, so existing
instruction tuning is relevant evidence rather than an uncontrolled nuisance.

The 7B checkpoint receives exactly:

1. one 4-bit NF4 prompt-only baseline on the frozen eight-poem V5 validation
   set;
2. one conservative local QLoRA memory calibration at context length 512;
3. a later fine-tuning recipe only if the calibration leaves at least 512 MiB
   of measured CUDA headroom after a representative optimizer update.

If calibration fails or leaves less headroom, 7B remains a quantized inference
baseline. The project will not force slow CPU-offloaded 7B training merely to
claim that it ran locally.

### Minerva 3B Base

Retain `sapienzanlp/Minerva-3B-base-v1.0` at revision
`129ae5366bae3611a1c9f8c68606c38b7de8b055` as the practical local adaptation
candidate. Before another run, audit the V5 sonnet texts and design a staged
recipe that may include:

- historical-Italian domain adaptation before sonnet specialization;
- instruction-formatted sonnet supervision;
- clean Italian replay or a base-model preservation penalty;
- a lower learning rate and narrower adapter scope;
- frequent validation-generation checkpoints rather than selection by
  validation loss alone.

Those choices are directions, not silently authorized hyperparameters. The
exact 3B training recipe requires a separate predeclared checkpoint after the
data audit and 7B baseline results.

## Validation Baseline

The 7B Instruct baseline uses the eight prompts already frozen in
`configs/minerva_3b_validation_sanity_prompts.json`. They cover eight authors
and five centuries and do not overlap the fixed final test.

- Chat format: the checkpoint's published tokenizer chat template.
- User request: exactly fourteen lines of classical Italian, exact supplied
  opening line, coherent topic, grammatical syntax, no commentary.
- Assistant prefill: the exact opening line and newline.
- Seed: `4242`.
- Temperature: `0.8`.
- Top-k: `50`.
- Ceiling: 512 generated tokens.
- Decoder control: stop after thirteen completed continuation lines.

The decoder-enforced line count is a mechanical control, not evidence of metre
or rhyme. The baseline counts as a credible quality parent if at least 5/8
outputs are generally grammatical, at least 5/8 sustain a topic or argument
for seven generated lines, and no more than 1/8 severely collapses. Results are
reported even if the threshold is missed.

## 7B Training Calibration

The one permitted calibration uses:

- 4-bit NF4 weights with double quantization and float16 computation;
- context length 512 and microbatch size one;
- gradient checkpointing;
- rank-8, alpha-16 LoRA on `q_proj`, `k_proj`, `v_proj`, and `o_proj` only;
- `PagedAdamW8bit` at learning rate `2e-5`;
- one representative instruction/sonnet optimizer update.

The calibration records the exact revision, package versions, trainable
parameters, loss, GPU identity, total memory, peak allocated and reserved CUDA
memory, and measured headroom. An out-of-memory outcome is a valid completed
result rather than a reason to vary context length, rank, or module scope.

## V5 Data Audit

Before designing another 3B run, inspect every selected V5 poem for structural
validity, residual archive or markup artifacts, suspicious line lengths,
duplicate texts, source and author concentration, period balance, and cleaning
metadata. Produce a deterministic representative training-text review sample.

Automated structural checks cannot certify historical Italian grammar. The
audit therefore separates machine-detectable defects from the later editorial
review of syntax and edition quality.

### Completed Audit Result

The audit completed against all 1,875 selected V5 poems. It found no residual
wiki, HTML, URL, or replacement-character markers, but the structural gate is
`review_required`:

- `cavalcanti_la_genealogia_dei_manoscritti` is an editorial apparatus page,
  not a sonnet, despite having fourteen extracted lines;
- six exact normalized duplicate groups remain;
- four duplicate groups cross split boundaries, including two
  train/validation and two train/test groups;
- the largest training-author shares are Vittoria Colonna at 18.0 percent and
  Francesco Petrarca at 17.0 percent.

The machine-readable and public summaries are
`reports/minerva_v5_sft_corpus_audit.json` and
`reports/minerva_v5_sft_corpus_audit.md`. A corrected V6 manifest must remove
the editorial page and assign each exact-text group wholly to one split before
another Minerva training recipe is frozen. This correction requires a separate
approved data-version checkpoint.

## Isolation And Exit Rules

- Use only training and validation records for repair design and checkpoint
  selection.
- Do not expose or regenerate the fixed final-test prompts during calibration.
- Do not use final-test results to tune prompts, adapter strength, or decoding.
- QLoRA is treated as a memory mechanism, not as one immutable training recipe.
- After the 7B baseline/calibration and V5 audit, predeclare the exact 3B recipe
  and, if hardware permits, the exact 7B recipe before either full run.
- Run at most one full repaired 3B experiment and one full 7B experiment.
- Evaluate every surviving model under one shared protocol before deciding
  whether to resume the judge/DPO/GRPO branch.
