# Minerva Repair Policy

Decision approved: 2026-08-08.

## Purpose

The completed Minerva 3B experiment rejected one fixed QLoRA recipe, not the
entire pretrained-model branch. That recipe learned reliable line production
and topic persistence but damaged grammatical composition. The project will
therefore run one bounded repair programme before returning to Minerva-guided
DPO and GRPO for the from-scratch model.

The untouched-Minerva judge gate and both dependent post-training branches were
paused while this repair programme ran. The repair programme is now complete,
so the judge gate resumes under its original final-test isolation rules.

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

### Completed Baseline Result

The 4-bit validation run completed on the local RTX 3060 Laptop GPU. Peak CUDA
allocation was 4,445.9 MiB and peak reservation was 4,568.0 MiB. Six of eight
outputs reached the decoder-controlled fourteen-line form, while two reached
the 512-token ceiling. The fixed qualitative review found 2/8 generally
grammatical outputs, 7/8 seven-line topic continuations, and 2/8 severe
collapses.

The prompt-only quality-parent gate therefore fails. This does not disqualify
7B Instruct from calibration or conservative adaptation: it remains materially
more coherent and controllable than the tested 3B Base conditions. The full
evidence and interpretation are in
`reports/minerva_7b_instruct_validation_evaluation.md`.

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

### Completed Local Calibration Result

The fixed local calibration failed during the forward/backward stage before the
first optimizer update. Peak allocation was 5,544.2 MiB, peak reservation was
5,636.0 MiB, and only 137.3 MiB was free after exception cleanup. The attempted
102 MiB allocation could not be satisfied. Local 7B QLoRA is therefore rejected
under this protocol. The retained evidence is
`reports/minerva_7b_qlora_local_calibration.md`.

## Remote Unquantized FP16 Extension

The user approved one remote 48 GiB Quadro RTX 8000 experiment after the local
hardware rejection. This extension tests whether quantization affected the 7B
baseline and enables a quality-oriented adapter run without altering all model
weights.

The remote preflight consists of exactly:

1. the same frozen eight validation prompts and decoding protocol, with the
   exact 7B Instruct revision loaded in unquantized FP16;
2. one context-512, microbatch-one FP16 LoRA optimizer update;
3. rank-8, alpha-16 adapters on `q_proj`, `k_proj`, `v_proj`, and `o_proj`;
4. non-reentrant gradient checkpointing and `PagedAdamW8bit` at `2e-5`;
5. a minimum 4,096 MiB measured headroom requirement before full training.

The RTX 8000 does not provide native BF16 arithmetic, so FP16 is the declared
unquantized precision. The untouched FP16 baseline must be evaluated before its
outputs can be compared with NF4. A successful calibration authorizes recipe
design, not immediate training. The corrected V6 corpus and a frozen full-run
protocol remain mandatory.

### Completed Remote Result

The unquantized FP16 baseline and LoRA calibration both completed on the 48 GiB
Quadro RTX 8000. FP16 inference reserved 14,320.0 MiB at peak. All eight outputs
reached the decoder-controlled form and none reached the token ceiling, but the
fixed review found only 1/8 generally grammatical outputs, 8/8 seven-line topic
continuations, and 1/8 severe collapses. The FP16 prompt-only quality-parent
gate therefore fails. Quantization is not an adequate explanation for the
model's grammatical weakness.

The one-update FP16 LoRA calibration passed. It trained 6,815,744 rank-8
attention-adapter parameters, reserved 14,706.0 MiB at peak, and left 33,485.4
MiB free after the optimizer update. The 48 GiB GPU is sufficient for this
adapter protocol; a larger GPU is not required for memory. The public evidence
is in `reports/minerva_7b_instruct_fp16_validation_evaluation.md` and
`reports/minerva_7b_fp16_lora_calibration.md`.

The corrected V6 corpus is now frozen. The next checkpoint is to predeclare a
two-stage FP16 LoRA lineage: historical-Italian prose continued pretraining,
then V6 sonnet instruction fine-tuning. The failed prompt-only gate remains
part of the evidence and must not be rewritten as a successful untouched
baseline.

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

## Completed V6 Correction

V6 removes exactly the one editorial apparatus page and six duplicate or
mislabeled V5 records. It retains the stronger Cino attributions for the four
Dante/Cino duplicate pairs, retains the Guittone records whose identifiers
match their actual first lines, and does not move any retained poem between
splits. The resulting corpus contains 1,868 poems: 1,481 train, 190 validation,
and 197 test. Its manifest SHA-256 is
`994c4c374f42ba26f1c352d7ad7c3adec7ec4671507770bd7c485cb6f977a4fa`.

The V6 automated gate passes with zero structural issues, zero exact duplicate
groups, zero cross-split duplicate groups, and no suspicious markers. All
frozen validation and final-test prompts remain present in their original
splits. A deterministic period-stratified editorial review accepted all 24
sampled poems without finding residual apparatus or cleaning contamination.
The retained evidence is `reports/minerva_v6_sft_corpus_audit.md` and
`reports/minerva_v6_training_text_review_sample.md`.

## Approved Two-Stage Direction

The 7B repair run will test domain-adaptive continued pretraining before sonnet
specialization. Stage A starts from the pinned Minerva 7B Instruct checkpoint
and trains unquantized FP16 LoRA adapters on the historical Italian prose
corpus. Stage B continues the same adapter lineage with instruction-formatted
V6 sonnet examples. This adapts historical vocabulary and syntax before the
narrow form task while leaving the 7.4-billion-parameter parent weights frozen.

The exact learning rates, adapter scope, token budget, replay and preservation
mixture, validation gates, checkpoint cadence, and stopping rules are frozen in
`docs/minerva_7b_staged_lora_protocol.md`. Stage A includes modern-Italian
replay plus modern-language and instruction-following preservation checks
because raw-text adaptation of an instruct model can weaken abilities already
present in the parent.

The exact Stage A calibration passed on the remote Quadro RTX 8000 at 1,145.2
tokens per second, with 14,786.0 MiB peak reserved memory and 33,405.4 MiB free
after a standard AdamW update. The planned 6,762-update command is estimated at
7–9 hours including validation and checkpoint overhead. Evidence is in
`reports/minerva_7b_historical_fp16_lora_calibration.md`.

## Completed Two-Stage Result

Stage A stopped after update 4,000 through patience and selected that checkpoint
at historical validation loss `3.187692` after both preservation gates passed.
Stage B stopped after epoch seven and retained epochs five, four, and three for
frozen validation generation. Epoch four was selected for the best qualitative
stability and lowest mean repetition while remaining near the minimum
validation loss. Its adapter SHA-256 is
`aff3f2c4d193ce880ec9c7a6df6373f433001662c3ca78d7f915890733cb0df3`.

On the unopened V6 final test, epoch four reached loss `3.212791` over 197
poems. All 20 generated outputs preserved the requested opening, reached the
decoder-controlled fourteen-line form, sustained a topic for at least seven
lines, and avoided high-risk training overlap. It did not pass the complete
acceptable-quality gate: 8/20 outputs were generally grammatical and 5/20
severely collapsed. The bounded repair programme therefore ends with a mixed
negative result rather than another tuning round. Full evidence is in
`reports/minerva_7b_v6_final_evaluation.md`; the next checkpoint is the
predeclared untouched-Minerva judge gate.

## Isolation And Exit Rules

- Use only training and validation records for repair design and checkpoint
  selection.
- Do not expose or regenerate the fixed final-test prompts during calibration.
- Do not use final-test results to tune prompts, adapter strength, or decoding.
- QLoRA is treated as a memory mechanism, not as one immutable training recipe.
- Predeclare the exact two-stage 7B recipe, including preservation controls,
  before either historical adaptation or sonnet fine-tuning begins.
- Run at most one full repaired 3B experiment and one full 7B experiment.
- Evaluate every surviving model under one shared protocol before deciding
  whether to resume the judge/DPO/GRPO branch.
