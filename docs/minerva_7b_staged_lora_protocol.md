# Minerva 7B Staged FP16 LoRA Protocol

Decision approved: 2026-08-08.

## Purpose

This experiment tests whether Minerva 7B can first acquire a stronger model of
historical Italian prose and then learn the sonnet-composition task. It starts
from `sapienzanlp/Minerva-7B-instruct-v1.0` at revision
`d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`.

The 7.4-billion-parameter parent is loaded directly in FP16. It is not loaded in
4-bit or 8-bit form and none of its base parameters are updated. Rank-8 LoRA
adapters on `q_proj`, `k_proj`, `v_proj`, and `o_proj` are the only trainable
model parameters. Training uses ordinary `torch.optim.AdamW`; this is therefore
unquantized FP16 LoRA, not QLoRA.

## Stage A: Historical Adaptation

The historical corpus is `pretraining_historical_italian_v2`: 36 licensed or
public-domain prose sources with source-level provenance. Each source is
tokenized independently with the pinned Minerva tokenizer. Its final one
percent remains validation data, and an end-of-text token separates sources.

Each optimizer update contains exactly:

- seven packed 512-token historical microbatches;
- one packed 512-token PAISÀ modern-Italian replay microbatch;
- 4,096 tokens in the nominal complete update.

The deterministic replay sample is taken only from PAISÀ training text. It is a
local artifact governed by PAISÀ's CC BY-NC-SA terms and is not committed. A
separate fixed sample of PAISÀ validation text is used only for preservation
measurement.

The locked Stage A recipe is:

- at most two complete historical passes;
- peak learning rate `2e-5`, 3 percent warmup, cosine decay to `2e-6`;
- AdamW weight decay `0.01` and maximum gradient norm `1.0`;
- validation every 500 updates and at every epoch boundary;
- atomic resume checkpoint every 100 updates;
- progress output every 25 updates;
- early-stopping patience three, but never before one historical pass.

Every candidate is measured on historical validation text, 256 fixed modern
Italian windows, and twelve project-authored Italian instruction/answer pairs.
A checkpoint qualifies only if:

- historical validation improves by at least `0.005` relative to stage zero;
- modern-Italian loss is no more than 5 percent above stage zero;
- instruction-response loss is no more than 10 percent above stage zero.

The selected Stage A checkpoint is the lowest historical validation-loss
candidate satisfying all three conditions. If none qualifies, Stage A is a
completed negative result and Stage B does not start from a damaged adapter.

## Stage B: Sonnet Specialization

Stage B is scheduled only after Stage A produces a qualifying selected adapter.
It continues that exact adapter lineage while starting a fresh optimizer. The
parent remains frozen and unquantized FP16.

Training uses only V6 train poems. Each example uses Minerva's published chat
template: the user requests an exact fourteen-line classical Italian sonnet and
supplies its required first line; the assistant target is the complete original
sonnet. Loss is masked over the system/user prompt and applied only to the
assistant response. V6 validation poems select checkpoints; final-test poems
remain unavailable until model selection is frozen.

The predeclared Stage B limits are context 512, batch one, accumulation eight,
at most ten epochs, peak learning rate `1e-5`, 5 percent warmup, cosine decay to
`1e-6`, and patience-three early stopping. Modern-Italian and instruction
preservation gates remain active. After training, the three strongest
validation-loss candidates are compared on the eight frozen validation prompts
before one model is selected. Validation loss alone does not choose the final
adapter.

Stage A completed on 2026-08-09 after 4,000 updates. Step 4,000 is the selected
parent because its historical validation loss of `3.187692` is the lowest among
all candidates that passed both preservation gates. Its adapter SHA-256 is
`acfad4d442ac8ea7349dcb1bd379c9b41859027ab45daac54c6b6aa35e0bbc63`.
The retained result is `reports/minerva_7b_historical_fp16_lora_result.md`.

Stage B is implemented in `scripts/train_minerva_7b_sonnet_lora.py`. The
executable refuses any Stage A adapter or V6 manifest whose SHA-256 differs from
the selected artifacts above. It loads only V6 train and validation poems;
final-test poems remain unavailable until selection is frozen. Every epoch
writes an adapter candidate and an optimizer-bearing resume checkpoint. The
result retains the three lowest validation-loss candidates that still satisfy
the original Minerva modern-Italian and instruction-loss limits for subsequent
generation review.

Stage B completed on 2026-08-09 after epoch seven through the declared
patience-three rule. Epochs five, four, and three were the three qualifying
lowest-loss candidates. Frozen validation generation selected epoch four: it
had the lowest mean repetition (`0.2208`) and better qualitative stability than
epochs three and five while remaining only `0.0063` above epoch five's
validation loss. The selected validation loss is `3.171254`; the adapter is
pinned by SHA-256
`aff3f2c4d193ce880ec9c7a6df6373f433001662c3ca78d7f915890733cb0df3` in
`configs/minerva_7b_v6_selected_adapter.json`.

The final test was opened only after that selection was hash-frozen. Loss over
all 197 V6 final-test poems is `3.212791`. All 20 generated outputs preserved
the exact opening and decoder-controlled fourteen-line form, all sustained a
topic for at least seven lines, and none had high-risk training overlap. The
strict acceptable-quality gate nevertheless failed: 8/20 outputs were
generally grammatical against a 12/20 requirement, and 5/20 severely collapsed
against a maximum of 2/20. The retained final result is
`reports/minerva_7b_v6_final_evaluation.md`.

## Runtime And Interruption Policy

The exact twelve-update calibration passed on the rented 48 GiB Quadro RTX
8000. It measured 1,145.2 tokens per second, 14,786.0 MiB peak reserved memory,
and 33,405.4 MiB free after the standard AdamW update. Its update-only estimate
for all 6,762 planned updates is 6 hours 43 minutes; the complete command is
expected to take roughly 7–9 hours after validation and checkpoint overhead.
The retained report is
`reports/minerva_7b_historical_fp16_lora_calibration.md`.

The long command was run by the user directly in the VM terminal. It writes
`resume.pt` every 100 updates, so an interruption loses at most the work since
the most recent resume checkpoint. The resume command must use the same output
directory and explicitly name that checkpoint.

All selected adapters, interval candidates, configuration, logs, data hashes,
baseline measurements, and final result files must be copied off the temporary
Vast instance before it is destroyed or recycled.

Both complete run directories and their generated evidence were copied to the
local repository workspace and hash-verified before the Vast instance was
stopped on 2026-08-09.
