# Minerva 7B Full-Weight Calibration Protocol

## Purpose

This checkpoint measures whether one H100 80 GB can safely update every weight
of the pinned Minerva 7B Instruct parent. It is a memory, throughput, and
numerical-stability measurement. It is not a quality experiment and does not
authorize the long mixed-corpus run.

The calibration follows the failed parent-decoding confirmation. That evidence
permits a full-weight measurement because the untouched parent remained
unstable and the earlier LoRA stages damaged grammar while improving form.

## Data Contract

The complete existing PAISA train/validation and historical prose
train/validation splits are retokenized with Minerva's pinned tokenizer. The
encoder streams documents, appends exactly one Minerva EOS token at each real
document boundary, writes resumable memory-mapped int32 shards, and records
source/split identities and hashes. It does not read sonnet or final-test data.

PAISA text, document identities, token shards, calibration windows, and later
checkpoints remain local under the CC BY-NC-SA lineage. Only aggregate metadata
and hashes enter the public repository.

## Locked Probe

- Model: `sapienzanlp/Minerva-7B-instruct-v1.0` at revision
  `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`.
- Hardware: one NVIDIA H100 with at least 75 GiB reported VRAM and native BF16.
- Weights: unquantized BF16, all parameters trainable, no adapter modules.
- Optimizer: bitsandbytes `PagedAdamW8bit`.
- Context and batch: 512 tokens, microbatch one, no accumulation.
- Updates: exactly five in modern/historical/modern/historical/modern order.
- Learning rate: constant `1e-6`; weight decay `0.01`; gradient clipping `1.0`.
- Activation control: non-reentrant gradient checkpointing.
- Validation: one fixed modern and one fixed historical window before and after.

## Pass Gate

The probe passes only when all five losses and pre-clipping gradient norms are
finite, exactly 100% of model parameters are trainable BF16 weights, no adapter
or quantization marker is present, and at least 8 GiB remains both after
optimizer-state allocation and above peak reserved memory.

The script records per-update loss, gradient norm, time, free memory, peak
allocation/reservation, validation losses, throughput, package versions, and
the final pass/reject decision. It deliberately saves no trained model weights.

After a pass, the measured throughput and memory report must be reviewed before
freezing the long-run mixture, token budget, schedule, preservation gates,
checkpoint interval, resume policy, abort rules, and projected Vast.ai cost.
