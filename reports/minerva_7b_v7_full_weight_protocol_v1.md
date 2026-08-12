# Minerva 7B V7 Full-Weight Training Protocol

Checkpoint 8E freezes the scientific training, validation, preservation,
checkpoint/resume, abort, hardware-qualification, and cost contracts. It
does not authorize GPU rental, benchmarking, or the long run.

## Frozen stages

| Stage | Windows | Target tokens | Updates | Peak LR | Minimum LR |
| --- | ---: | ---: | ---: | ---: | ---: |
| stage_1_historical_general | 33,040 | 67,665,920 | 2,065 | 1.0e-05 | 1.0e-06 |
| stage_2_non_sonnet_poetry | 12,160 | 24,903,680 | 760 | 5.0e-06 | 5.0e-07 |
| stage_3_sonnets | 2,160 | 4,423,680 | 135 | 1.0e-06 | 1.0e-07 |

Every update consumes 16 complete context-2,048 windows / 32,768 target tokens. Optimizer and scheduler state reset at each stage; model weights do not.

## Promotion and preservation

A stage promotes only its lowest primary validation-loss checkpoint that
also passes its minimum-improvement rule, any earlier-domain retention
gate, the held-out PAISÀ modern-loss gate (maximum 1.05x the untouched
parent), and the 12-prompt instruction-loss gate (maximum 1.10x). Stage 3
uses V7 validation for selection; the 106-window V7 test remains unopened
until the final stage-3 checkpoint has been selected.

The modern gate uses 128 deterministic held-out PAISÀ windows. Its local
index hash is `01299b685515fffdd5d3a1ec00204d2038d1639365efb8a2e0e2575e0e3fc582`; the index and token shard are not public repository data.

## Checkpoints and later change analysis

Resume checkpoints are atomic and include model, optimizer, scheduler, RNG,
sampler position, counters, histories, hashes, software, and topology. Two
resume generations are retained, and a fresh-process exact-resume proof is
required before the long run.

For a later study of what each adaptation stage changed, retain model-only
BF16 snapshots at each stage midpoint and validation-selected endpoint. The
untouched parent is referenced by its pinned published revision. Six new
snapshots project to about 88.8 GB before filesystem/compression overhead.

Activation changes will be measured post hoc in evaluation mode on a frozen
held-out probe suite. Exact token IDs, positions, tokenizer/model hashes, and
extraction settings must be frozen before training. The planned comparisons
include layerwise CKA, cosine/norm shifts, effective rank, domain probes,
attention summaries, and next-token distribution shifts. Raw tensors stay
local; training batches are not continuously archived.

## Hardware and cost boundary

Qualification requires two matching H100 80 GB SXM GPUs with NVLink, native BF16, at least 100 GB/s measured NCCL all-reduce bandwidth, and at least 8 GiB peak-reserved headroom. The bounded matrix contains 12 candidates; all preserve the same 16-window global batch.

The all-in ceiling is $60. A long-run launch requires a measured all-in
projection no greater than $48, leaving 20% contingency. Changing either
number requires explicit user approval. Instance lifecycle actions also
remain user-controlled.

## Authorization boundary

- Protocol design approved: `true`.
- GPU benchmark authorized: `false`.
- GPU rental authorized: `false`.
- Long training authorized: `false`.
- Cache deletion authorized: `false`.
