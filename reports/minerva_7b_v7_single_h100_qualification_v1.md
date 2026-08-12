# Minerva 7B V7 Single-H100 Qualification

Checkpoint 8H accepts the rented single-H100-SXM host for the three-stage V7
full-weight run. The bounded qualification accessed training and validation
artifacts only, did not open V7 test material, did not retain a quality
checkpoint, and did not start long training.

## Machine and invariant workload

- GPU: one NVIDIA H100 80GB HBM3 with 81,109.8 MiB visible to PyTorch.
- Runtime: PyTorch 2.12.0+cu126, CUDA 12.6, NCCL 2.29.3, native BF16.
- Model audit: all 7,399,542,784 parameters were trainable BF16 weights; no
  adapter or weight quantization was present. PagedAdamW8bit compressed only
  optimizer state.
- Scientific batch: 16 frozen context-2,048 windows / 32,768 target tokens per
  update. Every candidate preserved that batch exactly.

## Candidate results

| Microbatch | Accumulation | Checkpointing | Execution | Result | Tokens/s | Headroom |
| ---: | ---: | :---: | --- | --- | ---: | ---: |
| 1 | 16 | on | eager | passed | 5,731.8 | 34,744 MiB |
| 1 | 16 | on | compile/default | passed | 6,789.1 | 34,846 MiB |
| 1 | 16 | off | eager | passed | 7,160.5 | 22,522 MiB |
| 1 | 16 | off | compile/default | **selected** | **8,273.6** | **24,666 MiB** |
| 2 | 8 | on | eager | passed | 6,000.0 | 32,526 MiB |
| 2 | 8 | on | compile/default | passed | 7,039.3 | 32,804 MiB |
| 2 | 8 | off | eager | failed 8-GiB gate | 6,362.2 | 6,946 MiB |
| 2 | 8 | off | compile/default | passed | 7,203.8 | 9,904 MiB |
| 4 | 4 | on | eager | passed | 6,167.1 | 26,782 MiB |
| 4 | 4 | on | compile/default | passed | 7,211.4 | 27,948 MiB |
| 4 | 4 | off | eager | CUDA OOM | — | — |
| 4 | 4 | off | compile/default | CUDA OOM | — | — |

The selected candidate is the fastest configuration that passes every frozen
preliminary gate. Its 24,665.8 MiB reserved-memory margin is 16,473.8 MiB above
the required 8-GiB floor. Disabling gradient checkpointing and compiling the
microbatch-one configuration increased throughput by 44.3% relative to its
checkpointed eager counterpart while preserving a large safety margin.

## Checkpoint and resume proof

The selected runtime completed held-out validation before and after a real
optimizer update, wrote a hash-manifested atomic checkpoint, then reloaded it in
a fresh process. The reload reproduced the stage/update counters, next window
identity, learning rate, model and optimizer state, per-rank RNG, and sampler
state. The following update completed with finite loss and gradient norm. The
temporary 28-GB proof checkpoint was deleted only after that verification.

This proof covers interruption/resume mechanics. The long-run trainer still
needs a separately approved single-GPU enablement checkpoint before the user
can launch stage 1.

## Runtime and cost projection

At the measured 8,273.6 target tokens/s and `$2.617/hour`, applying the frozen
1.25x allowance gives:

| Stage | Updates | Target tokens | Update-only | All-in | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical general | 2,065 | 67,665,920 | 2.27 h | 2.84 h | $7.43 |
| Non-sonnet poetry | 760 | 24,903,680 | 0.84 h | 1.05 h | $2.74 |
| Sonnets | 135 | 4,423,680 | 0.15 h | 0.19 h | $0.49 |
| **Total** | **2,960** | **96,993,280** | **3.26 h** | **4.07 h** | **$10.65** |

These are qualification-based projections, not guarantees. Complete
validation, snapshot writes, compilation behavior, and filesystem throughput
can change actual stage times. They remain far below the frozen `$48` launch
gate and `$60` absolute ceiling.

## Decision and evidence boundary

The H100 is qualified, but long training remains unauthorized until the
single-GPU trainer, launch contract, snapshot retention, and monitoring paths
are adapted and verified. The user will personally launch each actual training
stage from an exact command supplied after that checkpoint.

All twelve candidate JSON files, save/resume proof reports, final report, and
run log were retrieved and matched their remote SHA-256 hashes. Their hashes
are recorded in the companion JSON report. Machine-local details remain local.
The reusable model and data caches remain present. No instance lifecycle action,
V7 test access, quality-checkpoint retention, cache deletion, or long training
occurred.
