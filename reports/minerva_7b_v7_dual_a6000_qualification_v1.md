# Minerva 7B V7 Dual-A6000 Qualification

Checkpoint 8G rejects the rented dual-RTX-A6000 host for the three-stage V7
full-weight run. The bounded qualification accessed training and validation
artifacts only, did not open V7 test material, did not retain a quality
checkpoint, and did not start long training.

## Machine and invariant workload

- GPUs: two NVIDIA RTX A6000 devices with 48,539.4 MiB visible CUDA memory each.
- Runtime: PyTorch 2.12.0+cu130, CUDA 13.0, NCCL 2.29.7, native BF16.
- Interconnect: `PIX`, bidirectional peer access, and 3.54 GB/s measured
  512-MiB NCCL algorithmic all-reduce bandwidth.
- Model audit: all 7,399,542,784 parameters were trainable BF16 weights; no
  adapter or weight quantization was present. PagedAdamW8bit compressed only
  optimizer state.
- Scientific batch: 16 frozen context-2,048 windows / 32,768 target tokens per
  update. Every candidate preserved that batch exactly.

## Candidate results

| Microbatch | Accumulation | Checkpointing | Execution | Result | Tokens/s | Headroom/GPU |
| ---: | ---: | :---: | --- | --- | ---: | ---: |
| 1 | 8 | on | eager | failed 8-GiB gate | 1,371.7 | 1,915 MiB |
| 1 | 8 | on | compile/default | failed 8-GiB gate | 1,441.8 | 2,283 MiB |
| 1 | 8 | off | eager | CUDA OOM | — | — |
| 1 | 8 | off | compile/default | CUDA OOM | — | — |
| 2 | 4 | on | eager | CUDA OOM | — | — |
| 2 | 4 | on | compile/default | CUDA OOM | — | — |
| 2 | 4 | off | eager | CUDA OOM | — | — |
| 2 | 4 | off | compile/default | CUDA OOM | — | — |

The two runnable candidates each completed three warm-up plus twenty timed
updates with finite losses and gradients and zero measured reserved-memory
growth. Compilation improved steady-state throughput by 5.1% and headroom by
368 MiB, but the resulting 2,283 MiB reserve remained 5,909 MiB below the
frozen 8-GiB reliability gate. It leaves too little protection for validation,
checkpoint serialization, allocator fragmentation, fresh-process restore, and
multi-hour behavior. A 4-GiB conditional gate would also fail. The compiled
candidate clears 2 GiB by only 235 MiB; this is not adequate evidence for a
costly full run.

## Cost interpretation

At the measured 1,441.8 tokens/s and `$1.008/hour`, the rejected compiled
candidate would project to 18.69 update-only hours or 23.36 all-in hours after
the frozen 1.25x contingency, costing `$23.55`. Its stage projections would be:

| Stage | Updates | Target tokens | Update-only | All-in | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical general | 2,065 | 67,665,920 | 13.04 h | 16.30 h | $16.43 |
| Non-sonnet poetry | 760 | 24,903,680 | 4.80 h | 6.00 h | $6.05 |
| Sonnets | 135 | 4,423,680 | 0.85 h | 1.07 h | $1.07 |

These figures demonstrate that the host is fast and inexpensive enough, but
they do not override the memory gate. They are not authorization to train.

## Decision

No candidate passed preliminary qualification, so validation transition,
atomic checkpoint installation, and fresh-process resume proofs were correctly
not attempted. The next hardware checkpoint is a separately approved single
H100 SXM qualification. The user will personally launch every actual training
stage from exact commands supplied after hardware qualification; the assistant
may monitor progress but will not launch training.

The retrieved machine-local evidence consists of eight candidate JSON files and
one final report. Their hashes are recorded in the companion JSON report. The
raw files remain local because they contain machine-execution detail rather
than public corpus data.

The focused qualification suite passes 11/11 tests and the complete repository
suite passes 1,062/1,062.
