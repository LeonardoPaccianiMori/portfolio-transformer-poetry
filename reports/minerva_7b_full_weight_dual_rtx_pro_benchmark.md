# Minerva 7B Full-Weight Dual-RTX DDP Benchmark

## Protocol

- Model: `sapienzanlp/Minerva-7B-instruct-v1.0`, revision
  `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`.
- GPUs: two NVIDIA RTX PRO 6000 Blackwell Workstation Edition cards with
  97,249.5 MiB reported CUDA memory each.
- Interconnect: CUDA peer access over one PCIe host bridge; no NVLink.
- Software: PyTorch `2.12.0+cu130`, NCCL `2.29.7`, Transformers `4.57.1`,
  Accelerate `1.10.1`, and bitsandbytes `0.48.1`.
- Weights: one unquantized BF16 replica per rank, with all 7,399,542,784
  parameters trainable and no adapters.
- Optimizer: `PagedAdamW8bit` at `1e-6`, weight decay `0.01`.
- Execution: two-rank PyTorch DDP, SDPA, no gradient checkpointing.
- Candidates: global batches of 4,096 and 8,192 tokens with 25, 100, and
  250 MiB DDP gradient buckets.
- Measurement: one warm-up and five timed full optimizer updates per candidate.
- Fit gate: finite losses and gradients plus at least 8,192 MiB of free and
  peak-reserved headroom on every rank.
- Projection: one traversal of 351,271,297 tokens plus 15% execution overhead
  at the rented rate of `$2.162/hour`.

The benchmark retained no trained checkpoint and did not authorize the long
training run.

## Communication

The cards have no NVLink. A two-rank NCCL all-reduce of a 512 MiB BF16 payload
averaged 13.73 ms, or 39.11 GB/s algorithmic bandwidth. This is sufficient for
DDP, but synchronizing all 7.4 billion gradients remains a material part of
each update.

## Results

| Global tokens | Local microbatch | Bucket | Tokens/s | Peak reserved/GPU | Projected hours | Projected cost |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 4 | 25 MiB | 4,229.8 | 43,240 MiB | 26.5 | $57.36 |
| 4,096 | 4 | 100 MiB | 4,283.4 | 43,240 MiB | 26.2 | $56.64 |
| 4,096 | 4 | 250 MiB | 4,258.7 | 43,240 MiB | 26.3 | $56.97 |
| **8,192** | **8** | **25 MiB** | **5,813.5** | **58,348 MiB** | **19.3** | **$41.73** |
| 8,192 | 8 | 100 MiB | 5,804.3 | 58,348 MiB | 19.3 | $41.80 |
| 8,192 | 8 | 250 MiB | 5,773.2 | 58,348 MiB | 19.4 | $42.02 |

Every candidate passed. The selected recipe retained 38,901.5 MiB of
peak-reserved headroom and at least 23,638.6 MiB of free CUDA memory per GPU.
Its update-only corpus traversal is 16.8 hours; the 19.3-hour figure includes
the fixed 15% overhead allowance.

The retrieved JSON report has SHA-256
`0a7a0c4e201c35c0d005b065c719782986b667da3186b4df82b94e0cfafdf37d`.

## H100 Comparison

The H100 benchmark reached 5,222.4 tokens/s at a 4,096-token batch, projecting
21.5 hours and `$45.55`. Holding that batch fixed, two-GPU DDP reached only
4,283.4 tokens/s because communication outweighed the parallel compute gain.

The selected 8,192-token DDP recipe performs 11.3% more tokens per second and
projects 2.2 fewer hours and `$3.82` lower cost than the H100. This gain depends
on doubling the effective batch, so it is an execution candidate rather than a
strict same-optimization comparison.

## Decision

Use the current two-GPU host for the full-run protocol with global batch 8,192,
local microbatch eight, 25 MiB DDP buckets, BF16 SDPA, no gradient
checkpointing, and `PagedAdamW8bit`. The global batch remains small by language
model pretraining standards and preserves substantial memory headroom.

Do not rent a speculative four-GPU PCIe system. The measured two-GPU scaling is
communication-limited, while the four-GPU listings cost roughly two to four
times as much per hour. A larger system could reduce wall time but is not
supported as the lower-cost choice by current evidence.

This benchmark used repeated in-memory calibration windows and sequentially
updated one model across candidates. It establishes throughput and memory fit,
not output quality. Real shard loading, validation, preservation checks,
checkpoint serialization, and recovery remain covered only by the 15% planning
allowance until the complete trainer is measured.
