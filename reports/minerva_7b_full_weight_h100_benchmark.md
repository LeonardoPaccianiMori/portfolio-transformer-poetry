# Minerva 7B Full-Weight H100 Throughput Benchmark

## Protocol

- Model: `sapienzanlp/Minerva-7B-instruct-v1.0`, revision
  `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`.
- GPU: NVIDIA H100 NVL with 95,321.4 MiB reported CUDA memory.
- Weights: unquantized BF16 with all 7,399,542,784 parameters trainable.
- Optimizer: bitsandbytes `PagedAdamW8bit` at `1e-6`, weight decay `0.01`.
- Attention: PyTorch scaled dot-product attention (`sdpa`).
- Fixed effective batch: eight 512-token sequences, or 4,096 tokens/update.
- Candidates: microbatch 1, 2, 4, and 8 with gradient checkpointing both on
  and off; accumulation was adjusted to keep the effective batch fixed.
- Measurement: one warm-up and five timed full optimizer updates per candidate.
- Fit gate: finite losses and gradient norms plus at least 8,192 MiB of both
  free post-update memory and peak-reserved headroom.
- Projection: one traversal of 351,271,297 training tokens, with 15% added for
  data access, evaluation, and checkpoint overhead at `$2.12/hour`.

The benchmark retained no trained checkpoint and did not authorize the long
training run.

## Result

**Selected recipe: microbatch 8, accumulation 1, gradient checkpointing off.**

| Candidate | Tokens/s | Peak reserved | Headroom | Projected hours | Projected cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| GC on, micro 1, accum 8 | 3,104.2 | 29,218 MiB | 66,103 MiB | 36.1 | $76.63 |
| GC on, micro 2, accum 4 | 3,982.1 | 29,594 MiB | 65,727 MiB | 28.2 | $59.74 |
| GC on, micro 4, accum 2 | 4,224.8 | 30,734 MiB | 64,587 MiB | 26.6 | $56.31 |
| GC on, micro 8, accum 1 | 4,488.4 | 29,372 MiB | 65,949 MiB | 25.0 | $53.00 |
| GC off, micro 1, accum 8 | 3,924.0 | 32,374 MiB | 62,947 MiB | 28.6 | $60.62 |
| GC off, micro 2, accum 4 | 4,609.6 | 36,068 MiB | 59,253 MiB | 24.3 | $51.61 |
| GC off, micro 4, accum 2 | 5,007.5 | 43,600 MiB | 51,721 MiB | 22.4 | $47.51 |
| **GC off, micro 8, accum 1** | **5,222.4** | **44,228 MiB** | **51,093 MiB** | **21.5** | **$45.55** |

Every candidate passed the numerical and memory gates. The selected recipe is
68.2% faster than checkpointed microbatch one. Its update-only traversal is
18.7 hours; the reported 21.5-hour estimate includes the fixed 15% overhead.

The retrieved JSON report has SHA-256
`eff598f13ebaa4047d8ce88eda135f4da8883646796f248d72046e2997b02b8d`.

## Interpretation

Gradient checkpointing is unnecessary on this 95 GiB H100 for the tested
effective batch. Disabling it avoids recomputing activations during backward,
while microbatch eight removes gradient-accumulation overhead. More than 35 GiB
of CUDA free memory remained after the selected updates, so the result is not a
borderline fit.

This is a short in-memory throughput benchmark, not a training-quality result.
It repeatedly uses a small fixed window pool, mutates one model sequentially
across candidates, and excludes real shard loading, full validation, checkpoint
serialization, interruption/resume, and preservation gates. The 15% allowance
is therefore a planning estimate rather than a guaranteed runtime.

A two-GPU host is not automatically faster. The current trainer is single-GPU,
and full-weight data-parallel training must synchronize approximately 7.4
billion gradients every update. A candidate multi-GPU machine needs its own
fixed-batch distributed benchmark before its advertised aggregate throughput or
hourly price can be compared with this H100 result.
