# Minerva 7B Full-Weight Dual-H100 Optimization

## Hardware And Fixed Model

- Model: `sapienzanlp/Minerva-7B-instruct-v1.0`, revision
  `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`.
- GPUs: two NVIDIA H100 80GB HBM3 SXM devices at `$3.495/hour` total.
- Interconnect: NVLink (`NV6` topology); a 512 MiB NCCL all-reduce sustained
  approximately 113 GB/s.
- Software: PyTorch `2.12.0+cu126`, NCCL `2.29.3`, Transformers `4.57.1`,
  Accelerate `1.10.1`, and bitsandbytes `0.48.1`.
- Weights: one unquantized BF16 replica per rank, with all 7,399,542,784
  parameters trainable and no adapters.
- Optimizer: `PagedAdamW8bit`; attention used SDPA; gradient checkpointing was
  disabled.
- Fixed sequence length: 512 tokens.

All runs in this report were execution measurements. They retained no trained
checkpoint and provide no output-quality evidence.

## Optimization Sequence

| Stage | Global tokens/update | Local microbatch | Accumulation | Execution | Tokens/s | Projected hours | Projected cost |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Dual-H100 endpoint | 4,096 | 4 | 1 | eager | 7,690.6 | 14.59 | $50.99 |
| Intermediate batch | 7,168 | 7 | 1 | eager | 10,563.9 | 10.62 | $37.12 |
| Maximum local batch | 8,192 | 8 | 1 | eager | 11,546.2 | 9.72 | $33.97 |
| Accumulation-8 short probe | 65,536 | 8 | 8 | eager | 14,934.1 | 7.51 | $26.26 |
| Accumulation-8 endurance | 65,536 | 8 | 8 | eager | 14,969.6 | 7.50 | $26.20 |
| **Compiled accumulation-8** | **65,536** | **8** | **8** | **`torch.compile`, default** | **17,356.6** | **6.47** | **$22.60** |

Projections cover one traversal of 351,271,297 training tokens and add the
fixed 15% allowance for data loading, validation, and checkpointing. The
selected batch requires 5,360 optimizer updates for that token budget.

The two DDP bucket sizes tested in the broad sweep were 25 and 250 MiB. The
25 MiB bucket was consistently at least as fast and is retained. PyTorch
`static_graph=True` completed accumulation-one work but raised an internal DDP
reducer assertion when combined with `no_sync()` accumulation, so the selected
recipe uses normal DDP.

## Endurance Evidence

The selected eager recipe completed five warm-up updates and 100 measured
updates in a fresh process. Throughput was 14,969.6 tokens/s over 6,553,600
measured tokens. Losses and pre-clipping gradient norms remained finite, and
there was no slowdown or progressive memory growth.

Peak allocation was 71,925.9 MiB and peak reservation was 72,688.0 MiB per GPU.
CUDA reported only 1 MiB free while gradients and allocator caches were live,
but releasing gradients and cached blocks restored 44,159 MiB free on each
GPU. A post-training validation forward pass completed successfully. The
two-window validation mean changed from 1.6805 to 1.7029.

The endurance run intentionally repeated a tiny fixed calibration pool. Its
training loss therefore fell almost to zero. This is expected for this stress
test and must not be interpreted as a useful training trajectory or as evidence
about generalization on the real corpus.

## Compile Decision

The single bounded `torch.compile(mode="default")` probe used two untimed
warm-ups and ten timed optimizer updates. It reached 17,356.6 tokens/s, 15.9%
above the 100-update eager result. Peak allocation also fell to 66,564.5 MiB.
The improvement is large enough to adopt compilation for the real run, subject
to resume and checkpoint tests in the trainer itself.

No local-microbatch-nine run was performed. The endurance recipe already
reached the practical live-memory ceiling, and a microbatch-nine,
accumulation-one recipe could not plausibly exceed the measured
microbatch-eight, accumulation-eight throughput enough to justify an expected
out-of-memory attempt.

## Decision

Use two-rank DDP with local microbatch 8, gradient accumulation 8, a 25 MiB
bucket, BF16 trainable weights, `PagedAdamW8bit`, SDPA, no gradient
checkpointing, normal DDP with `no_sync()` on intermediate microbatches, and
`torch.compile(mode="default")`. Learning-rate scheduling, validation, and
checkpoint cadence must be token-based because the selected 65,536-token batch
reduces the optimizer-step count.

This decision freezes the execution backend, not the scientific training
protocol. The corpus mixture, token budget, learning-rate schedule,
preservation gates, checkpoint/resume behavior, and abort rules still require a
separately reviewed configuration before the long run starts.

## Local Evidence Hashes

- Endpoint benchmark: `daec332154db63e1873465ea85f759be6e184aee271fa601a93e18586b6042d0`.
- Intermediate benchmark: `64b28651d39191fdd31f3d0e4f1e52f0a8124e0243eff437ac5c1be1b47fbb87`.
- Accumulation sweep: `906199061942012662c3d57b2ce665b2eb265e5f12b650797bdab6a8fdbd7bb6`.
- Endurance qualification: `63dd80fe59db911c51b2a644f27da1f1bb78f3147039666872b3b39c79be6c41`.
- Compile probe: `c8fa58f1faf7da58b978d600963fb122b9e3465845b50be3fb5fdfde56b7064d`.
