# Minerva 7B Historical FP16 LoRA Calibration

## Protocol

- Model: `sapienzanlp/Minerva-7B-instruct-v1.0`, revision
  `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`.
- GPU: Quadro RTX 8000 with 48,404.3 MiB reported CUDA memory.
- Base weights: unquantized FP16, frozen.
- Trainable parameters: rank-8, alpha-16 LoRA on `q_proj`, `k_proj`,
  `v_proj`, and `o_proj`.
- Optimizer: standard `torch.optim.AdamW`, not an 8-bit optimizer.
- Update: seven historical and one PAISÀ replay microbatch, each 512 tokens.
- Measurement: two warmup updates followed by ten timed updates.

## Result

**Status: pass.**

| Measurement | Result |
| --- | ---: |
| Trainable adapter parameters | 6,815,744 |
| Tokens per update | 4,096 |
| Mean timed loss | 2.9998 |
| Ten timed updates | 35.77 seconds |
| Throughput | 1,145.2 tokens/second |
| Peak allocated CUDA memory | 14,679.7 MiB |
| Peak reserved CUDA memory | 14,786.0 MiB |
| Free CUDA memory after calibration | 33,405.4 MiB |
| Required free-memory floor | 4,096.0 MiB |
| Planned full updates | 6,762 |
| Update-only full-plan estimate | 6 hours 43 minutes |

The exact recipe leaves more than 32 GiB beyond its required safety floor. The
long run should take approximately 7–9 hours after including initial baseline
measurement, fourteen or fewer periodic validation rounds, and atomic adapter
and resume checkpoint writes. Early stopping may finish it sooner, but the
upper plan remains two historical passes.
