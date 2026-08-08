# Minerva 7B Instruct Remote FP16 LoRA Calibration

## Protocol

- Model: `sapienzanlp/Minerva-7B-instruct-v1.0`, revision
  `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`.
- GPU: Quadro RTX 8000 with 48,404.3 MiB reported CUDA memory.
- Base loading: unquantized FP16 parameters and FP16 computation.
- Adapter: rank 8, alpha 16, on `q_proj`, `k_proj`, `v_proj`, and `o_proj`.
- Batch: one context-512, microbatch-one chat example.
- Memory control: non-reentrant gradient checkpointing.
- Optimizer: `PagedAdamW8bit` at `2e-5`.
- Gate: one complete forward, backward, and optimizer update with at least
  4,096 MiB reserved and free CUDA headroom.

## Result

**Status: complete. Remote training fit decision: pass.**

| Measurement | Result |
| --- | ---: |
| Model parameters | 7,406,358,528 |
| Trainable adapter parameters | 6,815,744 (0.0920%) |
| Calibration loss | 3.308944 |
| Peak allocated CUDA memory | 14,584.5 MiB |
| Peak reserved CUDA memory | 14,706.0 MiB |
| Reserved headroom | 33,698.3 MiB |
| Free memory after update | 33,485.4 MiB |
| Required headroom | 4,096.0 MiB |
| Timed update | 0.804 seconds for 171 tokens |

The optimizer update completed and leaves more than 32 GiB beyond the declared
safety threshold. A larger GPU is not required for memory under this LoRA
protocol. Faster hardware could reduce rental time, but would not change the
modeling experiment.

This pass authorizes design of one full unquantized FP16 LoRA recipe. It does
not authorize training before the corrected V6 sonnet corpus and the complete
selection, checkpoint, resume, progress, and evaluation protocol are frozen.
