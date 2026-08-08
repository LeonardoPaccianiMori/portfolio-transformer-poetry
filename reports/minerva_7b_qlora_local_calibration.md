# Minerva 7B Instruct Local QLoRA Calibration

## Protocol

- Model: `sapienzanlp/Minerva-7B-instruct-v1.0`, revision
  `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`.
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU with 5,804.3 MiB reported memory.
- Base loading: 4-bit NF4 with double quantization and FP16 computation.
- Adapter: rank 8, alpha 16, on `q_proj`, `k_proj`, `v_proj`, and `o_proj`.
- Batch: one 512-token example with gradient checkpointing.
- Optimizer: `PagedAdamW8bit` at `2e-5`.
- Gate: complete forward, backward, and optimizer update with at least 512 MiB
  measured CUDA headroom.

## Result

**Status: out of memory. Local training fit decision: reject.**

| Measurement | Result |
| --- | ---: |
| Peak allocated CUDA memory | 5,544.2 MiB |
| Peak reserved CUDA memory | 5,636.0 MiB |
| Reserved headroom | 168.3 MiB |
| Free memory after exception cleanup | 137.3 MiB |
| Required headroom | 512.0 MiB |

The allocation failed during the forward/backward stage while requesting an
additional 102 MiB. The optimizer update did not run, so its first-update state
allocation was not measured. The free-memory value was recorded only after the
exception was caught and cached memory was cleared; it is not usable training
headroom.

This closes 7B QLoRA training on the local 6 GiB GPU under the fixed protocol.
It does not reject the model or LoRA method. The approved replacement is a
remote unquantized FP16 baseline and LoRA calibration on a 48 GiB Quadro RTX
8000 before any full 7B adaptation recipe is frozen.
