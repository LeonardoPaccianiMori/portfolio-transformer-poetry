# Minerva 7B Full-Weight H100 Calibration

## Protocol

- Model: `sapienzanlp/Minerva-7B-instruct-v1.0`, revision
  `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`.
- GPU: NVIDIA H100 NVL with 95,321.4 MiB reported CUDA memory.
- Weights: unquantized BF16 with every model parameter trainable.
- Adapter scope: none; no LoRA, PEFT, or quantized parameter path.
- Optimizer: bitsandbytes `PagedAdamW8bit` at constant `1e-6` with weight
  decay `0.01`.
- Batch: microbatch one, context 512, no gradient accumulation.
- Memory control: non-reentrant gradient checkpointing.
- Updates: five fixed modern/historical/modern/historical/modern windows.
- Gate: finite losses and pre-clipping gradient norms, a verified 100%
  trainable BF16 parameter fraction, and at least 8,192 MiB post-optimizer and
  peak-reserved headroom.

The complete underlying data preparation contains 339,147,183 / 3,475,648
PAISA train/validation tokens and 12,124,114 / 113,526 historical
train/validation tokens under the pinned Minerva tokenizer. The GPU bundle
contained only seven fixed 512-token windows and aggregate metadata.

## Result

**Status: complete. Full-weight training fit decision: pass.**

| Measurement | Result |
| --- | ---: |
| Total model parameters | 7,399,542,784 |
| Trainable parameters | 7,399,542,784 (100.0000%) |
| Parameter dtype | BF16 |
| Quantized/adapted parameters | None |
| Peak allocated CUDA memory | 28,516.1 MiB |
| Peak reserved CUDA memory | 28,736.0 MiB |
| Reserved headroom | 66,585.4 MiB |
| Minimum free memory after optimizer allocation | 51,608.6 MiB |
| Required headroom | 8,192.0 MiB |
| First update, including optimizer-state allocation | 5.035 seconds |
| Mean of subsequent four updates | 0.388 seconds |
| Subsequent-update calibration throughput | approximately 1,320 tokens/second |
| Calibration report SHA-256 | `325133a462f4c99af578d84142994ce65ff52afcde71239ddc5200c6544415a0` |

Every loss and pre-clipping gradient norm was finite:

| Update | Split | Loss | Pre-clipping gradient norm |
| ---: | --- | ---: | ---: |
| 1 | PAISA train | 1.0678 | 43.50 |
| 2 | Historical train | 3.2375 | 35.25 |
| 3 | PAISA train | 1.4047 | 45.00 |
| 4 | Historical train | 3.3827 | 39.00 |
| 5 | PAISA train | 2.4340 | 24.25 |

The two-window validation mean changed from `1.67965` before calibration to
`1.68054` after five updates. PAISA validation changed from `1.21894` to
`1.22195`; historical validation changed from `2.14036` to `2.13913`. These
five updates are too few and too narrowly sampled to support a quality claim;
they only show finite, executable full-weight optimization.

## Interpretation

The H100 has ample memory for the measured microbatch-one recipe. The first
optimizer update successfully allocated state for all 7.40 billion parameters,
and the minimum remaining free memory exceeded the gate by more than 43 GiB.
The calibration retained no trained checkpoint.

At the measured microbatch-one steady-state rate, one traversal of the
351,271,297 training tokens would take about 74 update-only hours. Validation,
checkpointing, data access, and preservation checks would raise a conservative
single-pass estimate to roughly 82-90 hours, or about $174-$191 at the rented
instance's $2.12 hourly rate. A bounded microbatch throughput calibration may
reduce this estimate before a long run is approved.

This pass authorizes design and hardware calibration of the long mixed-corpus
recipe. It does not itself authorize that expensive run: mixture weighting,
effective batch size, learning-rate schedule, token/pass budget, preservation
gates, checkpoint/resume behavior, abort rules, and projected cost still need
to be frozen first.
