# PAISA-To-Historical Rescue Training Plan

This is the single final from-scratch rescue experiment. It uses the
GPU-selected fixed architecture and sequential PAISA then historical
Italian training stages; it is not an architecture or hyperparameter sweep.

## Hardware Selection

- Candidate: `rescue_upper_micro4`
- Microbatch size: `4`
- Gradient accumulation: `2`
- Targets per optimizer update: `4,096`
- Measured throughput: `7,924.4 tokens/s`
- Peak CUDA memory: `3,092.3 MiB`

## Architecture

- Embedding dim: `640`
- Num layers: `10`
- Num heads: `10`
- Head dim: `64`
- Feed forward dim: `1707`
- Feed forward type: `swiglu`
- Normalization type: `layer_norm`
- Position encoding type: `learned_absolute`
- Tie token embeddings: `False`
- Context length: `512`
- Max context length: `512`
- Microbatch size: `4`
- Gradient accumulation steps: `2`

## Stages

| Stage | Stream | Pass cap | Updates | Target tokens | Validation |
| --- | --- | ---: | ---: | ---: | --- |
| modern_italian_pretraining | `paisa_train` | 3 | 412,998 | 1,691,641,335 | random_batches every 2,000 updates |
| historical_italian_annealing | `historical_train` | 12 | 56,213 | 230,251,764 | sequential_windows every 500 updates |

The update budgets use floor division so neither stage exceeds its fixed
pass cap. The small unused remainder is below one optimizer update per
stage. PAISA selection uses 20 sampled validation batches; historical
selection uses all non-overlapping validation windows. The historical
stage loads the selected PAISA weights but starts a fresh AdamW optimizer,
so PAISA optimizer moments are not transferred across domains.

## Runtime

The measured-throughput estimate for forward/backward updates only is `67.4 hours`. The operational estimate is 75-90 hours after validation, atomic checkpoints, and normal runtime variation.
