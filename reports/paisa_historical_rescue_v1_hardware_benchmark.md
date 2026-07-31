# Pretraining Hardware Benchmark

This report benchmarks candidate broader-pretraining model sizes on the
local hardware using the current BPE-encoded broader Italian corpus.

## Configuration

- Device: `cuda:0`
- CUDA available: `True`
- Vocabulary size: `16000`
- Context length: `512`
- Warmup steps: `10`
- Timed steps: `100`
- Evaluation batches: `1`
- Learning rate: `0.0003`
- Candidate set: `paisa_historical_rescue`
- Dataset version: `paisa_historical_rescue_v1`
- Stream count: `2`
- Document count: `375382`
- Train split: `paisa_train`
- Validation split: `paisa_validation`
- Normalization: `layer_norm`
- Position encoding: `learned_absolute`
- Tied token embeddings: `False`

## Results

| Candidate | Status | Params | Microbatch | Accumulation | Tokens/Update | Seconds/Update | Tokens/Sec | Peak CUDA MiB | Train Loss | Validation Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rescue_upper_micro1 | ok | 70,055,900 | 1 | 8 | 4,096 | 0.5658 | 7239.7395 | 1704.8813 | 4.6825 | 5.0278 |
| rescue_upper_micro2 | ok | 70,055,900 | 2 | 4 | 4,096 | 0.5718 | 7163.2261 | 2175.2573 | 4.7239 | 4.7650 |
| rescue_upper_micro4 | ok | 70,055,900 | 4 | 2 | 4,096 | 0.5169 | 7924.4069 | 3092.3262 | 4.8781 | 4.7366 |

## Interpretation

Use this report to choose the largest model that fits reliably and still
processes enough tokens per second for a long local pretraining run.
A successful benchmark does not prove final generation quality; it only
measures practical training throughput and memory for candidate sizes.
