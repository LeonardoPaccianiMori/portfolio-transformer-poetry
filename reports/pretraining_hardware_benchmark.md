# Pretraining Hardware Benchmark

This report benchmarks candidate broader-pretraining model sizes on the
local hardware using the current BPE-encoded broader Italian corpus.

## Configuration

- Device: `cuda:0`
- CUDA available: `True`
- Vocabulary size: `8000`
- Context length: `512`
- Warmup steps: `3`
- Timed steps: `20`
- Evaluation batches: `1`
- Learning rate: `0.0003`

## Results

| Candidate | Status | Params | Batch | Seconds/Step | Tokens/Sec | Peak CUDA MiB | Train Loss | Validation Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | ok | 8,969,536 | 8 | 0.1000 | 40944.5170 | 1365.6851 | 5.1962 | 5.1059 |
| medium | ok | 20,535,872 | 4 | 0.0963 | 21275.4380 | 1155.0044 | 4.7810 | 4.5908 |
| larger | ok | 33,669,952 | 2 | 0.0762 | 13439.2360 | 1002.1670 | 4.9230 | 4.5541 |
| upper | ok | 59,792,960 | 1 | 0.0778 | 6577.7616 | 1357.4014 | 4.5085 | 4.6495 |

## Interpretation

Use this report to choose the largest model that fits reliably and still
processes enough tokens per second for a long local pretraining run.
A successful benchmark does not prove final generation quality; it only
measures practical training throughput and memory for candidate sizes.
