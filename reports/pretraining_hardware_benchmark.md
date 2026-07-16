# Pretraining Hardware Benchmark

This report benchmarks candidate broader-pretraining model sizes on the
local hardware using the current BPE-encoded broader Italian corpus.

## Configuration

- Device: `cuda:0`
- CUDA available: `True`
- Vocabulary size: `8000`
- Context length: `512`
- Warmup steps: `10`
- Timed steps: `100`
- Evaluation batches: `1`
- Learning rate: `0.0003`

## Results

| Candidate | Status | Params | Batch | Seconds/Step | Tokens/Sec | Peak CUDA MiB | Train Loss | Validation Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | ok | 8,969,536 | 8 | 0.1008 | 40652.7616 | 1365.6851 | 4.2735 | 4.8703 |
| medium | ok | 20,535,872 | 4 | 0.0973 | 21046.5999 | 1155.0044 | 4.2817 | 4.5554 |
| larger | ok | 33,669,952 | 2 | 0.0775 | 13215.0749 | 1002.1670 | 4.1357 | 5.3311 |
| upper | ok | 59,792,960 | 1 | 0.0854 | 5994.3443 | 1357.4014 | 4.6548 | 4.4406 |
| max | ok | 97,717,568 | 1 | 0.1252 | 4087.8556 | 2121.2881 | 4.7353 | 4.5719 |

## Interpretation

Use this report to choose the largest model that fits reliably and still
processes enough tokens per second for a long local pretraining run.
A successful benchmark does not prove final generation quality; it only
measures practical training throughput and memory for candidate sizes.
