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
- Candidate set: `quality_swiglu`

## Results

| Candidate | Status | Params | Batch | Seconds/Step | Tokens/Sec | Peak CUDA MiB | Train Loss | Validation Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| larger | ok | 33,671,312 | 2 | 0.0790 | 12956.1006 | 1118.3662 | 4.4307 | 4.4828 |
| upper | ok | 59,807,900 | 1 | 0.0866 | 5914.7155 | 1453.4541 | 4.7206 | 4.3026 |
| max | ok | 97,729,856 | 1 | 0.1287 | 3979.7132 | 2239.4756 | 4.3351 | 4.5612 |

## Interpretation

Use this report to choose the largest model that fits reliably and still
processes enough tokens per second for a long local pretraining run.
A successful benchmark does not prove final generation quality; it only
measures practical training throughput and memory for candidate sizes.
