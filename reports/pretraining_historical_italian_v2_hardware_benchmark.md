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
- Candidate set: `historical_v2_quality_swiglu`
- Dataset version: `pretraining_historical_italian_v2`
- Source count: `36`
- Normalization: `layer_norm`
- Position encoding: `learned_absolute`
- Tied token embeddings: `False`

## Results

| Candidate | Status | Params | Batch | Seconds/Step | Tokens/Sec | Peak CUDA MiB | Train Loss | Validation Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| max | ok | 110,025,856 | 1 | 0.1275 | 4015.1496 | 2451.9717 | 4.4465 | 6.9868 |
| xl | ok | 164,151,244 | 1 | 0.1853 | 2763.0102 | 3539.0303 | 4.3897 | 4.3708 |
| xxl | ok | 234,839,008 | 1 | 0.2578 | 1986.0630 | 4851.9404 | 4.1788 | 4.3105 |
| ceiling | error |  | 1 |  |  | 5351.0015 |  |  |

## Interpretation

Use this report to choose the largest model that fits reliably and still
processes enough tokens per second for a long local pretraining run.
A successful benchmark does not prove final generation quality; it only
measures practical training throughput and memory for candidate sizes.
