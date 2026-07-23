# Shared Sonnet Corpus Test Evaluation

This comparison scores only poems held out by both corpus versions. Each poem is scored independently, so no artificial document separator or cross-poem context contributes to the result.

## Evaluation Set

- Dataset selector: `expanded_with_petrarch`
- Shared held-out poems: 99
- Shared poem-ID SHA-256: `1ee6d307c861abcc614e67cb7d3491b3fb8bc4d33581d68b701cc90b9569cee3`
- Context length: 512
- Shared tokenizer SHA-256: `f62debf9411c9f7a743410b73c14a52f886e5189d5329d88b3cacf82bdf9e5eb`

## Results

| Arm | Selected step | Own validation loss | Shared target tokens | Shared test loss | Shared test perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 | 1,250 | 2.9863 | 23,738 | 3.0643 | 21.419 |
| v5 | 2,250 | 2.8999 | 23,738 | 3.0351 | 20.803 |

## Interpretation

v5 has the lower shared-test loss by 0.0292 nats per BPE token.
Own validation losses are shown only to document checkpoint selection; they are not compared because the corpus versions use different validation sets. The shared-test loss is directly comparable because both arms score the same poems with the same tokenizer and token positions.

## Per-Poem Evidence

Machine-readable per-poem losses: `reports/sonnet_corpus_scaling_shared_test_per_poem.json`.
