# Pretraining Data-Gap Report

This report fixes the scale target for the next broader-Italian corpus revision. It measures encoded BPE tokens, not whitespace words or raw characters, because BPE tokens are the units consumed by the model.

## Corpus Scale

| Measurement | Value |
| --- | ---: |
| Active sources | 33 |
| Cleaned characters | 43,418,117 |
| Current encoded corpus tokens | 17,294,440 |
| Current training tokens | 17,121,479 |
| Current validation tokens | 172,961 |
| Target unique corpus tokens | 75,000,000 |
| Additional unique tokens needed | 57,705,560 |
| Target currently assembled | 23.1% |
| Current tokenizer fertility | 2.511 characters/token |

The 75M figure is a corpus-assembly target before applying the deterministic training/validation split. It is not a claim that 75M tokens alone guarantee coherent generation.

## Training Exposure Budget

| Measurement | Value |
| --- | ---: |
| Model parameters | 33,671,312 |
| Batch size | 2 |
| Context length | 512 |
| Tokens processed per step | 1,024 |
| Completed pretraining steps | 200,000 |
| Completed token exposures | 204,800,000 |
| Completed passes over current train stream | 11.96 |
| Proposed maximum pretraining steps | 650,000 |
| Proposed maximum token exposures | 665,600,000 |
| Proposed passes over 75M-token corpus | 8.87 |

For orientation only, a commonly cited compute-optimal heuristic uses roughly 20:1 training tokens per parameter. For this model that is 673,426,240 exposures; the 75M-token corpus target is 11.1% of that rough exposure budget. This heuristic is not a training prescription, especially for a small historical corpus with repeated passes.

## Existing Composition

- Largest work: `ll_ramusio_navigazioni_viaggi` at 27.56% of cleaned characters.
- Largest author: `Giovan Battista Ramusio` at 27.56% of cleaned characters.
- The project decision is to retain the complete Ramusio compilation; this report records its share but does not propose capping it.

## Next Gate

Before page-level extraction, each candidate source must pass the documented metadata, license, composition, and representative-text gate. Only selected core-compatible prose sources proceed to the expensive revision-pinned audit and builder pipeline.
