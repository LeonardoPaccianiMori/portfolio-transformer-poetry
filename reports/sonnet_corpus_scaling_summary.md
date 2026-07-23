# Sonnet Corpus Scaling Summary

## Locked Comparison Protocol

- Dataset selector: `expanded_with_petrarch`
- Parent checkpoint: `runs/pretraining_quality_swiglu_larger_200k_001/best_validation.pt`
- Parent tokenizer: `runs/pretraining_quality_swiglu_larger_200k_001/tokenizer.json`
- Parent tokenizer SHA-256: `321b3337281f3e2fcfe88699034762a5470bf66be44bd62d6f6252766c3a5865`
- Context length: 512
- Batch size: 2
- Maximum training steps: 20,000
- Evaluation interval: 250 steps
- Validation mode: fixed sequential windows
- Early-stopping patience: 8 evaluations
- Minimum validation improvement: 0.01
- Learning rate: 3e-5
- Seed: 1337

## Corpus Measurements

| Version | Poems | Train / val / test poems | Train / val / test BPE tokens | Total BPE tokens | Vocabulary | Added characters | Validation windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 921 | 736 / 93 / 92 | 177,546 / 22,338 / 22,151 | 222,035 | 8,000 -> 8,003 | 3 | 43 |
| v4 | 1,011 | 809 / 103 / 99 | 196,422 / 25,006 / 23,935 | 245,363 | 8,000 -> 8,004 | 4 | 48 |
| v5 | 1,875 | 1,486 / 191 / 198 | 365,261 / 46,896 / 48,713 | 460,870 | 8,000 -> 8,004 | 4 | 91 |

## Manifest Provenance

- v1: `data/metadata/poems_manifest.csv` (SHA-256 `be193be024173dbc0a061c4f5c861b09004b567121b0932af0005ad45672eb64`)
- v4: `data/metadata/sonnets_expanded_v4_manifest.csv` (SHA-256 `c4ec631417c14349de46987a70ab5385ff13bba5930c453a1e0e22525ab8a6b4`)
- v5: `data/metadata/sonnets_expanded_v5_manifest.csv` (SHA-256 `d71abe5bbc048392b7579702124f25bd6dedb400a47a0c171e3f4e6e0aae6275`)

## Shared Test Records

| Versions | Shared test poems |
| --- | ---: |
| v1 + v4 | 92 |
| v1 + v5 | 92 |
| v4 + v5 | 99 |
| v1 + v4 + v5 | 92 |

## Interpretation Rules

- Select each run's checkpoint using only that run's own validation split.
- Do not compare validation-loss values directly across corpus versions, because their validation records and token counts differ.
- Compare generated outputs with identical prompts, seeds, and decoding settings.
- Compare held-out behavior on the shared test subset when a record-level cross-version metric is needed.
- Treat tokenizer vocabulary growth as part of the corpus change: parent token IDs stay fixed and only missing literal characters are appended.
