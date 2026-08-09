# Project Gutenberg Full-Text Value Gate

## Result

Inspected 42 deterministic samples from 416 metadata-compatible records.

| Sample status | Records |
| --- | ---: |
| `sample_quality_pass` | 42 |

## Role Projections

| Role | Candidates | Samples | Mean chars | Median chars | Projected chars |
| --- | ---: | ---: | ---: | ---: | ---: |
| `historical_general_candidate` | 66 | 10 | 955,230 | 849,642 | 63,045,206 |
| `historical_non_sonnet_poetry_candidate` | 41 | 14 | 344,672 | 166,628 | 14,131,540 |
| `nineteenth_century_bridge_candidate` | 308 | 17 | 448,591 | 309,155 | 138,166,100 |
| `sonnet_specialization_candidate` | 1 | 1 | 144,008 | 144,008 | 144,008 |

Approximate projected total: 215,486,854 characters.
Text-level duplicate signals among metadata overlaps: 5/6.

## Cross-Archive Duplicate Evidence

| Gutenberg ID | Existing match | 8-gram containment | Duplicate signal |
| ---: | --- | ---: | --- |
| 31079 | `bibit:bibit000014` | 0.8731 | yes |
| 31080 | `bibit:bibit000681` | 0.8367 | yes |
| 38012 | `bibit:bibit001054` | 0.8280 | yes |
| 57787 | `bibit:bibit000049` | 0.9772 | yes |
| 76738 | `bibit:bibit001479` | 0.9524 | yes |
| 77167 | `bibit:bibit001436` | 0.0462 | no |

## Boundaries

- The projection is a planning estimate, not an activated corpus size.
- Sample text remains machine-local and is not committed.
- Full-text editorial review and cross-corpus deduplication remain required.
- No V7 split or training-mixture weight is assigned.

## Next Gate

If sample availability and quality justify expansion, probe all eligible records. Estimated runtime: 10m-90m network-dependent.
