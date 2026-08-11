# Project Gutenberg Resolved Corpus Build

## Result

Materialized 572 of 587 audited source records and 501 of 611 sonnet candidates.

Every artifact is stored once in bounded UTF-8 shards. The manifests retain
exact byte ranges and SHA-256 values for independent recovery and verification.

## Record Roles

| Role | Retained source characters | Materialized shard characters |
| --- | ---: | ---: |
| `conditioned_source_variants` | 2,291,790 | 2,291,790 |
| `historical_general` | 52,431,696 | 52,431,698 |
| `historical_non_sonnet_poetry` | 11,494,505 | 11,495,264 |
| `nineteenth_century_bridge` | 228,427,424 | 228,427,431 |

## Sonnet Artifacts

| Status | Candidates |
| --- | ---: |
| `conditioned_sonnet_materialized_inactive` | 2 |
| `not_materialized_exclude_existing_corpus_sonnet_duplicate` | 3 |
| `not_materialized_exclude_intra_gutenberg_sonnet_duplicate` | 1 |
| `not_materialized_exclude_manual_not_sonnet` | 106 |
| `standard_sonnet_materialized_pending_v7` | 499 |

## Deduplication

- Standard broader-text records checked: 566.
- Cross-corpus references checked: 1,352.
- Final normalized exact duplicate groups: 0.
- Final internal near-duplicate pairs at the frozen threshold: 0.
- Final cross-corpus near-duplicate pairs at the frozen threshold: 0.
- Protected V6 validation/test evidence remains excluded through the hash-pinned 3A reconstruction.

## Boundaries

- Standard sonnets are absent from historical-general, non-sonnet-poetry, and bridge shards.
- Conditioned source and sonnet shards are materialized separately and remain inactive.
- The full Ottocento candidate pool is materialized; no 10% exposure cap is applied yet.
- Candidate-level poem authors are not guessed from source-record metadata.
- No V7 split, training-mixture weight, cache deletion, or GPU work occurs in this build.

## Artifacts

- Record manifest: `data/processed/project_gutenberg_resolved_v1/records_manifest.csv`
- Segment manifest: `data/processed/project_gutenberg_resolved_v1/segments_manifest.csv`
- Sonnet manifest: `data/processed/project_gutenberg_resolved_v1/sonnets_manifest.csv`
- Attribution manifest: `data/processed/project_gutenberg_resolved_v1/attribution_manifest.csv`
- Machine-readable report: `data/processed/project_gutenberg_resolved_v1/build_report.json`
