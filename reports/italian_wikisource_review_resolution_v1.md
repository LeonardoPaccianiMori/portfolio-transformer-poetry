# Italian Wikisource Review Resolution

## Result

Checkpoint 4D resolves all 2,095 checkpoint-4C review rows and accounts for all 4,641 extracted roots.

- Source scans with rights evidence: 1,282.
- Rights-passing scans: 1,175.
- Roots eligible for an inactive processed build: 2,674.
- Retained broader-text characters: 30,749,328.
- Verified sonnet candidates: 987.

## Final Root Decisions

| Decision | Roots |
| --- | ---: |
| `eligible_inactive_processed_build` | 2,452 |
| `eligible_sonnets_only_inactive` | 222 |
| `exclude_canonical_cross_corpus_duplicate` | 61 |
| `exclude_empty_extraction` | 9 |
| `exclude_incomplete_transcription` | 88 |
| `exclude_internal_exact_duplicate` | 449 |
| `exclude_internal_near_duplicate` | 50 |
| `exclude_no_unique_material_after_segmentation` | 68 |
| `exclude_post_segmentation_duplicate` | 2 |
| `exclude_quality_language_or_editorial_hold` | 453 |
| `exclude_rendered_boundary_failure` | 1 |
| `exclude_revision_mismatch` | 1 |
| `exclude_source_rights_unresolved` | 310 |
| `exclude_unresolved_markup` | 475 |

## Boundary

These decisions remain inactive. They create no V7 split, training mixture, GPU job, or cache deletion. Conditioned and checkpoint-4B-held material never enters the standard-Italian queue.
