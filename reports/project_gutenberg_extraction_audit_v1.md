# Project Gutenberg Extraction And Canonicalization Audit

## Result

Accounted for 587 cached Project Gutenberg records without materializing processed corpus text.

## Source Decisions

| Decision | Records |
| --- | ---: |
| `conditioned_candidate_not_activated` | 6 |
| `eligible_after_source_specific_extraction_pending_build` | 2 |
| `eligible_standard_core_pending_processed_build` | 564 |
| `exclude_canonical_cross_corpus_duplicate` | 15 |

## Measured Roles

| Role | Records | Retained characters |
| --- | ---: | ---: |
| `historical_general` | 73 | 52,431,696 |
| `historical_non_sonnet_poetry` | 44 | 11,494,505 |
| `nineteenth_century_bridge` | 449 | 228,427,424 |

## Sonnets

- Audited candidates: 611.
- Eligible standard candidates pending the processed build: 499.
- Unresolved structural reviews: 0.
- Held-out conflicts excluded: 0.

### Candidate Decisions

| Decision | Candidates |
| --- | ---: |
| `conditioned_sonnet_candidate_not_activated` | 2 |
| `eligible_standard_sonnet_pending_processed_build` | 499 |
| `exclude_existing_corpus_sonnet_duplicate` | 3 |
| `exclude_intra_gutenberg_sonnet_duplicate` | 1 |
| `exclude_manual_not_sonnet` | 106 |

### Structural Review

| Resolution | Candidates |
| --- | ---: |
| `accept_structurally_verified_standard_sonnet` | 104 |
| `exclude_nonstandard_language_sonnet` | 2 |
| `exclude_not_sonnet` | 106 |

False-positive fourteen-line windows remain in their broader-text role; only confirmed or unresolved sonnet units are quarantined from broader stages.

## Boundaries

- The fifteen fully covered Gutenberg editions remain excluded in favor of their existing canonical references.
- Unique material is retained from the six partial-overlap sources.
- The Cino validation sonnet is quarantined from eBook 35321.
- Six conditioned source records and two embedded non-standard-language sonnets remain outside the standard corpus.
- No processed text, V7 split, training-mixture weight, or GPU job is created by this audit.

## Reproduction

Run `python3 scripts/audit_project_gutenberg_extraction.py` with both preserved local Gutenberg caches. The JSON report pins every public input and tabular output with SHA-256.
