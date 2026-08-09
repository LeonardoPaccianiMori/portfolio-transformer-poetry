# Project Gutenberg Metadata Review Queue

## Result

Frozen 673 unresolved metadata-review records from the 1,112-record Italian catalog inventory. This artifact activates no text.

## Review Statuses

| Status | Records | Evidence required |
| --- | ---: | --- |
| `review_language_variety_before_download` | 6 | primary-text evidence for standard Italian, dialect, or mixed-language routing |
| `review_missing_period_evidence` | 79 | author/work dates from a title page, catalog record, or authoritative bibliography |
| `review_translation_edition_date` | 25 | Italian translation edition date and source-language/translator evidence |
| `review_work_publication_date` | 563 | work first-publication year from a title page, catalog record, or authoritative bibliography |

## Complete Inventory Accounting

| Route | Records |
| --- | ---: |
| `eligible_fulltext_probe` | 416 |
| `metadata_review_queue` | 673 |
| `already_registered` | 10 |
| `metadata_exclusions` | 13 |
| **Total** | **1,112** |

## Count Clarification

The earlier 613 figure counts records whose preliminary *role* is `date_and_role_review`. It is not the unresolved queue size. The 673-record queue also includes poetry, translation, sonnet, and language-variety candidates whose status still requires evidence.

## Boundaries

- Resolve each queued record before deciding whether to download it.
- A resolved record still requires full-text quality and deduplication gates.
- Dialect and mixed-language records remain outside the unconditioned core.
- No V7 split or training-mixture weight is assigned here.

## Artifacts

- Frozen input inventory: `data/metadata/project_gutenberg_italian_inventory_v1.csv`
- Review queue: `data/metadata/project_gutenberg_metadata_review_queue_v1.csv`
- Machine-readable report: `reports/project_gutenberg_metadata_review_queue_v1.json`
