# Italian Wikisource Page Extraction Audit

## Result

Checkpoint 4C accounts for 4,641 scan-anchored roots without activating their text.

- Selected hierarchy leaves: 36,287.
- Required `Pagina:` transcriptions: 238,802.
- Matched `Pagina:` transcriptions: 229,322.
- Local extracted characters: 325,751,198.
- Passing inactive characters: 31,000,426.
- Review rows: 2,095.

## Decisions

| Decision | Roots |
| --- | ---: |
| `eligible_inactive_pending_processed_build` | 2,546 |
| `hold_duplicate_review` | 1,015 |
| `hold_empty_extraction` | 10 |
| `hold_missing_transcription` | 114 |
| `hold_protected_v6_overlap` | 27 |
| `hold_quality_or_editorial_review` | 410 |
| `hold_rendered_validation` | 1 |
| `hold_revision_mismatch` | 1 |
| `hold_unresolved_markup` | 517 |

## Duplicate And Leakage Probe

- Internal normalized exact-duplicate groups: 337.
- Roots in internal exact-duplicate groups: 821.
- Internal threshold pairs: 189.
- Cross-corpus threshold pairs: 312.
- Protected V6 overlap roots: 28.

The comparisons cover BibIt, both Gutenberg probe pools, the resolved Gutenberg build, existing project corpora, and protected V6 validation/test sonnets.

## Boundary

All extracted text remains in ignored local interim storage. No Wikisource text is activated, no V7 split or mixture is created, no cache is deleted, and no GPU work occurs in this checkpoint.

## Rendered Validation

Sampled pages: 88.

| Status | Pages |
| --- | ---: |
| `hold_low_containment` | 7 |
| `pass` | 81 |
