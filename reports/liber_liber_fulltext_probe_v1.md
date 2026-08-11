# Liber Liber Bounded Full-Text Probe

## Result

Checkpoint 5B probes exactly 129 checkpoint-5A eligible records.

- Successful/failed status: `{'quality_pass': 119, 'review': 10}`.
- Cleaned volume: 40,884,715 characters and 6,914,277 words.
- Archive formats: `{'odt': 122, 'txt_zip': 7}`.
- Cross-corpus references indexed: 26,770.
- Cross-corpus threshold pairs: 70.
- Fully covered candidates: 30; embedded-reference-only candidates: 5.
- Internal near-duplicate pairs: 3.
- Protected V6 validation/test sonnets: 387; overlapping candidates: 5.
- Automated anomalies reviewed: 10; unresolved: 0.
- Conditioned language-variety records excluded from this queue: 151.

## Decisions

| Decision | Records |
| --- | ---: |
| `exclude_cross_corpus_duplicate_candidate` | 30 |
| `quality_pass_after_bounded_review` | 8 |
| `quality_pass_pending_extraction_audit` | 83 |
| `quarantine_embedded_duplicate_segments_before_activation` | 3 |
| `quarantine_protected_v6_segment_before_activation` | 5 |

## Reference Coverage

| Reference kind | Records |
| --- | ---: |
| `bibit` | 1,316 |
| `bibit_sonnet` | 20,331 |
| `existing_project_corpus` | 36 |
| `gutenberg_pass_1b` | 167 |
| `gutenberg_previous_pool` | 416 |
| `gutenberg_resolved` | 566 |
| `gutenberg_sonnet` | 499 |
| `wikisource_resolved` | 2,452 |
| `wikisource_sonnet` | 987 |

## Boundary

This probe records acquisition, quality, overlap, and protected-set evidence only. It activates no text, creates no V7 split, assigns no mixture weight, deletes no cache, and starts no GPU work.
