# Italian Wikisource Candidate And Source-Scan Resolution

## Result

Checkpoint 4B resolves all 6,863 provisional checkpoint-4A candidates against the pinned `20260801` metadata link graph.

Direct `Indice:` links exist for 6,092 candidates across 1,335 source scans. The metadata-only page-level audit queue contains 4,641 candidates projecting 16,353,125 wikitext bytes.

## Decisions

| Decision | Work roots | Projected wikitext bytes |
| --- | ---: | ---: |
| `eligible_page_level_audit_queue` | 4,641 | 16,353,125 |
| `hold_multiple_source_scans` | 3 | 195,000 |
| `hold_no_direct_scan_link` | 771 | 46,976,186 |
| `hold_redirected_index_page` | 1 | 614 |
| `hold_scan_language_conflict` | 1,447 | 1,309,797 |

## Eligible Role Projection

| Role | Work roots | Projected wikitext bytes |
| --- | ---: | ---: |
| `historical_general` | 1,413 | 4,274,599 |
| `historical_non_sonnet_poetry` | 1,724 | 2,766,081 |
| `nineteenth_century_bridge` | 1,245 | 9,130,749 |
| `standard_sonnets` | 259 | 181,696 |

## Language-Evidence Correction

Checkpoint 4A contains 3,883 language-review rows. Of these, 2,857 contain only citation-index labels and/or explicit standard Italian; they are not treated as dialect evidence in scan-group propagation.
The remaining 1,026 rows retain genuine nonstandard or unresolved language evidence. No held row is promoted into the candidate queue by this correction.

## Source-Scan Boundaries

The 6,092 linked candidates map to 1,335 scans; 242 scans support more than one candidate root. Scan grouping is retained for later extraction and deduplication rather than flattening anthology contents.
- Bounded review units: 823.
- Wikitext bytes remain projections, not cleaned characters or tokens.
- Current site terms remain CC BY-SA 4.0; scan/work rights still require final verification.

## Boundaries

- The full page-text dump was not downloaded.
- No primary text was extracted or activated.
- Conditioned or unresolved language material was not admitted to standard core.
- No V7 split, mixture weight, cache deletion, or GPU work occurred.

## Artifacts

- Candidate resolution: `data/metadata/italian_wikisource_candidate_resolution_v1.csv`
- Source-scan links: `data/metadata/italian_wikisource_source_scan_links_v1.csv`
- Bounded review ledger: `data/metadata/italian_wikisource_candidate_review_v1.csv`
- Machine-readable report: `reports/italian_wikisource_candidate_resolution_v1.json`
