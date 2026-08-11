# Remaining Archive Registry Resolution, v1

Audit date: `2026-08-11`

This metadata-only checkpoint closes the twelve unresolved registry rows. `eligible_bounded_inventory_inactive` permits only the stated metadata/source inventory; it does not authorize corpus-text acquisition, activation, V7 splitting, mixture weighting, cache deletion, or GPU work.

## Decisions

| Archive | Role | Final status | Measured scope | Closure |
| --- | --- | --- | --- | --- |
| Biblioteca Italiana - Scrittori d'Italia | `excluded` | `closed_excluded_scan_only_overlap_risk` | 179 works; 287 volumes; 125,171 image-text pages | No further extraction; retain as edition/source-scan reference. |
| Biblioteca Italiana - Incunaboli in volgare | `auxiliary_ocr_candidate` | `blocked_ocr_and_item_rights_gate` | More than 1,600 incunabula; about 70 libraries; more than 200,000 images | Do not audit pages now; require a later approved OCR-quality and item-rights experiment. |
| ELTeC Italian novel collection | `auxiliary_capped_ottocento_bridge` | `eligible_bounded_inventory_inactive` | 36 level-1 novels | Checkpoint 6B may inventory the 36 records without activating or downloading corpus text through this decision alone. |
| Internet Archive | `core_training_candidate` | `eligible_bounded_inventory_inactive` | 99,424 broad language/date metadata hits in the pinned 2026-08-11 query | Checkpoint 6B may run a bounded metadata inventory only, prioritizing explicit reusable rights and corrected OCR. |
| Gallica / Bibliothèque nationale de France | `core_training_candidate` | `eligible_bounded_inventory_inactive` | SRU inventory supports language/type queries and 0-100 OCR-quality metadata; candidate count not retrievable during this audit | Checkpoint 6B may inventory metadata only; no OCR/full text until content and item rights pass. |
| Internet Culturale | `core_training_candidate` | `eligible_bounded_inventory_inactive` | 291 digital collections in the official directory | Checkpoint 6B may inventory text-bearing collections and item terms without acquiring corpus text. |
| BEIC Digital Library | `core_training_candidate` | `eligible_bounded_inventory_inactive` | 39,821 digital resources; 98,327 bibliographic records; 5,617 authors | Checkpoint 6B may inventory Italian literary records and text formats without acquiring corpus text. |
| HathiTrust Digital Library | `core_training_candidate` | `blocked_official_terms_and_bulk_access` | Large Italian holdings stated by registry; no auditable count pinned | Keep blocked; obtain accessible official terms or direct research-corpus permission before any inventory/acquisition. |
| Google Books | `excluded` | `discovery_only_closed` | Large catalog; no corpus-eligible record count claimed | Use only to locate the same edition in a reusable archive. |
| OVI / TLIO | `core_training_candidate` | `blocked_bulk_and_training_permission` | Official query interface confirmed; bulk scope not published | Request or locate explicit official bulk/research-training permission and a source list. |
| MIDIA historical Italian corpus | `core_training_candidate` | `eligible_bounded_inventory_inactive` | About 800 texts and 7.8 million occurrences from the 13th to mid-20th century | Checkpoint 6B may inventory the source list, periods, and access routes; this status alone does not authorize corpus-text download. |
| DiaCORIS historical corpus | `auxiliary_capped_ottocento_bridge` | `blocked_bulk_and_training_permission` | Five time slices from 1861-2001; six subcorpus genres; no record/token count pinned | Seek explicit permission and isolate the 1861-1900 source list before any acquisition. |

## Accounting

- Frozen registry rows: 12
- Official evidence rows: 18
- Eligible bounded inventories: 6
- Concretely blocked rows: 4
- Closed exclusions/discovery-only rows: 2
- Existing inactive broader-pool subtotal: 626,379,622 characters
- New corpus characters activated: 0

## Fail-closed constraints

- First-party evidence is required for permission decisions.
- Item/content terms override portal-level defaults where stated.
- OCR-only archives need a separately approved measured quality gate.
- HathiTrust, TLIO, and DiaCORIS remain blocked; Google Books remains discovery-only.
- The open-ended final discovery pass remains checkpoint 6C.
