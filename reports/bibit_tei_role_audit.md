# Biblioteca Italiana Role-Specific TEI Audit

## Decision

Audited 1,373 canonical TEI records and 24,338 sonnet candidates. Status: `audit_complete_manual_review_required`.

This is an activation gate, not a blind concatenation. Explicit sonnets and
unverified 14-line verse units are absent from every earlier-stage text route.
V6 validation/test identities are checked before any candidate can enter training.

## Record Outcomes

| Status | Records |
| --- | ---: |
| `activation_candidate` | 1,204 |
| `error` | 1 |
| `review_required` | 168 |

## Routed Corpus

| Route | Records | Automatic candidates | Candidate characters | Review/error |
| --- | ---: | ---: | ---: | ---: |
| `historical_general` | 638 | 562 | 74,532,753 | 76 |
| `historical_non_sonnet_poetry` | 412 | 340 | 37,953,293 | 72 |
| `nineteenth_century_bridge` | 322 | 302 | 41,036,542 | 20 |

## Sonnet Candidates

- Explicit TEI sonnets: 18,742.
- Unverified structural 14-line candidates: 5,596.
- Held-out identity conflicts: 411.

| Status | Candidates |
| --- | ---: |
| `eligible_explicit_nonduplicate` | 13,037 |
| `excluded_exact_active_duplicate` | 24 |
| `excluded_exact_bibit_duplicate` | 169 |
| `excluded_held_out_identity_conflict` | 411 |
| `excluded_not_14_lines` | 1,105 |
| `review_candidate_editorial_markers` | 120 |
| `review_missing_author_attribution` | 3,592 |
| `review_near_active_duplicate` | 1,625 |
| `review_source_blocked_post_dedup` | 293 |
| `review_structural_form` | 3,962 |

## Review Queue

| Flag | Records |
| --- | ---: |
| `duplicate_exact_bibit_document` | 16 |
| `empty_or_too_short_after_sonnet_quarantine` | 3 |
| `fetch_or_parse_error` | 1 |
| `note_multilingual_metadata` | 575 |
| `review_editorial_brackets` | 88 |
| `review_editorial_references` | 1 |
| `review_language_variety` | 10 |
| `review_missing_source_edition` | 15 |
| `review_near_duplicate` | 6 |
| `review_no_sonnet_candidates` | 27 |
| `review_non_italian_language` | 7 |

## Evidence

- Per-record decisions: `data/metadata/bibit_tei_audit_records.csv`
- Per-sonnet decisions: `data/metadata/bibit_sonnet_candidates_audit.csv`
- Machine-readable summary: `reports/bibit_tei_role_audit.json`
- Raw TEI is cached only under ignored `data/local/`; every public record retains its TEI SHA-256.
