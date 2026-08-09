# Biblioteca Italiana Review Resolution

## Decision

Resolved 1,373 canonical records and 24,538 poem candidates with no open rows.

Dialect, vernacular, and Franco-Italian records remain documented evidence
but are excluded from the unconditioned standard-Italian core. Explicit
non-14-line sonnets are retained only in a conditioned variant role.

## Record Decisions

| Decision | Records |
| --- | ---: |
| `activate_as_non_sonnet_poetry` | 25 |
| `activate_as_non_sonnet_poetry_with_bracket_cleanup` | 1 |
| `activate_core` | 1,205 |
| `activate_core_with_bracket_cleanup` | 86 |
| `exclude_core_language_variety` | 14 |
| `exclude_duplicate_document` | 16 |
| `exclude_empty_or_too_short` | 3 |
| `exclude_empty_or_unavailable` | 1 |
| `exclude_missing_source_provenance` | 15 |
| `exclude_non_italian` | 7 |

## Activated Text

| Role | Characters |
| --- | ---: |
| `historical_general` | 126,721,674 |
| `historical_non_sonnet_poetry` | 41,408,267 |
| `nineteenth_century_bridge` | 60,722,622 |

## Poem Decisions

| Decision | Candidates |
| --- | ---: |
| `activate_explicit_sonnet_variant` | 961 |
| `activate_heading_backed_sonnet_variant` | 99 |
| `activate_inferred_standard_sonnet` | 3,063 |
| `activate_standard_explicit_sonnet` | 16,208 |
| `exclude_exact_duplicate` | 202 |
| `exclude_held_out_identity` | 411 |
| `exclude_near_active_duplicate` | 1,629 |
| `exclude_unhandled_form` | 9 |
| `exclude_unverified_structural_14_line_unit` | 1,432 |
| `exclude_with_source_record` | 405 |
| `excluded_implausible_sonnet_length` | 119 |

## Activated Poem Roles

| Role | Poems | Characters |
| --- | ---: | ---: |
| `sonnet_core_inferred_14_line` | 3,063 | 1,619,696 |
| `sonnet_core_standard_14_line` | 16,208 | 8,750,919 |
| `sonnet_variant_conditioned_auxiliary` | 1,060 | 670,175 |

## Evidence

- Record decisions: `data/metadata/bibit_record_activation_decisions.csv`
- Poem decisions: `data/metadata/bibit_sonnet_activation_decisions.csv`
- Machine-readable report: `reports/bibit_review_resolution.json`
- Raw TEI remains machine-local; decisions retain source hashes and URLs.
