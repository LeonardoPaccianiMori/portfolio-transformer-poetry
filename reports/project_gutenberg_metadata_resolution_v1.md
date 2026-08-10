# Project Gutenberg Metadata Resolution Pass 1A

## Result

Acquired and triaged 673 frozen review records. Automatically resolved 309; retained 364 for cited review.

## Resolution Status

| Status | Records |
| --- | ---: |
| `automatic_resolved` | 309 |
| `manual_review` | 364 |

## Automatic Decisions

| Decision | Records |
| --- | ---: |
| `eligible_historical_core_candidate` | 3 |
| `eligible_nineteenth_century_candidate` | 245 |
| `exclude_post_1900_original_text` | 61 |

## Manual Review Reasons

| Reason | Records |
| --- | ---: |
| `edition_proves_pre_1901_eligibility_but_work_period_is_ambiguous` | 39 |
| `no_direct_period_evidence` | 150 |
| `post_1900_edition_does_not_prove_work_first_publication` | 164 |
| `post_1900_translation_edition_requires_authoritative_review` | 1 |
| `primary_text_language_variety_review_required` | 6 |
| `translation_edition_date_not_found` | 4 |

## Evidence Boundaries

- Gutenberg release and update dates are recorded but never used as work-period evidence.
- Original-publication metadata, explicit first-Italian-version evidence, and qualified title work periods may resolve a row automatically.
- Generic first-edition mentions are recorded but remain non-decisive because front matter can describe another work or language edition.
- A title-page edition year can prove that text existed by 1900, but it cannot silently backdate an ambiguous work.
- Translation routing uses evidence for the Italian version or edition, not the source work's age.
- Language-variety candidates remain manual and outside the standard core.
- This pass activates no text, assigns no V7 split, and freezes no training weight.

## Artifacts

- Complete evidence: `data/metadata/project_gutenberg_metadata_resolution_v1.csv`
- Manual queue: `data/metadata/project_gutenberg_metadata_manual_review_v1.csv`
- Machine report: `reports/project_gutenberg_metadata_resolution_v1.json`
