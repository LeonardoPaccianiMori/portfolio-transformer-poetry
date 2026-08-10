# Project Gutenberg Authoritative Metadata Resolution Pass 1B

## Result

Audited all 364 pass-1B holds and reconciled 673 frozen Gutenberg records.

## Final Decisions

| Decision | Records |
| --- | ---: |
| `eligible_historical_core_candidate` | 14 |
| `eligible_nineteenth_century_candidate` | 401 |
| `exclude_post_1900_original_text` | 88 |
| `exclude_unresolved_authoritative_metadata` | 166 |
| `route_conditioned_bolognese_prose_and_drama` | 1 |
| `route_conditioned_romanesco_sonnets` | 1 |
| `route_mixed_language_segment_review` | 2 |

## Activation Classes

| Class | Records |
| --- | ---: |
| `conditioned_probe` | 4 |
| `eligible_probe` | 415 |
| `excluded` | 254 |

## Pass 1B Exclusion Reasons

| Reason | Records |
| --- | ---: |
| `authoritative_date_after_1900` | 27 |
| `bolognese_dialect_collection_is_outside_standard_italian_core` | 1 |
| `italian_translation_edition_date_not_authoritatively_resolved` | 1 |
| `mixed_standard_italian_and_neapolitan_segments_require_extraction` | 2 |
| `no_direct_sbn_title_author_match` | 11 |
| `post_1900_or_undated_sbn_edition_does_not_prove_work_date` | 154 |
| `romanesco_sonnets_are_outside_the_standard_italian_core` | 1 |

## Evidence Policy

- SBN/ICCU title-and-author matches are the primary bibliographic identity evidence.
- A matched SBN edition dated by 1900 proves only that the work existed by that date; a later edition does not prove a later first publication.
- Wikidata P577 is accepted only for an exact title match whose description also identifies the expected author, and only after SBN has anchored the work identity.
- Conflicting dates and unresolved identity/date evidence are documented exclusions, never guessed inclusions.
- Record-specific primary-text resolutions are limited to explicit title-page, edition, composition, delivery, performance, or anthology-content dates.
- Translation decisions use the Italian edition date, not the source work's original date.
- Romanesco, Bolognese, and mixed-language records remain outside the standard-Italian core and are routed separately.
- Open Library is not used as deciding evidence.
- This checkpoint authorizes no download, corpus activation, V7 split, or training weight.

## Artifacts

- Final 673-row resolution: `data/metadata/project_gutenberg_metadata_final_resolution_v1.csv`
- Explicit unresolved/exclusion rows: `data/metadata/project_gutenberg_metadata_final_exclusions_v1.csv`
- Machine report: `reports/project_gutenberg_authoritative_resolution_v1.json`
