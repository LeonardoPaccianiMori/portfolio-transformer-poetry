# Liber Liber Archive Inventory And Composition Gate

## Result

Checkpoint 5A inventories all 9,332 public WordPress pages and identifies 6,928 book work records without acquiring full text.

- Records advertising TXT ZIP or ODT: 5,011.
- WordPress pages requiring basic-metadata fallback: 1.
- Existing exact project source URLs: 26.
- Metadata-gated inactive full-text probes: 129.
- Planning projection: 91,565,232 cleaned characters (14.2% of the resulting frozen-archive pool).
- Full-text probe runtime estimate: 7-26 minutes before manual review and overlap analysis.

The character estimate uses the median of the earlier 23-record Liber Liber probe. It is not downloaded text, a token count, or an activation decision.

## Decisions

| Decision | Records |
| --- | ---: |
| `conditioned_language_candidate_inactive` | 151 |
| `eligible_fulltext_probe_inactive` | 129 |
| `exclude_post_1900_metadata` | 281 |
| `existing_project_corpus_reference` | 26 |
| `hold_drama_prose_verse_review` | 16 |
| `hold_no_supported_primary_text_format` | 15 |
| `hold_rights_or_item_license_unclear` | 1,902 |
| `hold_translation_edition_review` | 709 |
| `hold_work_language_review` | 3,679 |
| `hold_work_period_review` | 20 |

## Preliminary Roles

| Role | Records |
| --- | ---: |
| `conditioned_language_variants` | 159 |
| `historical_general` | 7 |
| `historical_non_sonnet_poetry` | 219 |
| `nineteenth_century_bridge` | 134 |
| `review_unassigned` | 6,389 |
| `standard_sonnets` | 20 |

## Rights Boundary

A work enters the inactive probe queue only when the item is marked free, pins the approved CC BY-NC-SA 4.0 edition license, advertises TXT ZIP or ODT, and has decisive standard-Italian historical/Ottocento metadata. Protected personal-use-only texts and unclear item licenses fail closed.

## Checkpoint Boundary

No archive full text was acquired. No source is activated, and no V7 split, mixture weight, cache deletion, or GPU work is authorized. Translation, dialect, drama-form, period, format, and rights holds require later bounded resolution.
