# Italian Wikisource Archive Inventory And Composition Gate

## Result

The pinned `20260801` metadata snapshot contains 117,297 main-namespace pages grouped into 22,165 structural work roots.

Metadata identifies 6,863 historical or nineteenth-century candidates before page-level review, projecting 64,834,722 wikitext bytes (28.7% of the archive projection). No corpus text was downloaded, extracted, or activated.

## Decisions

| Decision | Work roots |
| --- | ---: |
| `conditioned_language_candidate` | 382 |
| `exclude_explicit_non_italian` | 35 |
| `exclude_post_1900_scope` | 2,491 |
| `existing_project_reference` | 176 |
| `historical_core_metadata_candidate` | 3,775 |
| `hold_language_variety_review` | 3,883 |
| `hold_period_or_work_identity` | 7,003 |
| `hold_translation_edition_review` | 1,332 |
| `nineteenth_century_bridge_metadata_candidate` | 3,088 |

## Projected Roles

| Role | Work roots | Wikitext-byte projection |
| --- | ---: | ---: |
| `conditioned_language_variant` | 382 | 1,847,821 |
| `cross_archive_reference_only` | 176 | 2,951,080 |
| `excluded` | 2,526 | 100,763,991 |
| `historical_general` | 1,788 | 24,015,936 |
| `historical_non_sonnet_poetry` | 1,987 | 7,063,060 |
| `metadata_hold` | 12,218 | 55,287,562 |
| `nineteenth_century_bridge` | 1,599 | 32,835,373 |
| `standard_sonnets` | 1,489 | 920,353 |

## Candidate Concentration

| Author proxy | Wikitext-byte projection | Candidate share |
| --- | ---: | ---: |
| Emilio Salgari | 8,755,725 | 13.5% |
| Niccolò Machiavelli | 3,045,994 | 4.7% |
| Giovanni Villani | 2,812,924 | 4.3% |
| Giacomo Leopardi | 2,178,263 | 3.4% |
| Giorgio Vasari | 1,785,596 | 2.8% |
| Alessandro Manzoni | 1,550,839 | 2.4% |
| Giovanni Boccaccio | 1,498,071 | 2.3% |
| Luigi Pulci | 1,449,899 | 2.2% |
| Dante Alighieri | 1,309,471 | 2.0% |
| Emma Perodi | 1,246,235 | 1.9% |

These are metadata projections rather than cleaned-text or token shares. Unknown and multi-author rows prevent a final concentration claim; later activation must recompute and cap dominance.

## Bounded Inspection

The stratified sample rendered 30 exact dump-pinned revisions.
- Primary-text signal passes: 17.
- Rows requiring page-level review: 13.
- These signals validate inventory value; they do not authorize extraction.

## Rights And Boundaries

- Current site transcription terms: Creative Commons Attribution-Share Alike 4.0 (https://creativecommons.org/licenses/by-sa/4.0/deed.it).
- Underlying work and source-scan status still require record-level verification.
- Explicit dialect/language varieties remain conditioned or held, never standard core.
- Wikitext bytes are a metadata projection, not cleaned characters or Minerva tokens.
- The full page-text dump was not downloaded.
- No V7 split, mixture weight, cache deletion, or GPU work occurred.

## Artifacts

- Work-root inventory: `data/metadata/italian_wikisource_archive_inventory_v1.csv`
- Page hierarchy: `data/metadata/italian_wikisource_page_hierarchy_v1.csv`
- Composition gate: `data/metadata/italian_wikisource_composition_gate_v1.csv`
- Inspection sample: `data/metadata/italian_wikisource_inspection_sample_v1.csv`
- Machine-readable report: `reports/italian_wikisource_archive_inventory_v1.json`
