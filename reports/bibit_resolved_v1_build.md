# Biblioteca Italiana Resolved Corpus Build

## Result

Built text for 1,316 of 1,317 activated source records and 20,331 activated sonnets from the pinned TEI cache.

The text is stored once in bounded UTF-8 shards. The manifests record exact
byte ranges and SHA-256 hashes, so every source or poem can be recovered and
verified independently without creating tens of thousands of tiny files.

## Record Roles

| Role | Text records | Characters |
| --- | ---: | ---: |
| `historical_general` | 609 | 126,643,238 |
| `historical_non_sonnet_poetry` | 385 | 41,401,123 |
| `nineteenth_century_bridge` | 322 | 60,709,570 |

## Sonnet Roles

| Role | Sonnets | Characters |
| --- | ---: | ---: |
| `sonnet_core_inferred_14_line` | 3,063 | 1,619,327 |
| `sonnet_core_standard_14_line` | 16,208 | 8,750,705 |
| `sonnet_variant_conditioned_auxiliary` | 1,060 | 670,075 |

## Boundaries

- No V7 train/validation/test assignment is made by this build.
- No final training-mixture weight is authorized by this build.
- All audited sonnet candidates remain quarantined from record text.
- V6 validation/test identity conflicts remain excluded.
- 1 activated sonnet-source records have no residual record text after poem quarantine.

## Artifacts

- Record manifest: `data/processed/bibit_resolved_v1/records_manifest.csv`
- Sonnet manifest: `data/processed/bibit_resolved_v1/sonnets_manifest.csv`
- Machine-readable report: `data/processed/bibit_resolved_v1/build_report.json`
