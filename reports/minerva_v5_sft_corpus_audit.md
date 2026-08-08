# Minerva V5 SFT Corpus Audit

## Scope

- Dataset: `expanded_with_petrarch`
- Manifest: `data/metadata/sonnets_expanded_v5_manifest.csv`
- Manifest SHA-256: `d71abe5bbc048392b7579702124f25bd6dedb400a47a0c171e3f4e6e0aae6275`
- Selected poems: 1,875
- Split counts: train 1,486 / validation 191 / test 198

## Automated Structural Gate

- Result: **review_required**
- Structural issues: 1
- Exact normalized duplicate groups: 6
- Cross-split duplicate groups: 4
- Suspicious markers: `{}`

## Composition

### Periods

| Period | Poems |
| --- | ---: |
| XVI secolo | 701 |
| XIII secolo | 491 |
| XIV secolo | 430 |
| XVII secolo | 196 |
| XVIII secolo | 45 |
| XIX secolo | 12 |

### Largest Training Authors

| Author | Poems | Share |
| --- | ---: | ---: |
| Vittoria Colonna | 268 | 18.0% |
| Francesco Petrarca | 253 | 17.0% |
| Gaspara Stampa | 217 | 14.6% |
| Guittone d'Arezzo | 164 | 11.0% |
| Isabella Andreini | 151 | 10.2% |
| Cecco Angiolieri | 102 | 6.9% |
| Dante Alighieri | 66 | 4.4% |
| Cino da Pistoia | 65 | 4.4% |
| Guido Cavalcanti | 36 | 2.4% |
| Vittorio Alfieri | 36 | 2.4% |

### Largest Training Collections

| Collection | Poems | Share |
| --- | ---: | ---: |
| Rime (Vittoria Colonna) | 268 | 18.0% |
| Canzoniere (Rerum vulgarium fragmenta) | 253 | 17.0% |
| Rime (Stampa) | 217 | 14.6% |
| Rime (Guittone d'Arezzo) | 164 | 11.0% |
| Rime (Andreini) | 151 | 10.2% |
| Rime (Angiolieri) | 102 | 6.9% |
| Rime (Dante) | 66 | 4.4% |
| Rime (Cino da Pistoia) | 65 | 4.4% |
| Rime (Cavalcanti) | 36 | 2.4% |
| Rime varie (Alfieri, 1912) | 36 | 2.4% |

## Cleaning And Line Diagnostics

- Line length: minimum 1, mean 36.9, maximum 53 characters.
- Lines over 120 characters: 0.
- Lines under 4 characters: 2.
- Poems with recorded editorial-bracket removal: 1,875.
- Poems with recorded line-marker removal: 1,875.
- Poems with cleaning notes: 1,875.
- Poems with audit notes: 990.

## Interpretation

Automated checks cannot certify historical grammar; review the deterministic sample before freezing another Minerva recipe.
The companion review sample is not training data duplication; it is a deterministic view of committed V5 texts for editorial inspection.

## Structural Issues

| Poem | Author | Type | Detail |
| --- | --- | --- | --- |
| cavalcanti_la_genealogia_dei_manoscritti | Guido Cavalcanti | short_line | contains a line under 4 characters |

## Exact Duplicate Groups

| Poem IDs | Splits | Cross-split leakage |
| --- | --- | --- |
| cino_rime_dantecx_dante_quando_per_caso_s_abbandona; dante_cx_dante_quando_per_caso_s_abbandona | train; validation | yes |
| cino_rime_dantecxii_cercando_di_trovar_minera_in_oro; dante_cxii_cercando_di_trovar_minera_in_oro | train; test | yes |
| dante_xcvii_dante_i_non_so_in_qual_albergo_soni; cino_rime_dantexcvii_dante_i_non_so_in_qual_albergo_soni | train; validation | yes |
| dante_xcviii_dante_i_ho_preso_l_abito_di_doglia; cino_rime_dantexcviii_dante_i_ho_preso_l_abito_di_doglia | train; train | no |
| guittone_ben_si_conosce_lo_servente_e_vede; guittone_non_per_meo_fallo_lasso_mi_convene | train; train | no |
| guittone_de_vertù_de_scienzia_il_cui_podere; guittone_tu_costante_e_sicuro_fondamento | train; test | yes |
