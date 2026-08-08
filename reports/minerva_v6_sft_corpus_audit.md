# Minerva V6 SFT Corpus Audit

## Scope

- Dataset: `expanded_with_petrarch`
- Manifest: `data/metadata/sonnets_expanded_v6_manifest.csv`
- Manifest SHA-256: `994c4c374f42ba26f1c352d7ad7c3adec7ec4671507770bd7c485cb6f977a4fa`
- Selected poems: 1,868
- Split counts: train 1,481 / validation 190 / test 197

## Automated Structural Gate

- Result: **pass**
- Structural issues: 0
- Exact normalized duplicate groups: 0
- Cross-split duplicate groups: 0
- Suspicious markers: `{}`

## Composition

### Periods

| Period | Poems |
| --- | ---: |
| XVI secolo | 701 |
| XIII secolo | 484 |
| XIV secolo | 430 |
| XVII secolo | 196 |
| XVIII secolo | 45 |
| XIX secolo | 12 |

### Largest Training Authors

| Author | Poems | Share |
| --- | ---: | ---: |
| Vittoria Colonna | 268 | 18.1% |
| Francesco Petrarca | 253 | 17.1% |
| Gaspara Stampa | 217 | 14.7% |
| Guittone d'Arezzo | 162 | 10.9% |
| Isabella Andreini | 151 | 10.2% |
| Cecco Angiolieri | 102 | 6.9% |
| Cino da Pistoia | 65 | 4.4% |
| Dante Alighieri | 64 | 4.3% |
| Vittorio Alfieri | 36 | 2.4% |
| Guido Cavalcanti | 35 | 2.4% |

### Largest Training Collections

| Collection | Poems | Share |
| --- | ---: | ---: |
| Rime (Vittoria Colonna) | 268 | 18.1% |
| Canzoniere (Rerum vulgarium fragmenta) | 253 | 17.1% |
| Rime (Stampa) | 217 | 14.7% |
| Rime (Guittone d'Arezzo) | 162 | 10.9% |
| Rime (Andreini) | 151 | 10.2% |
| Rime (Angiolieri) | 102 | 6.9% |
| Rime (Cino da Pistoia) | 65 | 4.4% |
| Rime (Dante) | 64 | 4.3% |
| Rime varie (Alfieri, 1912) | 36 | 2.4% |
| Rime (Cavalcanti) | 35 | 2.4% |

## Cleaning And Line Diagnostics

- Line length: minimum 9, mean 36.9, maximum 53 characters.
- Lines over 120 characters: 0.
- Lines under 4 characters: 0.
- Poems with recorded editorial-bracket removal: 1,868.
- Poems with recorded line-marker removal: 1,868.
- Poems with cleaning notes: 1,868.
- Poems with audit notes: 996.

## Interpretation

Automated checks cannot certify historical grammar; review the deterministic sample before freezing another Minerva recipe.
The companion review sample is not training data duplication; it is a deterministic view of committed V6 texts for editorial inspection.
The companion review is complete: all 24 sampled poems were accepted as coherent historical sonnets without visible editorial or cleaning contamination.
