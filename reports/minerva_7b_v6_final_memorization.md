# Memorization Checks

Generation directory: `outputs/generations/minerva_7b_v6_final_001`

Comparison dataset: `expanded_with_petrarch`

Comparison split: `train`

Character n-gram size: `40`

| Prompt | Chars | Nearest Training Poem | Author | Containment | LCS Chars | Risk | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| giacomo_madonna__seed_1337 | 488 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| giacomo_madonna__seed_1338 | 547 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| dante_di_cio__seed_1337 | 512 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| dante_di_cio__seed_1338 | 451 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| cino_roma_superba__seed_1337 | 471 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| cino_roma_superba__seed_1338 | 566 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| cecco_chi_vol__seed_1337 | 588 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| cecco_chi_vol__seed_1338 | 557 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| cavalcanti_amore_lagia__seed_1337 | 508 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| cavalcanti_amore_lagia__seed_1338 | 553 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| petrarca_successor__seed_1337 | 554 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| petrarca_successor__seed_1338 | 527 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| stampa_meste_rime__seed_1337 | 588 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| stampa_meste_rime__seed_1338 | 546 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| colonna_gravosi_pensier__seed_1337 | 535 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| colonna_gravosi_pensier__seed_1338 | 518 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| andreini_alta_sorte__seed_1337 | 556 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| andreini_alta_sorte__seed_1338 | 562 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| alfieri_cessar_mai__seed_1337 | 572 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| alfieri_cessar_mai__seed_1338 | 552 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |

## Notes

- Text is lowercased and whitespace-normalized before comparison.

- Punctuation is preserved because copied punctuation is useful evidence.

- `Containment` is the fraction of generated character n-grams also found in the nearest training poem.

- `LCS Chars` is the longest contiguous copied character span after normalization.

- When no generated 40-character n-gram occurs in training text, `LCS Chars` is reported as a strict upper bound rather than an unnecessary exact alignment.

- Risk labels are heuristic surface-copying checks, not proof of memorization.

- `medium`: containment >= 0.15 or LCS >= 80 chars.

- `high`: containment >= 0.30 or LCS >= 160 chars.
