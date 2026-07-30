# Memorization Checks

Generation directory: `outputs/generations/sonnet_task_format_historical_v2_xxl_v5_best_acceptance`

Comparison dataset: `expanded_with_petrarch`

Comparison split: `train`

Character n-gram size: `40`

| Prompt | Chars | Nearest Training Poem | Author | Containment | LCS Chars | Risk | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| giacomo_madonna__seed_1337 | 551 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| giacomo_madonna__seed_1338 | 613 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| dante_di_cio__seed_1337 | 537 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| dante_di_cio__seed_1338 | 518 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| cino_roma_superba__seed_1337 | 536 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| cino_roma_superba__seed_1338 | 587 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| cecco_chi_vol__seed_1337 | 575 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| cecco_chi_vol__seed_1338 | 591 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| cavalcanti_amore_lagia__seed_1337 | 584 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| cavalcanti_amore_lagia__seed_1338 | 601 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| petrarca_successor__seed_1337 | 535 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| petrarca_successor__seed_1338 | 557 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| stampa_meste_rime__seed_1337 | 582 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| stampa_meste_rime__seed_1338 | 570 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| colonna_gravosi_pensier__seed_1337 | 615 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| colonna_gravosi_pensier__seed_1338 | 558 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| andreini_alta_sorte__seed_1337 | 557 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| andreini_alta_sorte__seed_1338 | 586 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |
| alfieri_cessar_mai__seed_1337 | 536 | No shared n-gram | - | 0.0000 | < 40 | low | 1337 |
| alfieri_cessar_mai__seed_1338 | 572 | No shared n-gram | - | 0.0000 | < 40 | low | 1338 |

## Notes

- Text is lowercased and whitespace-normalized before comparison.

- Punctuation is preserved because copied punctuation is useful evidence.

- `Containment` is the fraction of generated character n-grams also found in the nearest training poem.

- `LCS Chars` is the longest contiguous copied character span after normalization.

- When no generated 40-character n-gram occurs in training text, `LCS Chars` is reported as a strict upper bound rather than an unnecessary exact alignment.

- Risk labels are heuristic surface-copying checks, not proof of memorization.

- `medium`: containment >= 0.15 or LCS >= 80 chars.

- `high`: containment >= 0.30 or LCS >= 160 chars.
