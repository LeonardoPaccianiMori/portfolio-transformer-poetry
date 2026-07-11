# Memorization Checks

Generation directory: `outputs/generations/finetuning_larger_20k_001_step_2000`

Comparison dataset: `expanded_with_petrarch`

Comparison split: `train`

Character n-gram size: `40`

| Prompt | Chars | Nearest Training Poem | Author | Containment | LCS Chars | Risk | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 573 | Giovine bella, luce del mio core | Cino da Pistoia | 0.0000 | 16 | low | 1337 |
| donna | 597 | Amor, se cosa è che 'n signoria | Guittone d'Arezzo | 0.0000 | 13 | low | 1338 |
| io_son | 544 | LI - Maladetta sie l'or' e 'l punt'e 'l giorno | Cecco Angiolieri | 0.0000 | 14 | low | 1339 |
| solo_et_pensoso | 536 | LXXXII - I' ho sì poco di quel ch'i' vorrei | Cecco Angiolieri | 0.0000 | 17 | low | 1340 |
| line_start | 521 | Ben si conosce lo servente e vede | Guittone d'Arezzo | 0.0000 | 16 | low | 1341 |

## Notes

- Text is lowercased and whitespace-normalized before comparison.

- Punctuation is preserved because copied punctuation is useful evidence.

- `Containment` is the fraction of generated character n-grams also found in the nearest training poem.

- `LCS Chars` is the longest contiguous copied character span after normalization.

- Risk labels are heuristic surface-copying checks, not proof of memorization.

- `medium`: containment >= 0.15 or LCS >= 80 chars.

- `high`: containment >= 0.30 or LCS >= 160 chars.
