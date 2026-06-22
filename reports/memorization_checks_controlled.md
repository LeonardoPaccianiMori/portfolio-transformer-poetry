# Memorization Checks

Generation directory: `outputs/generations/transformer_context768_scaled_001_controlled`

Comparison dataset: `expanded_with_petrarch`

Comparison split: `train`

Character n-gram size: `40`

| Prompt | Chars | Nearest Training Poem | Author | Containment | LCS Chars | Risk | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 476 | Amore è uno desio | Giacomo da Lentini | 0.0000 | 9 | low | 1337 |
| donna | 459 | LXXVII - Bicci novel, figliuol di non so cui | Dante Alighieri | 0.0000 | 11 | low | 1338 |
| io_son | 469 | CXVII - Per quella via che la bellezza corre | Dante Alighieri | 0.0000 | 9 | low | 1339 |
| solo_et_pensoso | 539 | Solo et pensoso i piú deserti campi | Francesco Petrarca | 0.0000 | 15 | low | 1340 |
| line_start | 487 | De vertù de scienzia, il cui podere | Guittone d'Arezzo | 0.0000 | 11 | low | 1341 |

## Notes

- Text is lowercased and whitespace-normalized before comparison.

- Punctuation is preserved because copied punctuation is useful evidence.

- `Containment` is the fraction of generated character n-grams also found in the nearest training poem.

- `LCS Chars` is the longest contiguous copied character span after normalization.

- Risk labels are heuristic surface-copying checks, not proof of memorization.

- `medium`: containment >= 0.15 or LCS >= 80 chars.

- `high`: containment >= 0.30 or LCS >= 160 chars.

