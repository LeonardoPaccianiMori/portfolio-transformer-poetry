# Memorization Checks

Generation directory: `outputs/generations/sonnet_control_rmsnorm_pretrained_fresh_best`

Comparison dataset: `expanded_with_petrarch`

Comparison split: `train`

Character n-gram size: `40`

| Prompt | Chars | Nearest Training Poem | Author | Containment | LCS Chars | Risk | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 589 | XIX - Eo ho sì tristo il cor di cose cento | Cecco Angiolieri | 0.0000 | 14 | low | 1337 |
| donna | 539 | Levommi il mio penser in parte ov'era | Francesco Petrarca | 0.0000 | 15 | low | 1338 |
| io_son | 562 | LXX - Un danaio, non che far cottardita | Cecco Angiolieri | 0.0000 | 14 | low | 1339 |
| solo_et_pensoso | 476 | XXXIV - I' ho tutte le cose ch'io non voglio | Cecco Angiolieri | 0.0000 | 17 | low | 1340 |
| line_start | 504 | Se di voi, donna, mi negai servente | Guittone d'Arezzo | 0.0000 | 15 | low | 1341 |

## Notes

- Text is lowercased and whitespace-normalized before comparison.

- Punctuation is preserved because copied punctuation is useful evidence.

- `Containment` is the fraction of generated character n-grams also found in the nearest training poem.

- `LCS Chars` is the longest contiguous copied character span after normalization.

- Risk labels are heuristic surface-copying checks, not proof of memorization.

- `medium`: containment >= 0.15 or LCS >= 80 chars.

- `high`: containment >= 0.30 or LCS >= 160 chars.
