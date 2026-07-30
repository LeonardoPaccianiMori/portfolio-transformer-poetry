# Memorization Checks

Generation directory: `outputs/generations/sonnet_control_historical_v2_xxl_v5_best`

Comparison dataset: `expanded_with_petrarch`

Comparison split: `train`

Character n-gram size: `40`

| Prompt | Chars | Nearest Training Poem | Author | Containment | LCS Chars | Risk | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 546 | Lasso, ben so che dolorose prede | Francesco Petrarca | 0.0000 | 16 | low | 1337 |
| donna | 465 | Sonetto XXII | Ludovico Ariosto | 0.0000 | 15 | low | 1338 |
| io_son | 597 | Cotale gioco mai non fue veduto | Giacomo da Lentini | 0.0000 | 18 | low | 1339 |
| solo_et_pensoso | 525 | L'aura mia sacra al mio stanco riposo | Francesco Petrarca | 0.0000 | 19 | low | 1340 |
| line_start | 564 | XXXVI - S'i' non torni ne l'odïo d'Amore | Cecco Angiolieri | 0.0000 | 18 | low | 1341 |

## Notes

- Text is lowercased and whitespace-normalized before comparison.

- Punctuation is preserved because copied punctuation is useful evidence.

- `Containment` is the fraction of generated character n-grams also found in the nearest training poem.

- `LCS Chars` is the longest contiguous copied character span after normalization.

- Risk labels are heuristic surface-copying checks, not proof of memorization.

- `medium`: containment >= 0.15 or LCS >= 80 chars.

- `high`: containment >= 0.30 or LCS >= 160 chars.
