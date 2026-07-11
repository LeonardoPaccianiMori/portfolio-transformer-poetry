# Memorization Checks

Generation directory: `outputs/generations/sonnet_control_pretrained_fresh_best`

Comparison dataset: `expanded_with_petrarch`

Comparison split: `train`

Character n-gram size: `40`

| Prompt | Chars | Nearest Training Poem | Author | Containment | LCS Chars | Risk | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 493 | Il mio adversario in cui veder solete | Francesco Petrarca | 0.0000 | 18 | low | 1337 |
| donna | 565 | XXIX - Se tutta l'acqu'a balsamo tornasse | Cecco Angiolieri | 0.0000 | 16 | low | 1338 |
| io_son | 517 | Poscia ch'io vidi gli occhi di costei | Cino da Pistoia | 0.0000 | 18 | low | 1339 |
| solo_et_pensoso | 506 | Io vidi li occhi dove Amor si mise | Guido Cavalcanti | 0.0000 | 19 | low | 1340 |
| line_start | 532 | Apollo, s'anchor vive il bel desio | Francesco Petrarca | 0.0000 | 16 | low | 1341 |

## Notes

- Text is lowercased and whitespace-normalized before comparison.

- Punctuation is preserved because copied punctuation is useful evidence.

- `Containment` is the fraction of generated character n-grams also found in the nearest training poem.

- `LCS Chars` is the longest contiguous copied character span after normalization.

- Risk labels are heuristic surface-copying checks, not proof of memorization.

- `medium`: containment >= 0.15 or LCS >= 80 chars.

- `high`: containment >= 0.30 or LCS >= 160 chars.
