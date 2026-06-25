# Memorization Checks

Generation directory: `outputs/generations/transformer_bpe_512_context256_001_suppress_stop`

Comparison dataset: `expanded_with_petrarch`

Comparison split: `train`

Character n-gram size: `40`

| Prompt | Chars | Nearest Training Poem | Author | Containment | LCS Chars | Risk | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 362 | Dispregio pregio u' non pregi'ha pregianza | Guittone d'Arezzo | 0.0000 | 13 | low | 1337 |
| donna | 557 | Signor, e' non passò mai peregrino | Cino da Pistoia | 0.0000 | 13 | low | 1338 |
| io_son | 578 | Che fai alma? che pensi? avrem mai pace? | Francesco Petrarca | 0.0000 | 14 | low | 1339 |
| solo_et_pensoso | 473 | LXXI - - Voi, donne, che pietoso atto mostrate | Dante Alighieri | 0.0000 | 15 | low | 1340 |
| line_start | 527 | Perchè non furo a me gli occhi dispenti | Guido Cavalcanti | 0.0000 | 12 | low | 1341 |

## Notes

- Text is lowercased and whitespace-normalized before comparison.

- Punctuation is preserved because copied punctuation is useful evidence.

- `Containment` is the fraction of generated character n-grams also found in the nearest training poem.

- `LCS Chars` is the longest contiguous copied character span after normalization.

- Risk labels are heuristic surface-copying checks, not proof of memorization.

- `medium`: containment >= 0.15 or LCS >= 80 chars.

- `high`: containment >= 0.30 or LCS >= 160 chars.

