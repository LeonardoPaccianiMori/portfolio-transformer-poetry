# Memorization Checks

Generation directory: `outputs/generations/sonnet_control_pretrained_warmup_cosine_best`

Comparison dataset: `expanded_with_petrarch`

Comparison split: `train`

Character n-gram size: `40`

| Prompt | Chars | Nearest Training Poem | Author | Containment | LCS Chars | Risk | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 500 | Deh Gherarduccio, com' campasti tue | Cino da Pistoia | 0.0000 | 15 | low | 1337 |
| donna | 558 | Dante, io ho preso l'abito di doglia | Cino da Pistoia | 0.0000 | 14 | low | 1338 |
| io_son | 586 | Il mio adversario in cui veder solete | Francesco Petrarca | 0.0000 | 15 | low | 1339 |
| solo_et_pensoso | 504 | CXXII - – Udite udite, dico a voi, signori | Cecco Angiolieri | 0.0000 | 16 | low | 1340 |
| line_start | 878 | Se 'l dolce sguardo di costei m'ancide | Francesco Petrarca | 0.0000 | 17 | low | 1341 |

## Notes

- Text is lowercased and whitespace-normalized before comparison.

- Punctuation is preserved because copied punctuation is useful evidence.

- `Containment` is the fraction of generated character n-grams also found in the nearest training poem.

- `LCS Chars` is the longest contiguous copied character span after normalization.

- Risk labels are heuristic surface-copying checks, not proof of memorization.

- `medium`: containment >= 0.15 or LCS >= 80 chars.

- `high`: containment >= 0.30 or LCS >= 160 chars.
