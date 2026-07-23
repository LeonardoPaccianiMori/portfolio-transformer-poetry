# Memorization Checks

Generation directory: `outputs/generations/sonnet_corpus_scaling_v4_best`

Comparison dataset: `expanded_with_petrarch`

Comparison split: `train`

Character n-gram size: `40`

| Prompt | Chars | Nearest Training Poem | Author | Containment | LCS Chars | Risk | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 665 | LX - Deh ragioniamo insieme un poco, Amore | Dante Alighieri | 0.0000 | 14 | low | 1337 |
| donna | 543 | Vidi fra mille donne una già tale | Francesco Petrarca | 0.0000 | 17 | low | 1338 |
| io_son | 535 | Cosí potess'io ben chiuder in versi | Francesco Petrarca | 0.0000 | 14 | low | 1339 |
| solo_et_pensoso | 527 | Solo et pensoso i piú deserti campi | Francesco Petrarca | 0.0000 | 16 | low | 1340 |
| line_start | 573 | Occhi miei, oscurato è 'l nostro sole | Francesco Petrarca | 0.0000 | 15 | low | 1341 |

## Notes

- Text is lowercased and whitespace-normalized before comparison.

- Punctuation is preserved because copied punctuation is useful evidence.

- `Containment` is the fraction of generated character n-grams also found in the nearest training poem.

- `LCS Chars` is the longest contiguous copied character span after normalization.

- Risk labels are heuristic surface-copying checks, not proof of memorization.

- `medium`: containment >= 0.15 or LCS >= 80 chars.

- `high`: containment >= 0.30 or LCS >= 160 chars.
