# Memorization Checks

Generation directory: `outputs/generations/minerva_7b_v6_candidates_001/candidate_2_epoch_04`

Comparison dataset: `expanded_with_petrarch`

Comparison split: `train`

Character n-gram size: `40`

| Prompt | Chars | Nearest Training Poem | Author | Containment | LCS Chars | Risk | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| giacomo_or_come__seed_4242 | 494 | No shared n-gram | - | 0.0000 | < 40 | low | 4242 |
| cavalcanti_anima_mia__seed_4242 | 525 | No shared n-gram | - | 0.0000 | < 40 | low | 4242 |
| dante_due_donne__seed_4242 | 568 | No shared n-gram | - | 0.0000 | < 40 | low | 4242 |
| petrarca_laura__seed_4242 | 586 | No shared n-gram | - | 0.0000 | < 40 | low | 4242 |
| stampa_piangete__seed_4242 | 561 | No shared n-gram | - | 0.0000 | < 40 | low | 4242 |
| colonna_quando__seed_4242 | 557 | No shared n-gram | - | 0.0000 | < 40 | low | 4242 |
| andreini_gia_non_possio__seed_4242 | 568 | No shared n-gram | - | 0.0000 | < 40 | low | 4242 |
| alfieri_improvvisatrice__seed_4242 | 566 | No shared n-gram | - | 0.0000 | < 40 | low | 4242 |

## Notes

- Text is lowercased and whitespace-normalized before comparison.

- Punctuation is preserved because copied punctuation is useful evidence.

- `Containment` is the fraction of generated character n-grams also found in the nearest training poem.

- `LCS Chars` is the longest contiguous copied character span after normalization.

- When no generated 40-character n-gram occurs in training text, `LCS Chars` is reported as a strict upper bound rather than an unnecessary exact alignment.

- Risk labels are heuristic surface-copying checks, not proof of memorization.

- `medium`: containment >= 0.15 or LCS >= 80 chars.

- `high`: containment >= 0.30 or LCS >= 160 chars.
