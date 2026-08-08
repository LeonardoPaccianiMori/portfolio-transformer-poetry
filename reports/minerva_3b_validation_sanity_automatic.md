# Minerva 3B Validation Sanity Audit: Automatic Evidence

Generation root: `outputs/generations/minerva_3b_validation_sanity_v1`

## Frozen Scope

Eight V5 validation openings, seed 4242, temperature 0.8, top-k 50, a 512-token ceiling, and decoder-enforced 14-line stopping. No final-test prompt or output participates in selection.

## Conditions

| Condition | Role | Scale | Form | Ceiling | Mean chars | Mean repetition |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base_raw | diagnostic | 0.00 | 7/8 | 2/8 | 1176.5 | 0.3892 |
| base_instructed | diagnostic | 0.00 | 6/8 | 2/8 | 896.5 | 0.2917 |
| best_scale_025 | selectable | 0.25 | 8/8 | 0/8 | 584.6 | 0.3537 |
| best_scale_050 | selectable | 0.50 | 8/8 | 0/8 | 557.5 | 0.1902 |
| best_scale_075 | selectable | 0.75 | 8/8 | 0/8 | 558.2 | 0.1895 |
| best_scale_100 | selectable | 1.00 | 8/8 | 0/8 | 557.2 | 0.1693 |
| final_scale_100 | diagnostic | 1.00 | 8/8 | 0/8 | 550.1 | 0.1579 |

## Selection Rule

Only `best_scale_025`, `best_scale_050`, `best_scale_075`, and `best_scale_100` are eligible. A condition qualifies only with at least 7/8 controlled forms, at least 5/8 generally grammatical outputs, at least 5/8 seven-line topic continuations, and no more than 1/8 severe collapse. Rank qualifiers by grammatical count, then fewer collapses, then topic count, then lower adapter scale. Human judgments come from the separately blinded review.

Automatic repetition is diagnostic only and cannot replace the blinded review.
