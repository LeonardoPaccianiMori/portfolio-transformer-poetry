# Plan-Following SFT v1

Date: 2026-09-13

The planned condition shows the model the pre-committed line endings.
The mismatched condition shows the plan of a different opening, which
separates plan reading from a shifted ending distribution. The format
control shows non-rhyming placeholder endings. The baseline is the
verifier-DPO no-plan output. All conditions use the same openings,
seeds, prompt builder, and recipe; the sealed test set was not
accessed. Echo outputs are excluded from adherence and reported.

## Adherence

| Metric | Planned | Mismatched | Control |
|---|---:|---:|---:|
| Word match rate | 0.6982 | 0.7226 | 0.8443 |
| Key match rate | 0.7110 | 0.7336 | 0.8580 |
| Planned scheme compliance | 0.0042 | 0.0000 | 0.0000 |
| Echo outputs | 0 | 0 | 0 |

## Form metrics

| Metric | Planned | Mismatched | Control | Baseline |
|---|---:|---:|---:|---:|
| Accepted lines (mean) | 7.508 | 7.379 | 7.483 | 6.179 |
| Failed lines (mean) | 1.733 | 1.854 | 1.092 | 4.175 |
| Uncertain lines (mean) | 4.758 | 4.767 | 5.425 | 3.646 |
| Rhyme score (mean) | 0.9784 | 0.9763 | 0.0708 | 0.5696 |

## Planned key-match adherence by line position

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.017 | 0.958 | 0.875 | 0.938 | 0.887 | 0.775 | 0.754 | 0.633 | 0.725 | 0.708 | 0.700 | 0.646 | 0.596 | 0.742 |

## Paired comparisons

| Comparison | Field | Mean difference | 95% CI |
|---|---|---:|---|
| planned - baseline | hendecasyllable_lines | 1.3292 | [0.9224, 1.7360] |
| planned - baseline | failed_lines | -2.4417 | [-2.8370, -2.0464] |
| planned - baseline | uncertain_lines | 1.1125 | [0.7801, 1.4449] |
| planned - baseline | rhyme_score | 0.4087 | [0.3487, 0.4687] |
| mismatched - baseline | hendecasyllable_lines | 1.2000 | [0.8080, 1.5920] |
| mismatched - baseline | failed_lines | -2.3208 | [-2.7209, -1.9207] |
| mismatched - baseline | uncertain_lines | 1.1208 | [0.7746, 1.4670] |
| mismatched - baseline | rhyme_score | 0.4067 | [0.3475, 0.4659] |
| control - baseline | hendecasyllable_lines | 1.3042 | [0.9027, 1.7057] |
| control - baseline | failed_lines | -3.0833 | [-3.4434, -2.7232] |
| control - baseline | uncertain_lines | 1.7792 | [1.4577, 2.1007] |
| control - baseline | rhyme_score | -0.4988 | [-0.5626, -0.4350] |
| planned - mismatched | hendecasyllable_lines | 0.1292 | [-0.2082, 0.4665] |
| planned - mismatched | failed_lines | -0.1208 | [-0.4007, 0.1590] |
| planned - mismatched | uncertain_lines | -0.0083 | [-0.3660, 0.3493] |
| planned - mismatched | rhyme_score | 0.0020 | [-0.0126, 0.0166] |
| planned - control | hendecasyllable_lines | 0.0250 | [-0.3086, 0.3586] |
| planned - control | failed_lines | 0.6417 | [0.4021, 0.8813] |
| planned - control | uncertain_lines | -0.6667 | [-1.0050, -0.3283] |
| planned - control | rhyme_score | 0.9075 | [0.8748, 0.9402] |

## Pre-registered reading

- Result: ONE_MORE_EPOCH_THEN_REVIEW
- Adherence is in the partial band; one identical epoch is pre-registered, then a single re-evaluation.
- Planned key-match 0.7110; mismatched 0.7336; gap 95% CI [-0.0590, 0.0138]; planned scheme compliance 0.0042.

## Caveats

- Training uses each sonnet's own endings; reproduction is rewarded
  and prospective planning is not tested.
- Adherence can be met with odd or archaic endings and does not
  measure coherence or literary quality.
- The checker's definite coverage limit (87.1% on the ground truth)
  applies to the metre counts.

Verification: `python3 scripts/score_plan_then_poem_validation.py`
