# Plan-Then-Poem Inference Test v1

Date: 2026-09-13

The planned condition appends a numbered list of pre-committed line
endings to the frozen prompt. The format control appends the same list
shape with non-rhyming placeholder endings. The baseline is the
verifier-DPO no-plan output. All systems use the same openings, seeds,
prompt builder, and recipe; the sealed test set was not accessed.
The checker is reviewed and measures form only.

## Plan adherence

| Metric | Planned | Control |
|---|---:|---:|
| Word match rate | 0.0298 | 0.0137 |
| Key match rate | 0.0342 | 0.0143 |
| Planned scheme compliance | 0.0000 | 0.0000 |

## Form metrics

| Metric | Planned | Control | Baseline |
|---|---:|---:|---:|
| Accepted lines (mean) | 5.033 | 4.754 | 6.179 |
| Failed lines (mean) | 4.846 | 5.342 | 4.175 |
| Uncertain lines (mean) | 4.112 | 3.858 | 3.646 |
| Rhyme score (mean) | 0.6675 | 0.5581 | 0.5696 |

## Paired comparisons

| Comparison | Field | Mean difference | 95% CI |
|---|---|---:|---|
| planned - baseline | hendecasyllable_lines | -1.1458 | [-1.5703, -0.7214] |
| planned - baseline | failed_lines | 0.6708 | [0.1716, 1.1700] |
| planned - baseline | uncertain_lines | 0.4667 | [0.0708, 0.8625] |
| planned - baseline | rhyme_score | 0.0978 | [0.0201, 0.1756] |
| control - baseline | hendecasyllable_lines | -1.4250 | [-1.8715, -0.9785] |
| control - baseline | failed_lines | 1.1667 | [0.5824, 1.7509] |
| control - baseline | uncertain_lines | 0.2125 | [-0.1803, 0.6053] |
| control - baseline | rhyme_score | -0.0115 | [-0.0937, 0.0706] |
| planned - control | hendecasyllable_lines | 0.2792 | [-0.1513, 0.7096] |
| planned - control | failed_lines | -0.4958 | [-1.1282, 0.1365] |
| planned - control | uncertain_lines | 0.2542 | [-0.2221, 0.7305] |
| planned - control | rhyme_score | 0.1094 | [0.0262, 0.1926] |

## Pre-registered reading

- Result: TRAIN_PLAN_FOLLOWING_FIRST
- Key-match adherence is below 20%; the model cannot follow plans under this prompt.
- Observed key-match adherence 0.0342; planned scheme compliance 0.0000.

## Caveats

- Form and adherence only. No coherence or literary-quality claim.
- Adherence can be met with odd or archaic endings; the adherence
  rate does not measure meaning.
- The checker's definite coverage limit (87.1% on the ground truth)
  applies to the metre counts.

Verification: `python3 scripts/score_plan_then_poem_validation.py`
