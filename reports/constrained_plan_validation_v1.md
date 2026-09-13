# Constrained Plan Validation v1

Date: 2026-09-13

Opening-anchored plans fix line 1 to the opening's own ending and
build a feasible scheme. The repaired condition replaces line-final
words with the planned words. The baseline is the verifier-DPO no-plan
output. Form only; no literary-quality claim.

## Metrics

| Metric | Anchored | Repaired | Baseline |
|---|---:|---:|---:|
| Scheme compliance | 0.5083 | 1.0000 | 0.0000 |
| Key-match adherence | 0.9256 | 1.0000 | n/a |
| Mean repair distance | n/a | 1.217 | n/a |
| Accepted lines (mean) | 7.483 | 7.429 | 6.179 |
| Failed lines (mean) | 1.587 | 1.608 | 4.175 |
| Rhyme score (mean) | 0.9900 | 0.9957 | 0.5696 |

## Paired comparisons

| Comparison | Field | Mean difference | 95% CI |
|---|---|---:|---|
| anchored - baseline | hendecasyllable_lines | 1.3042 | [0.9015, 1.7068] |
| anchored - baseline | failed_lines | -2.5875 | [-2.9841, -2.1909] |
| anchored - baseline | uncertain_lines | 1.2833 | [0.9361, 1.6306] |
| anchored - baseline | rhyme_score | 0.4203 | [0.3621, 0.4786] |
| repaired - baseline | hendecasyllable_lines | 1.2500 | [0.8525, 1.6475] |
| repaired - baseline | failed_lines | -2.5667 | [-2.9589, -2.1745] |
| repaired - baseline | uncertain_lines | 1.3167 | [0.9700, 1.6633] |
| repaired - baseline | rhyme_score | 0.4261 | [0.3676, 0.4845] |
| repaired - anchored | hendecasyllable_lines | -0.0542 | [-0.1238, 0.0154] |
| repaired - anchored | failed_lines | 0.0208 | [-0.0128, 0.0545] |
| repaired - anchored | uncertain_lines | 0.0333 | [-0.0299, 0.0966] |
| repaired - anchored | rhyme_score | 0.0057 | [0.0018, 0.0097] |

## Pre-registered reading

- Result: FORM_OBJECTIVE_MET
- Repaired scheme compliance reaches 95% without a material accepted-line drop.
- Anchored scheme 0.5083, repaired 1.0000, repaired-minus-anchored accepted lines -0.054 [-0.124, 0.015], mean repair distance 1.217.

## Caveats

- Repair can break grammar and meaning; all claims are form-only.
- The coherence proxies are structural, not a literary judgment.

Verification: `python3 scripts/score_constrained_plan_validation.py`
