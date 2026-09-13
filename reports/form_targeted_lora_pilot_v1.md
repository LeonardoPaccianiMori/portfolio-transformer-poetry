# Form-Targeted LoRA Pilot v1

Date: 2026-09-13

Matched validation comparison of the Stage-3 baseline (adapter
disabled) and the form-targeted LoRA candidate (adapter enabled).
The 120 validation openings, seeds, prompt builder, and recipe are
identical across systems; the sealed test set was not accessed.
The checker is reviewed and measures form only, not literary quality.

## Per-system form metrics

| Metric | Baseline | Candidate |
|---|---:|---:|
| Outputs | 240 | 240 |
| Structure OK rate | 1.0000 | 1.0000 |
| Stanza pattern OK rate | 0.0333 | 0.0000 |
| All 14 lines valid rate | 0.0000 | 0.0000 |
| Quatrain scheme OK rate | 0.0000 | 0.0000 |
| Tercet scheme OK rate | 0.0000 | 0.0000 |
| Mean hendecasyllable lines | 5.688 | 1.246 |
| Mean failed lines | 1.917 | 8.267 |
| Mean uncertain lines | 6.396 | 4.487 |
| Mean rhyme score | 0.5315 | 0.6154 |

- Full-form valid outputs: baseline 0, candidate 0

## Paired comparisons (candidate minus baseline)

| Field | Mean difference | 95% CI | + pairs | - pairs | = pairs |
|---|---:|---|---:|---:|---:|
| hendecasyllable_lines | -4.4417 | [-4.7508, -4.1325] | 3 | 223 | 14 |
| failed_lines | 6.3500 | [5.9899, 6.7101] | 235 | 1 | 4 |
| uncertain_lines | -1.9083 | [-2.3329, -1.4837] | 59 | 154 | 27 |
| perfect_rhyme_pairs | 1.0917 | [0.7161, 1.4672] | 136 | 60 | 44 |
| soft_rhyme_pairs | 0.5417 | [0.3416, 0.7417] | 75 | 27 | 138 |
| rhyme_score | 0.0839 | [0.0042, 0.1636] | 92 | 67 | 81 |

The shared comparison helper labels the first paired system
`stage_3` and the second `dpo`; here the first is the candidate and
the second is the baseline, so the table is candidate minus baseline.

## Discordant binary outcomes (McNemar normal approximation)

| Field | Candidate only | Baseline only | Both | Neither | p |
|---|---:|---:|---:|---:|---:|
| structure_ok | 0 | 0 | 240 | 0 | 1 |
| stanza_pattern_ok | 0 | 8 | 0 | 232 | 0.004678 |
| all_lines_valid | 0 | 0 | 0 | 240 | 1 |
| quatrain_ok | 0 | 0 | 0 | 240 | 1 |
| tercet_ok | 0 | 0 | 0 | 240 | 1 |

## Pre-registered reading

- Result: NULL
- No pre-registered signal: no valid candidate output and no accepted-line increase that excludes zero and reaches 0.23 lines.

## Caveats

- Form only. No literary-quality or selection-for-quality claim.
- The checker's definite coverage limit (87.1% on the ground truth)
  applies to the metre counts.
- One LoRA arm can underpower the test by design; a null result
  reframes the direction rather than closing it.

Verification: `python3 scripts/score_form_targeted_validation.py`
