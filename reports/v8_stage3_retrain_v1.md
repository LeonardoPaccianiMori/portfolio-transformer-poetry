# V8 Stage-3 Retrain v1

Date: 2026-09-13

Matched validation comparison of the published Stage-3 baseline and the V8 Stage-3 retrained checkpoint. The 120 validation openings, seeds, prompt builder, and recipe are identical across systems; the sealed test set was not accessed. The checker is reviewed and measures form only, not literary quality.

## Per-system form metrics

| Metric | Baseline | Candidate |
|---|---:|---:|
| Outputs | 240 | 240 |
| Structure OK rate | 1.0000 | 0.9958 |
| Stanza pattern OK rate | 0.0333 | 0.0292 |
| All 14 lines valid rate | 0.0000 | 0.0000 |
| Quatrain scheme OK rate | 0.0000 | 0.0000 |
| Tercet scheme OK rate | 0.0000 | 0.0000 |
| Mean hendecasyllable lines | 5.688 | 5.463 |
| Mean failed lines | 1.917 | 1.700 |
| Mean uncertain lines | 6.396 | 6.825 |
| Mean rhyme score | 0.5315 | 0.5931 |

- Full-form valid outputs: baseline 0, candidate 0

## Paired comparisons (candidate minus baseline)

| Field | Mean difference | 95% CI | + pairs | - pairs | = pairs |
|---|---:|---|---:|---:|---:|
| hendecasyllable_lines | -0.2250 | [-0.5776, 0.1276] | 96 | 106 | 38 |
| failed_lines | -0.2167 | [-0.4860, 0.0526] | 81 | 91 | 68 |
| uncertain_lines | 0.4292 | [0.0557, 0.8027] | 121 | 85 | 34 |
| perfect_rhyme_pairs | 0.0375 | [-0.1923, 0.2673] | 85 | 82 | 73 |
| soft_rhyme_pairs | -0.0292 | [-0.1184, 0.0600] | 35 | 36 | 169 |
| rhyme_score | 0.0615 | [-0.0247, 0.1478] | 80 | 65 | 95 |

The shared comparison helper labels the first paired system
`stage_3` and the second `dpo`; here the first is the candidate and
the second is the baseline, so the table is candidate minus baseline.

## Discordant binary outcomes (McNemar normal approximation)

| Field | Candidate only | Baseline only | Both | Neither | p |
|---|---:|---:|---:|---:|---:|
| structure_ok | 0 | 1 | 239 | 0 | 0.3173 |
| stanza_pattern_ok | 7 | 8 | 0 | 225 | 0.7963 |
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
- The retrain is one full-weight stage on the corrected V8 corpus; a null result reframes the direction rather than closing it.

Verification: `python3 scripts/score_form_targeted_validation.py`
