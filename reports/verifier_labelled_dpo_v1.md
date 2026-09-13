# Verifier-Labelled Form DPO v1

Date: 2026-09-13

Matched validation comparison of the published Stage-3 baseline and the
verifier-labelled form DPO adapter. The 120 validation openings, seeds,
prompt builder, and recipe are identical across systems; the sealed test
set was not accessed. The checker is reviewed and measures form only,
not literary quality.

## Per-system form metrics

| Metric | Stage 3 | Verifier DPO |
|---|---:|---:|
| Outputs | 480 | 480 |
| Structure OK rate | 1.0000 | 1.0000 |
| Stanza pattern OK rate | 0.0521 | 0.0500 |
| All 14 lines valid rate | 0.0000 | 0.0000 |
| Quatrain scheme OK rate | 0.0000 | 0.0000 |
| Tercet scheme OK rate | 0.0000 | 0.0000 |
| Mean hendecasyllable lines | 5.417 | 6.183 |
| Mean failed lines | 2.269 | 4.173 |
| Mean uncertain lines | 6.315 | 3.644 |
| Mean rhyme score | 0.5696 | 0.5680 |

- Full-form valid outputs: Stage 3 0, verifier DPO 0

## Paired comparisons (verifier DPO minus Stage 3)

| Field | Mean difference | 95% CI | + pairs | - pairs | = pairs |
|---|---:|---|---:|---:|---:|
| hendecasyllable_lines | 0.7667 | [0.4580, 1.0753] | 262 | 156 | 62 |
| failed_lines | 1.9042 | [1.5921, 2.2163] | 323 | 103 | 54 |
| uncertain_lines | -2.6708 | [-2.9466, -2.3950] | 66 | 364 | 50 |
| perfect_rhyme_pairs | 0.0125 | [-0.1501, 0.1751] | 172 | 178 | 130 |
| soft_rhyme_pairs | 0.0458 | [-0.0279, 0.1195] | 82 | 69 | 329 |
| rhyme_score | -0.0016 | [-0.0600, 0.0567] | 146 | 144 | 190 |

The shared comparison helper labels the first paired system `stage_3`
and the second `dpo`; here the first is the verifier DPO adapter and
the second is the Stage-3 baseline, so the table is DPO minus Stage 3.

## Discordant binary outcomes (McNemar normal approximation)

| Field | DPO only | Stage 3 only | Both | Neither | p |
|---|---:|---:|---:|---:|---:|
| structure_ok | 0 | 0 | 480 | 0 | 1 |
| stanza_pattern_ok | 23 | 24 | 1 | 432 | 0.884 |
| all_lines_valid | 0 | 0 | 0 | 480 | 1 |
| quatrain_ok | 0 | 0 | 0 | 480 | 1 |
| tercet_ok | 0 | 0 | 0 | 480 | 1 |

## Pre-registered reading

- Result: SIGNAL
- Paired accepted-line increase excludes zero and reaches the sealed DPO effect, with no significant rhyme-score drop.
- Secondary movement: failed lines +1.904 [1.592, 2.216], uncertain lines -2.671 [-2.947, -2.395].

## Caveats

- Form only. No literary-quality or selection-for-quality claim.
- The checker's definite coverage limit (87.1% on the ground truth)
  applies to the metre counts.
- DPO on surface form can trade against coherence; this report does
  not measure coherence or grammar.

Verification: `python3 scripts/score_verifier_dpo_validation.py`
