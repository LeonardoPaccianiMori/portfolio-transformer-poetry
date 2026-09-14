# Self-Play Rejection SFT v1

Date: 2026-09-14

Rejection fine-tuning on anchored candidates that already satisfy the
scheme without repair. Generation uses the no-plan prompt with the same
openings, seeds, and recipe for both arms. Metrics exclude repair.

## Form

| Metric | Baseline | RFT |
|---|---:|---:|
| Scheme valid without repair | 0.0000 | 0.0000 |
| Accepted lines | 5.833 | 7.246 |
| Failed lines | 5.213 | 1.296 |
| Uncertain lines | 2.954 | 5.458 |
| 4+4+3+3 stanza pattern | 0.5792 | 0.9625 |
| Rhyme score | 0.8854 | 0.7004 |

## Coherence proxies

| Metric | Baseline | RFT |
|---|---:|---:|
| type_token_ratio | 0.5992 | 0.6647 |
| repeated_line_ratio | 0.0000 | 0.0000 |
| repeated_bigram_ratio | 0.0682 | 0.0449 |
| final_word_repetition | 0.2310 | 0.0756 |

## Memorization screen

Verbatim overlap with 16,298 training sonnets (223,892 lines, 5-word shingles). Opening line excluded.

| Metric | Baseline | RFT |
|---|---:|---:|
| Outputs with a copied line | 0 | 0 |
| Max copied lines in one output | 0 | 0 |
| Mean 5-gram overlap | 0.0027 | 0.0057 |

## Pre-registered reading

- Result: SELF_PLAY_NULL
- Autonomous scheme validity 0.0000 against baseline 0.0000 and gate 0.30.
- Accepted-line gap +1.413 (minimum -0.5).
- Judge gain None (minimum -0.3).

## Caveats

- Training examples come from model outputs on the same 120 openings
  with different seeds, so the measurement covers form, not new
  openings.
- The judge panel is secondary evidence; the gate uses judge gain only
  as a guard against coherence loss.

Verification: `python3 scripts/score_self_play_rft.py`
