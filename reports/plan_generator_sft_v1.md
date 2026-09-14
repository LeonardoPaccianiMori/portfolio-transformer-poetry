# Plan Generator and Composed Pipeline v1

Date: 2026-09-14

Stage one writes the rhyme plan only. Stage two writes the poem with
the existing plan-following model, using every valid stage-one plan.
The composed rate counts the full opening-and-seed grid where the plan
is valid and the poem is scheme-valid without repair.

## Stage one: plan validity

| Metric | Baseline | Plan SFT |
|---|---:|---:|
| Outputs | 240 | 240 |
| Parsed plans | 0 | 239 |
| Valid scheme patterns | 0 | 239 |
| Valid plans (true rhymes) | 0 | 221 |

## Stage two and composed

| Metric | Baseline | Plan SFT |
|---|---:|---:|
| Stage-two poems | 0 | 221 |
| Poems scheme-valid | n/a | 0.5475 |
| Accepted lines | n/a | 7.484 |
| Failed lines | n/a | 1.190 |
| Composed pipeline rate | 0.0000 | 0.5042 |

## Pre-registered reading

- Result: SELF_PLAY_SIGNAL
- Autonomous scheme validity 0.5042 against baseline 0.0000 and gate 0.30.
- Accepted-line gap +7.484 (minimum -0.5).
- Judge gain None (minimum -0.3).

## Caveats

- Stage two uses the previously trained plan-following model; the
  composed rate is the honest end-to-end number for this pipeline.
- Plans come from corpus poems and lexicon augmentation, so the
  planning skill is imitation of valid plans, not free invention.

Verification: `python3 scripts/score_plan_generator.py`
