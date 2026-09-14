# Few-Shot Context A/B v1

Date: 2026-09-13

Zero, one, and three complete plan-plus-sonnet examples in the chat
context. Same anchored plans, opening, seed, and recipe. Repaired
scheme is the guaranteed-form condition. Judge means come from the
calibrated panel; form-only claims otherwise.

## Form

| Metric | Zero | One | Three |
|---|---:|---:|---:|
| Scheme before repair | 0.5167 | 0.5333 | 0.5583 |
| Scheme after repair | 1.0000 | 1.0000 | 1.0000 |
| Key-match adherence | 0.9315 | 0.9071 | 0.9143 |
| Mean repair distance | 1.1583 | 1.5083 | 1.4167 |
| Accepted lines | 7.508 | 7.342 | 7.508 |

## Coherence proxies

| Metric | Zero | One | Three |
|---|---:|---:|---:|
| type_token_ratio | 0.7281 | 0.7293 | 0.7252 |
| repeated_line_ratio | 0.0000 | 0.0000 | 0.0000 |
| repeated_bigram_ratio | 0.0239 | 0.0255 | 0.0260 |
| final_word_repetition | 0.0119 | 0.0095 | 0.0161 |

## Memorization screen

Verbatim overlap with 16,298 training sonnets (223,892 lines, 5-word shingles). Opening line excluded.

| Metric | Zero | One | Three |
|---|---:|---:|---:|
| Outputs with a copied line | 0 | 0 | 0 |
| Outputs with 4+ copied lines | 0 | 0 | 0 |
| Max copied lines in one output | 0 | 0 | 0 |
| Mean 5-gram overlap | 0.0026 | 0.0023 | 0.0012 |

## Judge panel

| Judge | Zero | One | Three | Separation |
|---|---:|---:|---:|---:|
| opencode-go/glm-5.2 | 2.892857142857143 | 3.09375 | 3.357142857142857 | 1.2083333333333335 |
| opencode-go/qwen3.6-plus | 2.96875 | 3.0 | 2.75 | 1.3333333333333335 |

## Pre-registered reading

- Result: CONTEXT_NULL
- No few-shot condition improves the judge mean by 0.3 or more while keeping form.
- Best condition fewshot1; repaired scheme 1.0000; accepted gap -0.167; judge gain 0.1160714285714286.
- Memorization screen: max copied corpus lines 0; max mean 5-gram overlap 0.0026.

## Caveats

- Repair can break grammar and meaning; the judge panel is the
  coherence evidence, not the checker.
- Judge calibration uses deterministic corruptions of corpus
  sonnets; separation near zero weakens the panel.

Verification: `python3 scripts/score_few_shot_context_validation.py`
