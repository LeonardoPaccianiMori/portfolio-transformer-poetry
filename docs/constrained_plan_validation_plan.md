# Constrained Plan Validation Plan (Plan-Then-Poem, Step 2)

## Status and scope

Leonardo approved this step on 2026-09-13 after the plan-following SFT showed
blind list copying and a line-1 construction flaw that made every plan
infeasible. This arm fixes plan construction and adds deterministic ending
repair, the cheap equivalent of constrained decoding for this pilot. A
decode-time finite-state constraint is a later refinement.

## Question

With feasible, opening-anchored plans and repaired endings, does the system
produce consistently valid sonnets, and what happens to coherence proxies?

## Plans

- Line 1 is the opening prefill; its ending is the opening's own final word
  and anchors rhyme class A.
- The other class-A lines take words from the same lexicon key as the
  opening's final word. Classes B to E take distinct keys from the lexicon,
  excluding the most frequent word per key.
- If the opening's key is absent from the lexicon, the opening is skipped and
  counted.
- The plan seed is recorded.

## Conditions

- `anchored`: the plan-following model with the anchored plan in the prompt,
  no repair.
- `anchored_repaired`: the same outputs with each line-final word replaced by
  the planned word when it differs. The repair distance per output is
  recorded.
- Baseline: the verifier-DPO no-plan outputs at the same openings and seeds.

## Metrics

- Scheme compliance (the checker's observed scheme equals the plan),
  key-match and word-match adherence, repair distance per line, accepted,
  failed, and uncertain lines, rhyme score, plus repeated-line and
  line-length checks as coherence proxies.

## Pre-registered reading

- If repaired scheme compliance is at least 95% and accepted lines stay
  within half a line of the no-repair anchored condition, the form objective
  is met; next comes coherence evaluation and a larger-base run.
- If repair fixes the scheme but accepted lines collapse or the mean repair
  distance exceeds 4 of 14 lines, report the trade-off and try decode-time
  constraints before further training.
- If the no-repair anchored condition already reaches 95% scheme compliance,
  skip repair as unnecessary.

## Budget

- About 10 minutes of H100 for 240 outputs, under $0.50, plus CPU scoring.
  The instance is already running.

## Limits

- Repair can break grammar and meaning; all claims remain form-only.
- Coherence proxies are structural and are not a literary judgment.
- No release, no checkpoint publication, no public wording.

## Status on 2026-09-13

The constrained validation ran: 480 outputs (anchored and repaired).
Anchored scheme compliance is 0.5083 with 0.9256 key-match adherence; the
repaired condition reaches 1.0000 scheme compliance with a mean repair
distance of 1.217 lines. Repaired minus anchored accepted lines is -0.054
(95% CI -0.124 to 0.015) and the rhyme score changes by +0.006, so repair
costs no measurable metre or rhyme quality. Against the verifier-DPO
baseline, accepted lines rise by 1.25 and failed lines fall by 2.57. The
pre-registered reading is FORM_OBJECTIVE_MET. The remaining open problem is
coherence: repair can break grammar or meaning, and the checker does not
measure it. Next candidates: a few-shot context A/B, a larger-base plan SFT,
or a documented coherence review. See
`reports/constrained_plan_validation_v1.md`.
