# Reasoning-Trace SFT Plan

Date: 2026-09-14
Status: approved by Leonardo

## Objective

Teach autonomous rhyme planning. The model first writes a rhyme plan (the
scheme plus one ending word per line) and then the sonnet. Self-play
rejection SFT failed because imitation alone does not teach the choice of
endings; this arm makes the choice an explicit product.

## Data

- Source: the 16,298 V8 sonnets. Exclude every poem whose source hash matches
  one of the 120 evaluation prompts.
- Per poem, keep only poems whose checker scheme is valid (quatrain in
  `ABBAABBA` or `ABABABAB`, tercet in `ABCABC`, `ABCACB`, or `ABABAB`).
- Derive the trace from the poem itself: canonical 14-letter scheme, then the
  ending word of lines 2-14 from the real text. Traces are deterministic and
  free of teacher-licence issues.
- Training pair: instruction + opening line (no endings, no scheme) ->
  trace + full poem with the original stanza breaks.
- Builder: `scripts/build_reasoning_trace_data.py` writes
  `artifacts/local/reasoning_trace/data/traces_v1.jsonl` plus stats.

## Training

- Base: the local merged model (rebuilt on the instance, byte-identical).
- Response-masked SFT on the trace plus poem. LoRA rank 16, alpha 32,
  dropout 0.05, 1 epoch, micro-batch 4 with accumulation 4 (effective batch
  16; batch 8 with accumulation 2 exceeded 48 GB on the A6000 because the
  trace targets are long), learning rate 1e-4, seed 11415. Config:
  `configs/reasoning_trace_sft.json`.
- Trainer: `train_plan_following_sft` with a direct base model and a
  separate tokenizer directory.

## Evaluation

- Two arms, same 120 openings, seeds, and recipe: `trace_baseline` (merged
  model) and `trace_sft` (trained model). Single-shot generation: the model
  writes the trace and the poem.
- Parsing: `parse_trace` extracts the scheme, the 13 entries, and the 14 poem
  lines. `validate_trace` checks canonical form, the valid scheme patterns,
  and that every key group shares a true rhyme key. A baseline output without
  a trace is scored as poem-only with an invalid trace.
- Metrics: trace parse rate, trace validity, poem scheme validity without
  repair (primary), poem scheme equals traced scheme, accepted, failed, and
  uncertain lines, 4+4+3+3 stanza pattern, rhyme score, coherence proxies,
  memorization screen, and an optional calibrated judge panel.
- Gate: trained scheme validity at least 0.30 with accepted-line gap at
  least -0.5 and judge gain at least -0.3; otherwise NULL.

## Budget and Boundaries

- One A6000 run: data build is local, training and generation on the instance,
  scoring local. Roughly 1-1.5 hours of GPU time.
- No model release, no push, and no public wording change.

## Run History

- Run 1 (2026-09-14): SELF_PLAY_NULL on the gate, with a clear diagnosis.
  The trained arm parses a trace in 0.9917 of outputs and every parsed trace
  has a valid scheme pattern (ABBA ABBA CDCDCD, ABBA ABBA CDECDE, ABAB ABAB
  CDCDCD), so the format and the scheme structure are learned. Rhyme
  consistency fails: only 11 of 238 planned ending sets actually rhyme
  (0.046). When the trace is valid, the poem is scheme-valid without repair
  7 of 11 times. Overall scheme validity is 0.0292 against 0.0000 for the
  baseline. Failed lines fall from 8.883 to 1.496 and final-word repetition
  from 0.405 to 0.066. The bottleneck is the choice of rhyming endings, not
  the format. See `reports/reasoning_trace_sft_v1.md`.

## Plan-Generator Follow-up (approved 2026-09-14)

- Objective: attack the rhyme-choice bottleneck directly by training a
  plan-only model (opening -> trace, EOS after the plan), then composing it
  with the existing plan-following poem model.
- Data: the 11,264 corpus traces plus 11,258 lexicon-augmented valid plans
  (one per corpus opening), 22,522 cards in
  `artifacts/local/plan_generator/data/plans_v1.jsonl`.
- Training: plan-only target, LoRA rank 16, 1 epoch, micro-batch 8, effective
  batch 16, seed 11416. Config `configs/plan_generator_sft.json`.
- Evaluation: stage one generates plans for both arms (merged baseline and
  plan model); stage two writes poems with the existing merged model for every
  valid plan; the composed rate counts the full grid where the plan is valid
  and the poem is scheme-valid without repair.
- Result (2026-09-14): SELF_PLAY_SIGNAL. Plan validity is 221/240 (92.1%) for
  the trained arm and 0/240 for the baseline. Stage-two poems are scheme-valid
  0.5475 of the time with 7.484 accepted and 1.190 failed lines. The composed
  pipeline rate is 0.5042 against the 0.30 gate. The accepted-line comparison
  is trivial because the baseline produced no valid plans.
  See `reports/plan_generator_sft_v1.md`.

## Plan-Follower v2 and Composed Comparison (approved 2026-09-14)

- Stage-one plans from the v1 plan model at eight seeds (5200-5207):
  955/960 parsed, 955 valid schemes, 676 valid plans (70.4%).
- Stage two wrote poems for all 676 valid plans with two followers on the
  same plan grid: the original merged plan-follower (v1) and a follower
  fine-tuned on 324 valid generated-plan poems (v2, 160 steps, $0.11).
- Composed scheme-valid rate without repair: v1 follower 0.3646, v2 follower
  0.5583. Given a valid plan, the v2 follower writes a valid poem 0.7929 of
  the time (v1: 0.5178), with 7.525 accepted and 0.994 failed lines.
- Coherence of the v1 composed poems, calibrated judge panel: GLM-5.2 mean
  2.875 (separation 1.625), Qwen 3.6 Plus mean 2.750 (separation 1.917).
  Form is solved; coherence remains the open problem.
- Reports: `reports/plan_composer_follower_v1.md`,
  `reports/plan_composer_follower_v2.md`.
- Plan sampling temperature: at 0.85 the plan model is seed-sensitive (seed
  5204: 35% valid, seed 5205: 26%, other seeds 90%+). At 0.4 the unlucky seeds
  recover to 92.1% and good seeds reach 92.9%, so the plan recipe now uses
  temperature 0.4.
- Final form evaluation (2026-09-14, frozen): 960-plan grid at temperature
  0.4 with the v2 follower. Valid plans 891/960 (0.9281); stage-two poems
  scheme-valid 0.6734 with 7.327 accepted and 1.147 failed lines; composed
  pipeline rate 0.6250. Form is frozen at this level; further form work is
  limited to the single-model attempt. Report:
  `reports/plan_composer_temp04_v1.md`.

## Single-Model Attempt (approved 2026-09-14)

- Objective: one model that writes plan and poem in a single generation,
  initialized from the plan generator and trained on the joint dataset
  (11,264 corpus trace cards plus 600 valid pipeline pairs; 727 steps,
  $0.69). Data builder: `scripts/build_joint_trace_data.py`.
- Result (960-plan grid, eight seeds): plans parse 0.9510 and are valid
  0.5396 (planning transferred from the plan generator), but poems are
  scheme-valid only 0.2604 - below the 0.50 success bar and the 0.30 gate.
  Given a valid plan, the poem follows it 48% of the time versus 79% for the
  dedicated follower. Rhyme score 0.9798, accepted lines 8.048, no
  memorization. Reading: SELF_PLAY_NULL for the single-model bar.
- One more attempt is allowed by the stop rule; otherwise the two-model
  pipeline (0.6250) remains the form deliverable. See
  `reports/joint_trace_sft_v1.md`.

## Caveats

- Training targets are real poems, so the model learns to imitate corpus
  texts. The memorization screen and the excluded evaluation openings limit
  the risk, but the arm measures imitation of real poetry, not new
  composition.
- Synthetic poems from larger models remain a later lever, with licence and
  provenance review.
