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

## Caveats

- Training targets are real poems, so the model learns to imitate corpus
  texts. The memorization screen and the excluded evaluation openings limit
  the risk, but the arm measures imitation of real poetry, not new
  composition.
- Synthetic poems from larger models remain a later lever, with licence and
  provenance review.
