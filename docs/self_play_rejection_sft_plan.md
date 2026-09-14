# Self-Play Rejection SFT Plan

Date: 2026-09-14
Status: approved by Leonardo

## Objective

Teach the merged model to produce a valid rhyme scheme without a plan list in
the prompt and without ending repair. This is the first experiment of the
agreed autonomous-rhyme route (self-play rejection SFT, then reasoning-trace
SFT, with RL later).

## Data

- Source candidates: anchored generations from
  `artifacts/local/self_play/validation/generation/` (seeds 5202-5205, 480
  anchored outputs; plan to extend to seeds 5202-5213 for 1,440 anchored
  outputs).
- Keep rule (`candidate_keep`): scheme valid before repair (ABBA ABBA CDE CDE
  with valid quatrain and tercet), zero copied continuation lines against the
  V8 training corpus, at least 7 accepted hendecasyllable lines, at most 2
  failed lines, type-token ratio at least 0.60, no repeated lines, repeated
  bigram ratio at most 0.05, and final-word repetition at most 0.05.
- The first build kept 164 of 480 candidates (scheme 214 rejections, accepted
  lines 61, failed lines 29, degenerate 12, copied 0).

## Training

- Base: local merged model
  `artifacts/local/plan_following_sft/training/merged_model`.
- Objective: response-masked SFT with the no-plan prompt
  (`autonomous_prompt`), target = the 13 continuation lines plus EOS.
- LoRA rank 16, alpha 32, dropout 0.05, 1 epoch, batch 8, accumulation 2,
  learning rate 1e-4, seed 11414. Config: `configs/self_play_rft.json`.
- Trainer: `train_plan_following_sft` with a direct base model and no
  verifier adapter (commit `84ad96c`).

## Run History

- Run 1 (2026-09-14): the example builder removed interior blank lines, so
  every RFT output lost the 4+4+3+3 stanza breaks (stanza pattern `[14]`) and
  the run is invalidated as a test of its hypothesis. The builder now
  preserves interior blank lines and run 2 repeats the same protocol.

## Evaluation

- Two arms for the same 120 openings and seeds 5200-5201:
  `autonomous_baseline` (merged model) and `autonomous_rft` (RFT merged
  model), both with the no-plan prompt and the shared recipe.
- Primary metric: scheme valid without repair (quatrain and tercet valid).
- Secondary: accepted, failed, and uncertain lines, 4+4+3+3 stanza pattern,
  rhyme score, coherence proxies, and the corpus memorization screen.
- Coherence guard: calibrated judge panel on a sample (optional, secondary).
- Success gate: RFT scheme validity at least 0.30, accepted-line gap at least
  -0.5, and judge gain at least -0.3. Otherwise the reading is
  SELF_PLAY_NULL and the next step is reasoning-trace SFT.

## Budget and Boundaries

- GPU: roughly two hours of one H100 for extra generation, training, and the
  two evaluation arms.
- No model release, no push, and no public wording change is part of this
  experiment.
