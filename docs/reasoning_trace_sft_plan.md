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

## Caveats

- Training targets are real poems, so the model learns to imitate corpus
  texts. The memorization screen and the excluded evaluation openings limit
  the risk, but the arm measures imitation of real poetry, not new
  composition.
- Synthetic poems from larger models remain a later lever, with licence and
  provenance review.
