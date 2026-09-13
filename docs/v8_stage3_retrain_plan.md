# V8 Stage-3 Retrain Plan

## Status and scope

Leonardo approved this retrain on 2026-09-13 after the form-targeted LoRA
pilot returned a null result. The independent review accepted the direction
and required pre-registered lineage, evaluation intervals, retention checks,
and no single blind endpoint.

This plan covers one full-weight retrain of the sonnet stage on the corrected
V8 corpus, starting from the published Stage-2 BF16 model, plus its frozen
evaluation. It does not approve a model release, checkpoint publication, DPO
access, or any public wording.

## Question

Does retraining the sonnet stage on the corrected V8 corpus improve formal
sonnet validity over Stage-3 without degrading fluency?

## Data and lineage

- V8 train: the 4+4+3+3 derived view of `v8_split == "train"`, encoded with
  the frozen Minerva tokenizer. The encode report and the token shard hash are
  the lineage record.
- Preservation replay: `data/local/minerva_7b_v7/encoded/
  modern_preservation_replay-00000.int32.bin`, used at 5% of target-token
  exposure. The shard hash and the used token range are recorded.
- V8 validation: the `v8_split == "validation"` records, encoded with the same
  tokenizer, used only for loss evaluation.
- Report window count, update count, and replay token count are computed from
  the shard sizes and hashes at run time, not estimated.

## Method

- Base: the published Stage-2 BF16 model, loaded from a verified local copy.
- Full weights, BF16, no quantization, no adapters.
- Optimizer: PagedAdamW8bit, weight decay 0.01, gradient clip 1.0.
- Schedule: peak learning rate 1e-6, minimum 1e-7, cosine decay, 7-update
  warmup, one pass over the frozen window order.
- Windows: 2,048 tokens, one micro-batch window per step with 16-step gradient
  accumulation (16 windows per update).
- Abort rules: non-finite loss, pre-clip gradient norm above 100, or the spend
  ceiling in the config.

## Safeguards

- Validation loss on the V8 validation windows and retention loss on the
  held-out replay tail every 15 updates.
- Retention gate: replay loss ratio at the final update at most 1.05 against
  the pre-training value.
- A midpoint checkpoint at update 45 and the final checkpoint.
- Small generation probes (20 openings, 1 seed) at the midpoint and the end,
  scored with the reviewed prosody checker. Probe abort: mean accepted lines
  below half of the Stage-3 baseline mean (5.688).

## Frozen evaluation

- Openings and recipe: the same recorded validation prompts, seeds 6200 and
  6201, temperature 0.85, `top_p` 0.95, `no_repeat_ngram_size` 4, at most 512
  new tokens, and the frozen chat prompt builder.
- Baseline: the Stage-3 outputs already generated in the pilot, reused
  unchanged.
- Candidate: the retrained Stage-3 checkpoint, 240 outputs.
- Scoring: paired comparison with the reviewed checker and a multivariate
  pre-registered rule:
  - signal if full-form valid candidate outputs appear, or accepted
    hendecasyllable lines rise by at least 0.23 with a 95% interval that
    excludes zero, while the rhyme score does not drop significantly and the
    retention gate passes;
  - otherwise NULL.
- The sealed test set is not accessed.

## Deliverables

- Full-weight checkpoint (mid and final) on the instance; final model retained
  locally if the result is informative.
- Training report and log with losses, retention ratios, and probe scores.
- `reports/v8_stage3_retrain_v1.md` and `.json`.
- Career Center and poetry-repository records.

## Budget and limits

- One H100 80 GB session: Stage-2 download, encoding of V8 validation,
  training, 240 candidate generations, and scoring. About 1.5 to 2.5 hours and
  $4 to $6.
- V8 is a corrected V7 with explicit stanza rendering. Corpus corrections
  alone may move form only slightly; the result may be another null.
- The simplified trainer does not reproduce every original protocol feature
  (frozen stage windows, checkpoint selection by validation loss, the full
  retention gate set). Deviations are recorded.
- Form-only result. No literary-quality or public claim.
