# Plan-Following SFT Plan

## Status and scope

Leonardo approved this direction on 2026-09-13. The plan-then-poem inference
test showed that the base model cannot follow pre-committed line endings:
3.42% key-match adherence against 1.43% in the format control, with zero
scheme compliance and degraded metre. The pre-registered gate is
TRAIN_PLAN_FOLLOWING_FIRST.

This plan covers one supervised fine-tuning arm that teaches the model to
emit planned line endings, starting from the verifier-DPO model. A second
plan-then-poem training stage is not approved here. The plan was revised
after an independent review required a mismatched-plan control, echo
detection, position-wise adherence, and fixed retry rules.

## Question

Can supervised fine-tuning teach the model to end each line with the word
shown in the prompt, and does plan following produce a valid rhyme scheme?

## Training data

- Source: the same V8 corrected sonnet train split used by the retrain. Any
  record whose source identity overlaps the 120 evaluation prompts is
  excluded; a leak check reports the overlap count, which must be zero.
- For every sonnet, the plan is the sonnet's own 14 line-final words in
  order. This rewards reproduction of the endings and does not test
  prospective planning; that limitation is recorded.
- The prompt is the frozen instruction arm with the opening line and the
  numbered plan appended, exactly as in the inference test. The target is the
  continuation after the opening prefill, with the recorded 4+4+3+3 stanza
  blank lines and the EOS token.
- Loss is computed only on the continuation (response masking). The prompt
  contributes no gradient.
- Records: 14 non-empty lines, a usable final word per line, and the derived
  V8 text hash. Malformed records are excluded and counted.

## Method

- Base: Stage-3 with the verifier-DPO adapter merged into the weights, so the
  metre gains are retained.
- Adapter: PEFT LoRA, rank 16, alpha 32, dropout 0.05, all linear modules,
  BF16.
- One epoch, batch 8 with gradient accumulation 2, learning rate 1e-4 with 3%
  warmup and cosine decay to 1e-5, clip 1.0.
- Checkpoints: a midpoint adapter at the halfway optimizer step and a final
  adapter. The final adapter is reported by default; the midpoint is reported
  too, and either can be selected by the held-out validation loss on a
  prompt-disjoint 5% split that is fixed before training.
- Stop rules: non-finite loss, gradient norm above 100 before clipping, or
  the spend ceiling. If the ceiling is reached, the run stops after the last
  saved adapter and skips the merged-model save; the final adapter remains
  the deliverable, and the report states the skip.

## Frozen evaluation

- Conditions, all on the recorded 120 validation prompts and seeds 5200 and
  5201, with the explicit recipe temperature 0.85, `top_p` 0.95,
  `no_repeat_ngram_size` 4, at most 512 new tokens, and the
  13-continuation-line stop rule:
  - planned: the same lexicon-built per-prompt plans used by the inference
    test (scheme `ABBAABBACDECDE`, the same recorded seeds), so the
    before-and-after comparison is like for like;
  - mismatched: the plan of the next prompt in a fixed cyclic order, the
    control that distinguishes plan reading from a shifted ending
    distribution;
  - format control: the same list shape with non-rhyming placeholder
    endings.
- Baseline: the verifier-DPO no-plan outputs at the same openings and seeds.
- Echo guard: an output is marked `echo` when it contains three or more
  numbered-list lines or the instruction phrase. Echo outputs are excluded
  from the adherence numerator and reported separately.
- Metrics: word-match and key-match adherence overall and per line position,
  planned scheme compliance, accepted, failed, and uncertain lines, rhyme
  score, with paired comparisons.

## Pre-registered reading

- Success: planned key-match adherence at least 40% after echo exclusion,
  planned scheme compliance above zero, and planned adherence clearly above
  the mismatched condition. Then propose the full plan-then-poem stage with
  lexicon-chosen endings and theme fields.
- Partial: adherence between 20% and 40%. Then run at most one additional
  epoch with the same configuration and re-evaluate once; no other change.
- Failure: adherence below 20%, or planned adherence not above the
  mismatched condition. Then stop this route and reconsider full-weight SFT
  or a different plan representation.
- A metre drop against the baseline of more than half a line is a material
  warning even if adherence rises.

## Budget

- Data build and tests: CPU only.
- One H100 session: data build, LoRA SFT, merged-model save, 720 evaluation
  outputs, scoring. About 1.5 hours and $3 to $5. The spend ceiling is $6.

## Deliverables

- Plan-following adapter and merged model on the instance, retained locally.
- Training report and log.
- `reports/plan_following_sft_v1.md` and `.json`.
- Career Center and poetry-repository records.

## Limits

- Training plans come from each sonnet's own endings, so the objective
  rewards reproduction, not prospective planning. Prospective planning
  remains a later question.
- Plan following can be satisfied by copying endings without coherence; the
  report must separate adherence from form and coherence.
- The checker's definite coverage limit (87.1% on the ground truth) applies.
- No release, no checkpoint publication, no public wording before a separate
  review.

## Status on 2026-09-13

The SFT completed 968 of 968 steps (15,476 train and 814 validation examples,
8 malformed records excluded) with a final validation loss of 2.41 and a cost
of $1.11. The 20-prompt adherence probe reached 0.729 key match.

The full evaluation ran 720 outputs. Planned key-match is 0.711, mismatched
0.734, and control 0.858; the planned-minus-mismatched interval is
[-0.059, 0.014]. The model copies whichever list it is shown and does not
discriminate plans, but the copying mechanism works. Planned scheme
compliance is 0.0042. Per line position, adherence is 0.017 on line 1 and
between 0.60 and 0.96 elsewhere. Line 1 is the forced opening prefill, so the
planned first ending is impossible by construction; this plan-construction
flaw explains most of the near-zero scheme rate. The pre-registered reading
is ONE_MORE_EPOCH_THEN_REVIEW.

Next steps, in order: fix plan construction so line 1 is anchored to the
opening's own ending and the remaining endings complete a feasible scheme;
then either one more identical epoch or constrained ending decoding that
forces the planned final word per line. See
`reports/plan_following_sft_v1.md`.
