# Verifier-Labelled Form DPO Plan

## Status and scope

Leonardo approved this experiment on 2026-09-13 after the full-weight V8
retrain returned a null result. It is one bounded DPO run that uses the
reviewed prosody checker as the preference verifier. It does not approve a
model release, checkpoint publication, sealed-test access, or public wording.
Any public claim remains form-only.

## Question

Do checker-labelled form preferences improve formal sonnet validity over
Stage-3?

## Data

- Candidates: the frozen 4,096 Stage-3 DPO preference candidates
  (512 openings, 2 recipes, 4 seeds) under
  `artifacts/local/minerva_7b_v7_dpo/candidates/authoritative/`.
- Verifier: the reviewed `sonnet_prosody` checker scores each candidate. The
  form score is accepted hendecasyllable lines plus 2 times `quatrain_ok`
  plus 2 times `tercet_ok` plus 2 times `all_lines_valid` plus `rhyme_score`.
- Degeneracy filter: alphabetic ratio below 0.65 or mean word length below
  3.3. Degenerate candidates cannot be chosen or rejected.
- Pairs: strongest versus weakest candidate per opening, with a minimum score
  gap of 2.0 and a minimum chosen score of 8.0.
- Built result: 4,096 candidates scored, 103 degenerate candidates excluded,
  425 pairs over 425 openings; chosen scores average 9.71 (range 8 to 14),
  rejected scores average 2.42 (range 0 to 8). The primary agent audited
  samples of the chosen and rejected texts. The frozen preference file stays
  local because it embeds raw model output.

## Method

- Base: the published Stage-3 model, loaded from a verified local copy.
- Loss and adapter: the existing hand-written DPO loss and rank-8 all-linear
  LoRA, BF16, with the same hyperparameters as the published AI-judged DPO.
- Preferences: a new schema and loader
  (`minerva_7b_v7_verifier_form_preferences_v1`). No AI-judge vote counts are
  fabricated or reused.
- One epoch with the prompt-disjoint validation split and the recorded seeds.

## Frozen evaluation

- Matched Stage-3 versus verifier-DPO generation on the recorded 120
  validation prompts and the recorded seeds, with the same recipe and prompt
  builder in both systems.
- Scoring with the reviewed checker and paired comparisons.
- Pre-registered reading: signal if full-form valid outputs appear, or
  accepted hendecasyllable lines rise by at least 0.23 with a 95% interval
  that excludes zero and the rhyme score does not drop significantly;
  otherwise NULL.
- The sealed test set is not accessed.

## Deliverables

- Verifier preference dataset and its statistics.
- Trained adapter on the instance, retained locally if the result is
  informative.
- `reports/verifier_labelled_dpo_v1.md` and `.json`.
- Career Center and poetry-repository records.

## Budget

About 30 to 45 minutes of H100, roughly $1 to $2.

## Limits

- Form only. The checker's definite coverage limit (87.1% on the ground
  truth) applies.
- The form score is a proxy; it does not measure coherence, grammar, or
  literary quality.
- DPO on surface form can trade against coherence; the report must state
  this and not claim a general quality gain.
- No release, no checkpoint publication, no public wording before a separate
  review.

## Status on 2026-09-13

The run completed. Training used 383 train pairs and 42 validation pairs and
finished 48 of 48 updates with validation preference accuracy 0.762, at a
cost of $0.10. The matched validation grid produced 960 outputs with no test
access. On 480 paired comparisons the verifier DPO gained 0.767 accepted
hendecasyllable lines per output (95% CI 0.458 to 1.075) with no significant
rhyme-score change (-0.002, 95% CI -0.060 to 0.057), so the pre-registered
reading is SIGNAL. No full-form valid output exists in either system, and the
quatrain and tercet scheme rates remain zero.

The gain is partial and carries a material caveat: failed lines rose by 1.904
(95% CI 1.592 to 2.216) while uncertain lines fell by 2.671. The adapter
sharpened outcomes more than it reduced errors, and fewer than a third of the
lines that left the uncertain state became accepted. The accepted-line
increase is therefore not a reduction of total formal errors, and no valid
sonnet was produced. See `reports/verifier_labelled_dpo_v1.md`.
