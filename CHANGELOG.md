# Changelog

All notable public-release changes are documented here.

## [Unreleased]

- Added the plan-then-poem inference test implementation: planned and format
  control prompts, adherence metrics, scoring, and tests. No generation run
  has started.
- Added the reviewed plan-then-poem pilot plan and built the V8 train rhyme
  lexicon: 1,220 rhyme keys from 228,164 line endings. No generation run has
  started.
- Ran the verifier-labelled form DPO: 48 updates, 425 preference pairs, and
  960 matched validation outputs. Accepted hendecasyllable lines rose by 0.77
  (95% CI 0.46 to 1.08) with no rhyme-score change, but failed lines rose by
  1.90 and no valid sonnet was produced. Partial signal under the
  pre-registered rule.
- Added the verifier-labelled form DPO pipeline and built its frozen
  preference dataset from the 4,096 existing candidates: 425 pairs over 425
  openings, with 103 degenerate candidates excluded. No training run has
  started.
- Ran the full-weight V8 Stage-3 retrain: 93 updates, retention gate passed,
  and no formal gain on 240 matched validation pairs (accepted lines -0.225,
  95% CI -0.578 to 0.128; no valid outputs). Reading: NULL.
- Fixed causal language-model label alignment in both trainers. The earlier
  LoRA pilot used off-by-one targets, so its null result is invalid as a test
  of that hypothesis.
- Added the approved full-weight V8 Stage-3 retrain: frozen window plan with
  5% preservation replay, safeguarded trainer, candidate generation and
  merged evaluation, and tests. The preflight plan is 1,490 windows and 93
  optimizer updates.
- Ran the form-targeted LoRA pilot: 178 steps on 2.9M tokens, then 480 matched
  validation generations. The candidate degraded generation quality, lost
  accepted hendecasyllable lines, and produced no valid sonnet. The
  pre-registered reading is NULL.
- Added the approved form-targeted LoRA pilot: the V8 train encoder, one-arm
  LoRA trainer, matched validation generation, scoring, tests, and the frozen
  pilot plan. No training run has started.
- Reviewed all 54 metre flags and the one rhyme-scheme mismatch in the A2
  packet. Every flag is conservative, no definite checker error was found, and
  one reason-set limitation (triphthong synaeresis) is documented.
- Added retroactive prosody scoring of the sealed Stage-3 and DPO outputs. No
  output is fully definite-valid; DPO shows a small increase in
  hendecasyllable lines and no rhyme-score difference.
- Added the frozen sonnet-prosody ground truth of 30 Dante and Petrarch poems,
  the validation report, and the 100-line owner review packet.
- Hardened the `sonnet_prosody` checker for editorial accents, diaeresis,
  silent leading `h`, long-s normalization, and documented metric
  ambiguities. Metre accuracy on definite ground-truth lines is 100% with
  87.1% coverage; rhyme scheme agreement is 29/30 exact and 30/30 soft.
- Added a rule-based Italian sonnet prosody checker (`sonnet_prosody`) with a
  command-line entry point and unit tests. It reports sonnet structure,
  hendecasyllable metre, stress type, rhyme keys, scheme comparison, and
  uncertainty flags. Accuracy against reviewed ground truth remains a later
  validation checkpoint.
- Added the corrected V8 future-training corpus while preserving frozen V1/V7
  evidence. V8 resolves 67 anthology attribution records and two incomplete sonnets,
  quarantines two unresolved author abbreviations and 3,599 author/work split
  collisions, and derives one consistent `4+4+3+3` stanza view.
- Replaced 49,889 occurrences of the historical long-s character `ſ` with `s`
  in V8 and quarantined 41 broader records that met the frozen held-out overlap
  threshold. No meaningful pair remains after quarantine.
- Fixed memorization reference ranking so the highest-risk match takes
  precedence. Rechecking all 4,976 final outputs left every saved risk label and
  nearest reference unchanged, with zero medium- or high-risk outputs.
- Published selected Minerva V7 Stages 1–3 and the DPO adapter in one public,
  ungated Hugging Face repository after private upload, exact remote inventory
  review, clean re-download, hash and safetensors validation, three-model load,
  adapter attachment, and fixed-equivalence checks.
- Retained exact cumulative sampled-window lineage, layered CC BY-NC 4.0 /
  Apache-2.0 scope, model cards, notices, and training-content summaries for
  the published package.
- Corrected the Stage-3 peak learning rate in the model card from `2e-6` to
  the protocol- and telemetry-confirmed `1e-6`.

## [1.0.0] - 2026-08-20

- Added fail-closed current-tree and published-history release inventories.
- Defined mixed licensing, notices, citation, AI contribution, and public scope.
- Reconciled the completed PAISÀ preservation replay with the earlier
  prospective rescue curriculum.
- Added public CPU verification and deterministic aggregate portfolio evidence.
- Sanitized machine-specific operational material from current public reports.
