# Changelog

All notable public-release changes are documented here.

## [Unreleased]

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
