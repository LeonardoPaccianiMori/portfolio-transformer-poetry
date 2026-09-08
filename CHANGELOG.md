# Changelog

All notable public-release changes are documented here.

## [Unreleased]

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
