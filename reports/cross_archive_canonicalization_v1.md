# Checkpoint 7A: Global Cross-Archive Canonicalization

Audit date: `2026-08-11`

## Outcome

The decision-only index contains 27,311 audited units:
4,646 broader units and
22,665 standard-sonnet units. It uses exact
normalized hashes plus directional normalized eight-word-shingle containment at
`0.8`. It records 604
threshold overlap pairs and 264 bounded checkpoint-7B
segment decisions.

The indexed units contain 671,417,836 characters
before global exclusions. Removing only fully covered or role-misrouted whole
units leaves 644,304,926
characters, including protected evaluation sonnets. The corresponding
pre-segment-removal training projection is
644,099,304
characters. These are ceilings, not final corpus totals: checkpoint 7B must
remove the hash-pinned embedded spans before final character counts are frozen.

## Frozen input universe

| Source group | Units | Characters |
| --- | ---: | ---: |
| `bibit` | 20,587 | 239,123,963 |
| `existing_historical` | 36 | 48,005,056 |
| `gutenberg` | 1,065 | 292,606,677 |
| `ilc_ota` | 185 | 32,886,660 |
| `liber_liber` | 131 | 26,535,638 |
| `v6_sonnets` | 1,868 | 991,742 |
| `wikisource` | 3,439 | 31,268,100 |

Only completed text-level audits are included. Conditioned material and the
metadata-only archive inventories remain inactive and outside this index.

## Overlap scopes

- `same_role`: 346
- `sonnet_cross_role`: 258

## Canonical decisions

- `exclude_broader_unit_misrouted_as_sonnet`: 75
- `exclude_fully_covered_by_preferred_canonical`: 330
- `retain_canonical_candidate_7b`: 24,931
- `retain_existing_canonical_locked`: 1,517
- `retain_protected_v6_split_locked`: 387
- `retain_unique_after_canonical_segment_quarantine_7b`: 71

Canonical precedence is protected/existing V6 and existing historical text,
then BibIt, Project Gutenberg, Italian Wikisource, Liber Liber, ILC/OTA, and a
stable unit-ID tie-break. A larger lower-priority unit containing a preferred
canonical unit is not silently discarded: its matched span is quarantined for
checkpoint 7B so unique material can be retained.

## Safety boundary

- Protected V6 validation/test identities remain split-locked.
- No conditioned text or metadata-only archive candidate is included.
- No corpus text is activated or newly materialized.
- No V7 split, mixture weight, cache deletion, or GPU work occurs.
- Next checkpoint: 7B role-specific extraction, segment quarantine, and inactive final builds.
