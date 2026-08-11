# Canonical Italian Corpora v1: Inactive Logical Build

## Result

Checkpoint 7B freezes 4,544 broader training records and 22,003 training-eligible standard sonnets as an inactive, manifest-backed logical corpus. The logical training total is 643,822,187 characters.

An exact Minerva token count is deliberately not estimated here; checkpoint 8 will tokenize the frozen V7 training mixtures with the pinned Minerva tokenizer.

- `historical_general`: 210,873,928 characters.
- `historical_non_sonnet_poetry`: 58,032,412 characters.
- `nineteenth_century_bridge`: 363,119,974 characters.
- `standard_sonnets`: 11,795,873 characters.

## Storage

The build references 26,652 unchanged committed slices and writes only 282 new or rewritten slices (37,352,309 logical characters) to 4 bounded delta shards. This avoids copying the unchanged corpus.

## Isolation And Boundary

All 264 checkpoint-7A review decisions are accounted, and all 387 protected V6 validation/test identities remain excluded from training. Conditioned language/form variants are absent. The build is inactive: it creates no V7 split, assigns no mixture weight, starts no GPU work, and deletes no reusable cache.
