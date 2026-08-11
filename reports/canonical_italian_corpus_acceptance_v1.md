# Canonical Italian Corpus Acceptance Freeze v1

## Result

Checkpoint 7C exhaustively verifies 26,934 stored logical units across 1,930 committed physical files. Every physical-file hash, logical byte range, UTF-8 boundary, character count, and logical hash passes.

The accepted inactive training view contains 4,544 broader records, 22,003 standard sonnets, and 643,822,187 logical characters.

- `historical_general`: 210,873,928 training characters.
- `historical_non_sonnet_poetry`: 58,032,412 training characters.
- `nineteenth_century_bridge`: 363,119,974 training characters.
- `standard_sonnets`: 11,795,873 training characters.

## Safety Boundary

All 387 protected V6 validation/test sonnets remain readable only through the explicit protected-audit iterator and are excluded from the default training iterator. All paths are repository-relative, no logical storage points into `data/local/`, and conditioned material is absent.

The corpus remains inactive. This checkpoint creates no V7 split, performs no Minerva tokenization, assigns no mixture weight, starts no GPU work, and deletes no reusable cache.

## Frozen Identities

- Logical identity SHA-256: `0aeb0ee8ffed91c294b31f27fa85471418acf4e5ff47cf84a17a5e2deb666b57`
- Physical identity SHA-256: `749c585cc7d32687cb2196b742b2925135fd6bf35807e9f94abbb96fe5f82473`
