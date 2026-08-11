# Expanded Standard-Sonnet Corpus V7 Split Freeze

## Result

Checkpoint 8A accounts for all 22,693 canonical sonnet identities and includes 22,390 in the V7 train/validation/test corpus. The remaining 303 retain their canonical exclusion decisions.

- `test`: 1,244 sonnets.
- `train`: 19,899 sonnets.
- `validation`: 1,247 sonnets.

All 1,868 V6 assignments are preserved exactly: 1,481 train, 190 validation, and 197 test. The V6 evaluation tier remains exact-identity/work held out but is explicitly not claimed to be author-disjoint.

## Clean V7 Held-Out Cohorts

New grouped assignment adds 1,057 validation and 1,047 test sonnets. Resolved authors are absent from V6 and from V7 training; generic author labels are grouped by complete source work. Author/work connected components cannot cross the new train/validation/test boundary.

- `bibit`: validation 800; test 834.
- `gutenberg`: validation 109; test 69.
- `ilc_ota`: validation 2; test 0.
- `wikisource`: validation 146; test 144.

The approved revised policy retains all 2,118 new sonnets whose canonical author also appears in protected V6 evaluation. This does not make the legacy tier author-disjoint; it preserves valuable training text while the separate clean V7 cohorts measure author-level generalization.

## Boundary

This checkpoint freezes split identities only. It copies no corpus text, includes no conditioned material, performs no Minerva tokenization, assigns no training-mixture weight, starts no GPU work, and deletes no reusable cache.
