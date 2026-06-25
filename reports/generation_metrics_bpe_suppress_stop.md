# Generation Metrics

Generation directory: `outputs/generations/transformer_bpe_512_context256_001_suppress_stop`

| Prompt | Chars | Lines | Separators | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 362 | 14 | 0 | 0.0884 | 0.0669 | yes | 1337 |
| donna | 557 | 14 | 0 | 0.0862 | 0.0830 | yes | 1338 |
| io_son | 578 | 14 | 0 | 0.0606 | 0.0730 | yes | 1339 |
| solo_et_pensoso | 473 | 14 | 0 | 0.0761 | 0.0894 | yes | 1340 |
| line_start | 527 | 14 | 0 | 0.0740 | 0.0763 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Separators` counts `<|poem_end|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.

