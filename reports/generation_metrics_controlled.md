# Generation Metrics

Generation directory: `outputs/generations/transformer_context768_scaled_001_controlled`

| Prompt | Chars | Lines | Separators | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 476 | 14 | 0 | 0.0882 | 0.0190 | yes | 1337 |
| donna | 459 | 14 | 0 | 0.0893 | 0.0219 | yes | 1338 |
| io_son | 469 | 14 | 0 | 0.0853 | 0.0300 | yes | 1339 |
| solo_et_pensoso | 539 | 14 | 0 | 0.0668 | 0.0224 | yes | 1340 |
| line_start | 487 | 14 | 0 | 0.0821 | 0.0124 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Separators` counts `<|poem_end|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.

