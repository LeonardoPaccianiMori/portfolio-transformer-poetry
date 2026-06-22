# Generation Metrics

Generation directory: `outputs/generations/transformer_context768_scaled_001`

| Prompt | Chars | Lines | Separators | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 904 | 27 | 1 | 0.0520 | 0.0577 | yes | 1337 |
| donna | 905 | 26 | 1 | 0.0497 | 0.0732 | yes | 1338 |
| io_son | 906 | 28 | 0 | 0.0530 | 0.0487 | yes | 1339 |
| solo_et_pensoso | 915 | 26 | 1 | 0.0492 | 0.0384 | yes | 1340 |
| line_start | 906 | 27 | 1 | 0.0519 | 0.0498 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Separators` counts `<|poem_end|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.

