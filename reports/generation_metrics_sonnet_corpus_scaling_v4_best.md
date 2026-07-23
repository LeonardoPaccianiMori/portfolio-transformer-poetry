# Generation Metrics

Generation directory: `outputs/generations/sonnet_corpus_scaling_v4_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 665 | 14 | 0 | 0.0647 | 0.1133 | yes | 1337 |
| donna | 543 | 14 | 0 | 0.0792 | 0.0963 | yes | 1338 |
| io_son | 535 | 14 | 0 | 0.0654 | 0.1147 | yes | 1339 |
| solo_et_pensoso | 527 | 14 | 0 | 0.0702 | 0.0706 | yes | 1340 |
| line_start | 573 | 14 | 0 | 0.0855 | 0.0702 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
