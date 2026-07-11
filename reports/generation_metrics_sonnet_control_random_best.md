# Generation Metrics

Generation directory: `outputs/generations/sonnet_control_random_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 483 | 14 | 0 | 0.0642 | 0.0875 | yes | 1337 |
| donna | 520 | 14 | 0 | 0.0712 | 0.0696 | yes | 1338 |
| io_son | 519 | 14 | 0 | 0.0713 | 0.1240 | yes | 1339 |
| solo_et_pensoso | 603 | 14 | 0 | 0.0630 | 0.0717 | yes | 1340 |
| line_start | 521 | 14 | 0 | 0.0729 | 0.0946 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
