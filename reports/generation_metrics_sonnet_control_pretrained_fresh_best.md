# Generation Metrics

Generation directory: `outputs/generations/sonnet_control_pretrained_fresh_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 493 | 14 | 0 | 0.0872 | 0.1449 | yes | 1337 |
| donna | 565 | 14 | 0 | 0.0673 | 0.1121 | yes | 1338 |
| io_son | 517 | 14 | 0 | 0.0658 | 0.0875 | yes | 1339 |
| solo_et_pensoso | 506 | 14 | 0 | 0.0731 | 0.1054 | yes | 1340 |
| line_start | 532 | 14 | 0 | 0.0677 | 0.1078 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
