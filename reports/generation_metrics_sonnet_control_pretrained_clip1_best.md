# Generation Metrics

Generation directory: `outputs/generations/sonnet_control_pretrained_clip1_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 493 | 14 | 0 | 0.0791 | 0.1224 | yes | 1337 |
| donna | 558 | 14 | 0 | 0.0699 | 0.0919 | yes | 1338 |
| io_son | 498 | 14 | 0 | 0.0723 | 0.0768 | yes | 1339 |
| solo_et_pensoso | 555 | 14 | 0 | 0.0721 | 0.1268 | yes | 1340 |
| line_start | 532 | 14 | 0 | 0.0677 | 0.1078 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
