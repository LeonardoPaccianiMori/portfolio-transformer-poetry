# Generation Metrics

Generation directory: `outputs/generations/sonnet_control_historical_v2_xxl_v5_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 546 | 14 | 0 | 0.0842 | 0.0921 | yes | 1337 |
| donna | 465 | 14 | 0 | 0.0796 | 0.0368 | yes | 1338 |
| io_son | 597 | 14 | 0 | 0.0704 | 0.1263 | yes | 1339 |
| solo_et_pensoso | 525 | 14 | 0 | 0.0686 | 0.1303 | yes | 1340 |
| line_start | 564 | 14 | 0 | 0.0674 | 0.1070 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
