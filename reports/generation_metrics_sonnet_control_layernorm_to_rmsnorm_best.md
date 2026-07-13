# Generation Metrics

Generation directory: `outputs/generations/sonnet_control_layernorm_to_rmsnorm_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 520 | 14 | 0 | 0.0692 | 0.1122 | yes | 1337 |
| donna | 535 | 14 | 0 | 0.0822 | 0.0752 | yes | 1338 |
| io_son | 596 | 14 | 0 | 0.0654 | 0.0961 | yes | 1339 |
| solo_et_pensoso | 562 | 14 | 0 | 0.0587 | 0.1002 | yes | 1340 |
| line_start | 501 | 14 | 0 | 0.0699 | 0.0904 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
