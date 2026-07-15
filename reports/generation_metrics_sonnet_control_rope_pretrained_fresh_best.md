# Generation Metrics

Generation directory: `outputs/generations/sonnet_control_rope_pretrained_fresh_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 639 | 14 | 0 | 0.0501 | 0.1352 | yes | 1337 |
| donna | 560 | 14 | 0 | 0.0714 | 0.0934 | yes | 1338 |
| io_son | 496 | 14 | 0 | 0.0665 | 0.0507 | yes | 1339 |
| solo_et_pensoso | 580 | 14 | 0 | 0.0638 | 0.1421 | yes | 1340 |
| line_start | 502 | 14 | 0 | 0.0737 | 0.0802 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
