# Generation Metrics

Generation directory: `outputs/generations/sonnet_control_pretrained_warmup_cosine_best`

| Prompt | Chars | Lines | Boundary Markers | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 500 | 14 | 0 | 0.0860 | 0.0946 | yes | 1337 |
| donna | 558 | 14 | 0 | 0.0681 | 0.1171 | yes | 1338 |
| io_son | 586 | 14 | 0 | 0.0580 | 0.1063 | yes | 1339 |
| solo_et_pensoso | 504 | 14 | 0 | 0.0655 | 0.0719 | yes | 1340 |
| line_start | 878 | 14 | 0 | 0.0444 | 0.1554 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Boundary Markers` counts `<|endoftext|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.
