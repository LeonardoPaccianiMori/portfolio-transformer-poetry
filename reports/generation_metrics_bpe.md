# Generation Metrics

Generation directory: `outputs/generations/transformer_bpe_512_context256_001_controlled`

| Prompt | Chars | Lines | Separators | Unique Chars | Repeat Ratio | Prompt Kept | Seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| amor | 362 | 14 | 0 | 0.0884 | 0.0669 | yes | 1337 |
| donna | 97 | 4 | 1 | 0.3711 | 0.0106 | yes | 1338 |
| io_son | 373 | 8 | 1 | 0.0992 | 0.0378 | yes | 1339 |
| solo_et_pensoso | 57 | 2 | 1 | 0.3684 | 0.0185 | yes | 1340 |
| line_start | 96 | 3 | 1 | 0.3021 | 0.0108 | yes | 1341 |

## Notes

- `Lines` counts non-empty lines.

- `Separators` counts `<|poem_end|>` occurrences.

- `Repeat Ratio` is based on repeated character 4-grams by default.

- These are basic automatic checks, not a full quality evaluation.

