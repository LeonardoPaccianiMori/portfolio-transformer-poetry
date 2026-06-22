# Decoding Experiments

Checkpoint: `runs/transformer_context768_scaled_001`

Prompt file: `configs/evaluation_prompts.json`

Base seed: `1337`

All settings used:

- `stop_text="<|poem_end|>"`
- `target_lines=14`
- `max_new_tokens=900`

Raw generation outputs are stored under ignored `outputs/generations/` directories. This report records the public comparison.

## Settings

| Setting | Output Directory | Temperature | Top-k |
| --- | --- | --- | --- |
| temp1_full | `outputs/generations/transformer_context768_scaled_001_controlled` | 1.0 | none |
| temp08_full | `outputs/generations/decoding_temp08_full` | 0.8 | none |
| temp08_topk20 | `outputs/generations/decoding_temp08_topk20` | 0.8 | 20 |
| temp07_topk20 | `outputs/generations/decoding_temp07_topk20` | 0.7 | 20 |
| temp06_topk10 | `outputs/generations/decoding_temp06_topk10` | 0.6 | 10 |

## Automatic Metrics

| Setting | Avg Chars | Avg Lines | Avg Repeat Ratio | Avg Unique Char Ratio | Total Separators | Prompts Preserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| temp1_full | 486.0 | 14.0 | 0.0212 | 0.0824 | 0 | 5/5 |
| temp08_full | 278.6 | 8.4 | 0.0285 | 0.1389 | 4 | 5/5 |
| temp08_topk20 | 327.2 | 9.8 | 0.0313 | 0.1184 | 4 | 5/5 |
| temp07_topk20 | 301.0 | 9.2 | 0.0494 | 0.1340 | 3 | 5/5 |
| temp06_topk10 | 148.6 | 4.8 | 0.0232 | 0.2000 | 5 | 5/5 |

## Stop Reasons

| Setting | amor | donna | io_son | solo_et_pensoso | line_start |
| --- | --- | --- | --- | --- | --- |
| temp1_full | target_lines | target_lines | target_lines | target_lines | target_lines |
| temp08_full | stop_text | stop_text | stop_text | stop_text | target_lines |
| temp08_topk20 | stop_text | stop_text | stop_text | stop_text | target_lines |
| temp07_topk20 | stop_text | target_lines | stop_text | stop_text | target_lines |
| temp06_topk10 | stop_text | stop_text | stop_text | stop_text | stop_text |

## Memorization

All settings remain low risk under the nearest-neighbor memorization proxy.

For every setting:

- maximum 40-character containment against the training split is `0.0000`;
- maximum longest common substring is 15 characters;
- every sample receives a `low` memorization-risk label.

## Qualitative Inspection

Lowering temperature and adding top-k makes the samples more conservative. This sometimes improves local word-like texture. For example, `temperature=0.6, top_k=10` produces strings such as:

```text
Amore ava da piassese al do canorte
che se sa me sorirani, se e ce stol en cole,
ch'an chi san degnoro a ase l l ca aro
```

This is still not coherent Italian, but it is less visually chaotic than the baseline `temperature=1.0` sample:

```text
Amorziavta?
Nè digrse aze siú so vesí cel viave si,
Orbive e sú begreco en colene smenttri.
```

The main tradeoff is early stopping. More conservative decoding makes `<|poem_end|>` more likely, so several samples stop before reaching 14 lines. The strongest case is `temperature=0.6, top_k=10`, where all five samples stop by `stop_text` and average only 4.8 non-empty lines.

`temperature=1.0, top_k=None` is best for enforcing 14-line output under the current rules. `temperature=0.8, top_k=20` gives a reasonable middle point: more word-like than the baseline in quick inspection, but still often stops early.

None of the decoding settings solves the core model-quality problem. The outputs remain mostly malformed at the word level and do not maintain semantic coherence.

## Conclusion

Decoding controls help with output shape and expose useful tradeoffs, but they do not make the current character model generate good sonnets.

Current recommendation:

1. Use `temperature=1.0, top_k=None, target_lines=14` when the goal is fixed-length structural comparison.
2. Use `temperature=0.8, top_k=20` as a qualitative sampling variant when inspecting whether more conservative decoding improves local texture.
3. Do not spend much more time tuning decoding for this checkpoint.
4. Move next to the BPE tokenizer track, because character-level tokenization appears to be a major source of unstable word formation.
