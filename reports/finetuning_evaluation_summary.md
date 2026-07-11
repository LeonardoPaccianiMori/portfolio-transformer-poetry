# Fine-Tuning Evaluation Summary

This report evaluates the selected checkpoint from
`runs/finetuning_larger_20k_001`: local checkpoint `step_2000.pt`, selected as
the latest saved checkpoint at or before the best recorded validation step
2,500. It compares the selected model with the earlier sonnet-only BPE baseline
using the same five prompts, temperature 1.0, a 14-line target, and suppression
of the active boundary marker until the target is reached.

## Evidence

- Fine-tuning configuration and selection: `reports/finetuning_larger_20k_001.md`
- Fine-tuned generation metrics: `reports/generation_metrics_finetuning_larger_20k_001_step_2000.md`
- Fine-tuned memorization checks: `reports/memorization_checks_finetuning_larger_20k_001_step_2000.md`
- Assistant-authored first-pass review: `reports/qualitative_review_finetuning_larger_20k_001_step_2000.md`
- Sonnet-only BPE stop-suppression baseline: `reports/bpe_stop_suppression.md`

## Compared Systems

| System | Training Path | Parameters | Vocabulary | Context | Selected/Final Step |
| --- | --- | ---: | ---: | ---: | ---: |
| Sonnet-only BPE baseline | random initialization, sonnet corpus only | 0.28M | 512 | 256 | 5,000 |
| Selected fine-tuned model | 200k-step broader-Italian pretraining, then sonnet fine-tuning | 33.67M | 8,003 | 512 | 2,000 |

The systems are not a controlled causal ablation: they differ in pretraining,
parameter count, tokenizer vocabulary, context length, and training budget. The
comparison is useful as preliminary evidence, not proof that pretraining alone
caused any observed difference.

## Controlled Generation Metrics

| System | Avg Chars | Avg Lines | Avg Repeated 4-gram Ratio | Avg Unique-Character Ratio | Prompts Preserved | Boundary Markers in Outputs |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Sonnet-only BPE baseline | 499.4 | 14.0 | 0.0777 | 0.0770 | 5/5 | 0/5 |
| Selected fine-tuned model | 554.2 | 14.0 | 0.1072 | 0.0692 | 5/5 | 0/5 |

The 14-line result is decoder-enforced in both systems. It demonstrates that the
generation controls work; it does not demonstrate learned sonnet form.

## Memorization Check

All five selected fine-tuned samples have `low` risk under the current
40-character containment and longest-common-substring heuristic. Their longest
matched spans range from 13 to 17 normalized characters, below the 80-character
medium-risk threshold. This is a surface-copying diagnostic, not proof that the
model has no memorization risk.

## Qualitative Assessment

The selected fine-tuned samples contain more readable words, punctuation, and
Italian/archaizing surface texture than the small sonnet-only BPE baseline. They
are still not realistic sonnets: all reviewed samples received low coherence and
low sonnet-like-structure ratings. The primary failure is that plausible local
phrases do not form stable grammar, meaning, rhyme, or a coherent 14-line poetic
argument.

The fine-tuned samples also have a higher repeated-character-4-gram ratio than
the baseline. This is consistent with the qualitative observation that the model
often reuses function words, punctuation patterns, and local phrase shapes
without improving semantic control.

## Conclusion

The broader-pretrained model is a materially stronger language-modeling system
than the original educational BPE baseline, but this first fine-tuning result is
not yet a decent sonnet generator. The selected early checkpoint is preferable
to the overfit 20,000-step final checkpoint, yet it still requires a controlled
sonnet-only large-model comparison before the project can attribute gains to
broader pretraining.
