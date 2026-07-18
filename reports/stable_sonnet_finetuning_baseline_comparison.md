# Stable Sonnet Fine-Tuning Baseline Comparison

This report compares the selected SwiGLU fine-tuning run with the earlier
stable-protocol ReLU baseline under matched prompts, seeds, and decoding
settings. It is a descriptive comparison, not a controlled architecture
ablation.

## Shared Evaluation Conditions

Both runs use five fixed prompts, seeds 1337 through 1341, temperature 1.0, no
top-k restriction, a 900-token safety limit, and decoder-enforced 14-line
stopping with `<|endoftext|>` suppression.

## Results

| Measurement | ReLU Stable Baseline | Selected SwiGLU |
| --- | ---: | ---: |
| Feed-forward block | ReLU, width 2,048 | SwiGLU, width 1,365 |
| Parent checkpoint | step 200,000 final parent | step 110,000 selected parent |
| Fine-tuning selected step | 1,250 | 1,000 |
| Validation windows | 47 | 43 |
| Best recorded validation loss | 2.7381 | 2.9771 |
| Prompt preservation | 5 / 5 | 5 / 5 |
| Decoder-enforced 14 lines | 5 / 5 | 5 / 5 |
| Average generated characters | 550.4 | 521.0 |
| Average repeated character-4-gram ratio | 0.0971 | 0.0846 |
| Low memorization-risk labels | 5 / 5 | 5 / 5 |
| Maximum normalized longest common span | 17 | 17 |

## Interpretation

The SwiGLU samples have lower average local repetition under the matched
generation settings. However, both sets of samples are rated low for
sonnet-like structure, language plausibility, and coherence. Neither produces
reliably grammatical Italian or a sustained poetic argument.

The validation losses must not be compared as a direct score. The ReLU baseline
used 47 sequential validation windows, while the SwiGLU run used 43 after the
sonnet corpus changed. The parent checkpoints also differ in pretraining corpus
version, parent selection point, and feed-forward architecture. These changes
prevent a one-factor causal conclusion.

## Conclusion

The comparison does not justify choosing one fine-tuning architecture as a
quality winner. It does show that neither existing parent and fine-tuning setup
is sufficient for realistic sonnet generation. The next experiment should be a
documented corpus-mixture or curriculum change, evaluated with the stable
protocol, rather than another unstructured hyperparameter change.
