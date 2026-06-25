# BPE Stop-Text Suppression Experiment

This report compares two controlled decoding runs from
`runs/transformer_bpe_512_context256_001`.

The experiment tests whether early `<|poem_end|>` generation is partly a decoding
problem. The model and checkpoint are unchanged.

## Decoding Settings

| Run | Temperature | Top-k | Stop Text | Target Lines | Suppress Stop Text Until Target Lines |
| --- | --- | --- | --- | --- | --- |
| baseline BPE controlled | 1.0 | none | `<|poem_end|>` | 14 | no |
| suppressed-stop BPE controlled | 1.0 | none | `<|poem_end|>` | 14 | yes |

In the suppressed-stop run, `<|poem_end|>` is forbidden while the generated text
has fewer than 14 completed non-empty lines. Once the generation reaches 14
completed lines, the generation script stops with `target_lines`.

## Automatic Metrics

| Run | Prompt | Chars | Lines | Separators | Repeat Ratio | Memorization Risk |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | `amor` | 362 | 14 | 0 | 0.0669 | low |
| baseline | `donna` | 97 | 4 | 1 | 0.0106 | low |
| baseline | `io_son` | 373 | 8 | 1 | 0.0378 | low |
| baseline | `solo_et_pensoso` | 57 | 2 | 1 | 0.0185 | low |
| baseline | `line_start` | 96 | 3 | 1 | 0.0108 | low |
| suppressed-stop | `amor` | 362 | 14 | 0 | 0.0669 | low |
| suppressed-stop | `donna` | 557 | 14 | 0 | 0.0830 | low |
| suppressed-stop | `io_son` | 578 | 14 | 0 | 0.0730 | low |
| suppressed-stop | `solo_et_pensoso` | 473 | 14 | 0 | 0.0894 | low |
| suppressed-stop | `line_start` | 527 | 14 | 0 | 0.0763 | low |

## Findings

Suppressing the stop token fixes the early stopping issue for this run. In the
baseline BPE controlled generation, only one of five samples reached 14 lines.
With stop-text suppression, all five samples reached 14 lines.

This does not make the model good at sonnet generation. The text remains locally
word-like but semantically incoherent:

```text
Donnafo qual delate,
I dopassi ne la beggior ve: — Queste svel!
Sì Be; né llor val—
```

The suppressed-stop outputs are also more repetitive by the current repeated
character 4-gram heuristic. This is expected: forcing generation to continue
after the model wanted to emit `<|poem_end|>` can expose weak long-range modeling.

## Interpretation

The experiment separates two issues:

1. **Decoding control issue:** early `<|poem_end|>` can be prevented with a
   simple logit mask.
2. **Model quality issue:** preventing early stop does not create coherent
   syntax, semantics, or sonnet argument structure.

The next modeling decision should not be more stop-token tuning. The recommended
next checkpoint is to inspect whether the BPE model is undertrained by running a
longer same-architecture BPE training experiment, then comparing validation loss,
controlled generation, repetition, and memorization risk.
