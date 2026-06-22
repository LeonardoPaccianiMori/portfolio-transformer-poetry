# BPE Transformer Comparison

This report compares the strongest character-token transformer run with the first
BPE-token transformer run.

## Runs Compared

| Run | Tokenizer | Vocab | Context | Batch | Steps | Emb | Layers | Heads | FF | Train Tokens | Val Tokens | Checkpoint MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `transformer_context768_scaled_001` | character | 97 | 768 chars | 8 | 5000 | 64 | 4 | 4 | 256 | 394952 | 49840 | 39.08 |
| `transformer_bpe_512_context256_001` | BPE | 512 | 256 BPE tokens | 16 | 5000 | 64 | 4 | 4 | 256 | 161150 | 20338 | 7.32 |

## Training Loss

| Run | Final Train | Final Val | Best Val | Best Step |
| --- | --- | --- | --- | --- |
| `transformer_context768_scaled_001` | 2.1727 | 2.1896 | 2.1896 | 5000 |
| `transformer_bpe_512_context256_001` | 3.9522 | 4.1658 | 4.1658 | 5000 |

The loss values are not directly comparable across tokenizers. Character loss is
computed over a 97-token vocabulary where each prediction is one character. BPE
loss is computed over a 512-token vocabulary where each prediction may be a
character, a word fragment, a whole short word, punctuation, or a special token.

The useful comparison is behavioral: generation quality, line control, repetition,
memorization risk, model size, and how much text each token covers.

## Controlled Generation Metrics

| Run | Prompt | Chars | Lines | Separators | Repeat Ratio | Memorization Risk |
| --- | --- | --- | --- | --- | --- | --- |
| character scaled | `amor` | 476 | 14 | 0 | 0.0190 | low |
| character scaled | `donna` | 459 | 14 | 0 | 0.0219 | low |
| character scaled | `io_son` | 469 | 14 | 0 | 0.0300 | low |
| character scaled | `solo_et_pensoso` | 539 | 14 | 0 | 0.0224 | low |
| character scaled | `line_start` | 487 | 14 | 0 | 0.0124 | low |
| BPE | `amor` | 362 | 14 | 0 | 0.0669 | low |
| BPE | `donna` | 97 | 4 | 1 | 0.0106 | low |
| BPE | `io_son` | 373 | 8 | 1 | 0.0378 | low |
| BPE | `solo_et_pensoso` | 57 | 2 | 1 | 0.0185 | low |
| BPE | `line_start` | 96 | 3 | 1 | 0.0108 | low |

## Qualitative Findings

The BPE run produces more recognizable word-level texture than the character
model. It creates outputs such as:

```text
Amore
del tua terzell'Amor co,
non posso futole.
Volgo gente.
```

This is a meaningful improvement over pure character noise because the tokenizer
lets the model predict larger recurring chunks of Italian-like text.

The BPE run is still not a good sonnet generator. Several controlled samples stop
early after producing `<|poem_end|>`, and the longer samples remain semantically
incoherent:

```text
Io soni posso ave'i' son su' sicuore,
cosa si ragione, al dìa
più tramentre nel lui senza asi prov'opiteco
```

The character model is better at obeying the current 14-line generation target.
That does not mean it understands sonnet form. The line target is enforced by the
generation script, and the character model's text is mostly subword character
noise.

## Interpretation

The BPE tokenizer is worth keeping. It reduces the sequence length needed to
represent the corpus and gives the model more useful prediction units than single
characters.

The first BPE transformer run should be treated as a tokenizer proof of concept,
not as the final model. The next modeling question is whether better decoding,
more training, a larger model, or explicit poem-structure handling improves BPE
generation without increasing memorization risk.

## Next Checkpoint

The next checkpoint should use this report to choose one controlled experiment:

1. Improve BPE decoding controls, especially early `<|poem_end|>` stopping.
2. Train the BPE model longer with the same architecture to check undertraining.
3. Increase the BPE model size modestly if the loss curve suggests capacity is
   limiting.

The recommended next experiment is decoding control first, because it is cheaper
than retraining and directly addresses the early-stop behavior visible in the
controlled samples.
