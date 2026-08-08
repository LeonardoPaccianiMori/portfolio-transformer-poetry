# Fixed Minerva 3B QLoRA Sonnet Comparison

This is the single pretrained-model comparison experiment. It is distinct from
the completed from-scratch track: Minerva already learned broad Italian and
English language patterns during its original pretraining, while this project
trains only small LoRA adapters on the curated V5 sonnet task.

## Task Data

Each V5 sonnet becomes one document-level example:

```text
<exact held or training opening line>\n
<remaining thirteen original lines>\n
```

The opening line and following newline are model input only. Their labels are
set to `-100`, so the loss begins at the first token of line two. This directly
matches the user-facing task: continue a supplied first line into the remaining
thirteen lines. It intentionally uses ordinary Minerva tokenizer text rather
than adding fresh control tokens, because QLoRA keeps the original token
embedding matrix frozen.

Measured with Minerva's tokenizer, all 1,875 V5 examples fit the locked
512-token context: maximum 253 tokens and mean 182.23 tokens.

## Fixed Recipe

| Setting | Value |
| --- | --- |
| Base model | `sapienzanlp/Minerva-3B-base-v1.0` at revision `129ae5366bae3611a1c9f8c68606c38b7de8b055` |
| Base-model storage | 4-bit NF4 with double quantization and float16 computation |
| Trainable weights | Rank-16 LoRA adapters on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` |
| Context / microbatch / accumulation | 512 / 1 / 8 |
| Epoch ceiling | 20 full passes over the 1,486 V5 training examples |
| Learning rate | `1e-4`, 5% linear warmup, cosine decay to `1e-5` |
| Validation | Full document-level validation split after each epoch |
| Selection | Minimum meaningful validation improvement `0.01`; patience 3 |
| Gradient clipping | Global norm `1.0` |
| Checkpoints | Adapter-only best, final, and resumable checkpoints |

The run has 186 optimizer updates per epoch and a 3,720-update ceiling. It is
one fixed comparison, not a QLoRA hyperparameter search.

## Artifacts

`scripts/train_minerva_qlora.py` creates a run directory containing
configuration metadata, per-epoch loss history, `best_adapter.pt`,
`final_adapter.pt`, and `resume.pt`. The frozen Minerva weights are never copied
into those checkpoints. Progress is flushed every 25 updates, full validation
is reported after each epoch, and `resume.pt` is rewritten after every epoch.
It includes the optimizer and random-generator state, so an interrupted run
continues with the same remaining example order. The resume path itself is not
part of the locked recipe identity; all actual modeling choices remain locked.

## GPU Run

Run this command from the repository root using the project virtual
environment. It is GPU-only. It will print one progress line every 25 optimizer
updates and a complete validation result at the end of each epoch:

```bash
.venv/bin/python scripts/train_minerva_qlora.py \
    --output-dir runs/minerva_3b_qlora_v5_001 \
    --device cuda:0
```

The run has a maximum of 3,720 updates, but early stopping may finish it after
four or more epochs. The initial ETA printed after 25 updates is the reliable
machine-specific estimate; it includes 4-bit computation, gradient
checkpointing, and full validation.

After selection, the same fixed ten held-out openings, two seeds, controlled
14-line stopping, memorization checks, and qualitative rubric used for the
from-scratch model will be adapted for the Minerva adapter.
