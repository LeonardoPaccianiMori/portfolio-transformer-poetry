# Historical Italian V2 XXL Pretraining Plan

## Purpose

Train the quality-focused from-scratch parent model on
`pretraining_historical_italian_v2`. The run uses the committed 16,000-token
pretraining tokenizer and the reproducible local token tensors described in
[`pretraining_historical_italian_v2_encoded_report.json`](../reports/pretraining_historical_italian_v2_encoded_report.json).

## Model And Optimization

- Vocabulary: 16,000 tokens.
- Architecture: 1,024 embedding dimensions, 16 transformer blocks, 16 heads,
  64 dimensions per head, and 2,731-wide SwiGLU blocks.
- Position encoding: learned absolute positions with a maximum context length
  of 512 tokens.
- Normalization: LayerNorm with epsilon `1e-5`.
- Token embeddings: untied from the output projection.
- Microbatch: 1 sequence of 512 next-token targets.
- Gradient accumulation: 8 microbatches per optimizer update, for 4,096
  target-token exposures per update without increasing activation-memory use.
- Optimizer updates: 100,000, for 409,600,000 total target-token exposures.
- Learning rate: warmup-cosine schedule from a 1,000-update warmup to a
  `3e-4` peak and `3e-5` minimum.
- Validation: all deterministic non-overlapping 512-token windows from the
  held-out stream every 1,000 updates.

The model has 234,839,008 parameters and was the largest successful FP32
candidate on the local RTX 3060 Laptop GPU. The benchmarked 323M-parameter
ceiling candidate ran out of memory.

## Checkpoints And Selection

`best_validation.pt` is replaced whenever validation loss improves and is the
parent-selection artifact for later sonnet fine-tuning. It intentionally omits
AdamW state to reduce repeated disk writes.

`resume.pt` is atomically replaced every 5,000 updates and retains optimizer
state. It is the only interval checkpoint, preventing a multi-day run from
accumulating many multi-gigabyte checkpoint files.

## Start Command

```bash
python3 scripts/train_pretraining.py \
  --output-dir runs/pretraining_historical_italian_v2_xxl_accum8_100k_001 \
  --device auto \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --context-length 512 \
  --max-context-length 512 \
  --train-steps 100000 \
  --eval-interval 1000 \
  --eval-batches 1 \
  --validation-mode sequential_windows \
  --learning-rate 3e-4 \
  --learning-rate-schedule warmup_cosine \
  --warmup-steps 1000 \
  --min-learning-rate 3e-5 \
  --seed 1337 \
  --embedding-dim 1024 \
  --num-layers 16 \
  --num-heads 16 \
  --head-dim 64 \
  --feed-forward-dim 2731 \
  --feed-forward-type swiglu \
  --normalization-type layer_norm \
  --normalization-eps 1e-5 \
  --position-encoding-type learned_absolute \
  --checkpoint-interval 5000 \
  --checkpoint-retention latest_only \
  --progress-interval 100
```

## Resume Command

Run the same command with this additional argument:

```bash
  --resume-from-checkpoint runs/pretraining_historical_italian_v2_xxl_accum8_100k_001/resume.pt
```

The saved configuration, loss history, and best checkpoint remain in the same
output directory. The runner verifies the dataset report and all artifact
compatibility before either a fresh or resumed run begins.
