# Minerva 7B V7 Execution and Evidence Contract

Checkpoint 8F implements the CPU-verifiable reader, atomic checkpoint
contract, frozen activation probes, evidence-retention policy, transfer
bundle definition, and dual-H100 qualification entry point. GPU execution
remains unauthorized.

## Exact data execution

The reader reconstructs each 2,049-token source span directly from the
local signed-int32 shards, then forms 2,048 input and shifted-target
sequences. It consumes 16 frozen windows per optimizer update across
2,960 updates. No repacked corpus is created.

## Preserved evidence

Permanent compact telemetry contains one row for every optimizer update
(2,960 rows), plus approximately
49 evaluation events. It records loss,
pre-clipping gradient norm, learning rate, throughput, memory, exact window
identity, cost when available, validation/preservation results, and promotion
decisions. Stage midpoints and ends also retain compact per-module parameter,
gradient, optimizer-state, and allocator summaries. Fixed-probe logit summaries
and deterministic generations accompany the seven model states.

Full per-update gradients, optimizer copies, ordinary batch activations, and
unbounded attention tensors are deliberately not retained because their cost
would be disproportionate and the saved model states can reproduce probes.

## Activation probes

The ignored local manifest contains 48 exact probes:
12 modern instructions, 12 historical-general excerpts, 12 historical
non-sonnet-poetry excerpts, and 12 validation sonnets. It freezes token IDs,
masks, positions, all 32 block names, final normalization, pooling, BF16 local
capture, FP32 aggregation, top-20 logits, and a bounded raw-attention sample.
Its SHA-256 is `3557b4e455357ca165b4689a3876de7965ad59677cba6f0c0b00d2fad956488b`. V7 test data was not accessed.

## Checkpoint, transfer, and GPU boundary

Resume checkpoints install through a sibling temporary directory, fsync and
hash-verification, then atomic rename. The fresh-process proof compares exact
stage/update/window/LR/data/topology state and still requires a finite next
update. The private transfer bundle includes the training/validation shards,
indexes, preservation material, and probe manifest, but excludes V7 test data,
raw corpus caches, model weights, and prior runs.

After the checkpoint-8G qualification implementation fixes, the authoritative
V2 bundle contains 83 verified files / 283,524,173 bytes at SHA-256
`f31d987ddb1fc1dae3036c5e11f186430103764df6eb78c61ff1d2d3d9fbebca`.
It was installed only after its sibling temporary archive passed complete
manifest verification. The earlier V1 archive remains historical 8F evidence.

The original frozen dual-H100 qualification command is:

```bash
torchrun --standalone --nproc_per_node=2 scripts/qualify_minerva_7b_v7_full_weight.py
```

Checkpoint 8G subsequently tested a lower-cost dual-RTX-A6000 alternative with
an eight-candidate fail-closed matrix. Two gradient-checkpointed microbatch-one
candidates completed, but the best left only 2,283 MiB per GPU against the
frozen 8-GiB gate; the remaining six candidates OOMed. The host is rejected and
the measured evidence is published in
[`minerva_7b_v7_dual_a6000_qualification_v1.md`](minerva_7b_v7_dual_a6000_qualification_v1.md).
No V7 test access or long training occurred. A single-H100-SXM fallback requires
separate qualification, and the user will personally launch actual training.
Instance lifecycle actions and cache deletion remain unauthorized.
