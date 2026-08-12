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

After adding the checkpoint-8H single-H100 qualification profile and launcher,
the authoritative V2 bundle contains 85 verified payload files / 283,525,043
bytes at SHA-256
`73b038ac622fdac2ab387db2cdd062345adc29f6c639919f2bc130cfa48b0777`.
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
No V7 test access or long training occurred. Checkpoint 8H therefore tested the
single-H100-SXM fallback separately; the user still personally launches actual
training. Instance lifecycle actions and cache deletion remain unauthorized.

Checkpoint 8H subsequently qualified one H100 80GB HBM3 through a 12-candidate
matrix. The selected microbatch-one / accumulation-16 / no-checkpointing /
compile runtime measured 8,273.6 target tokens/s with 24,666 MiB reserved-memory
headroom. It also passed held-out validation transition, atomic checkpoint
installation, fresh-process RNG/sampler/LR restore, and a finite next update.
The public evidence is in
[`minerva_7b_v7_single_h100_qualification_v1.md`](minerva_7b_v7_single_h100_qualification_v1.md).
Qualification did not authorize long training: the single-GPU trainer and its
snapshot/monitoring contract must be adapted and verified first, after which
the user will personally launch stage 1.
