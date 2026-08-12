# Minerva 7B V7 Single-H100 Launch Readiness

Checkpoint 8I enables user-controlled, stage-scoped full-weight BF16 training
on the checkpoint-8H qualified single H100. It does not itself start training,
open V7 test material, delete caches, or authorize instance lifecycle actions.

## Qualified launch contract

The immutable checkpoint-8H execution configuration remains unchanged at
SHA-256 `6be2481b171c880224b62fbe1c65c6a8bcf2946463b5b0b12275c51db745e36a`.
The new single-H100 launch contract references that file, the scientific
protocol, and the passed qualification report by exact hash. It permits only
the selected runtime: one H100, context 2,048, microbatch one, accumulation 16,
no gradient checkpointing, and `torch.compile` default. This preserves the
scientific batch of 16 windows / 32,768 target tokens per optimizer update.

Each invocation runs exactly one named stage. Stage 1 may start only from the
pinned untouched parent. Stages 2 and 3 require the hash-verified validation-
selected endpoint of the immediately preceding stage. A within-stage resume
must name an atomic checkpoint for the same stage and launch contract; it
restores model, optimizer, scheduler, RNG, sampler position, validation history,
preservation counters, and early-stopping selection state.

## Evidence retained for later study

The trainer keeps the full frozen change-analysis contract:

- six permanent model-only BF16 snapshots: midpoint and validation-selected
  endpoint for each of the three stages;
- permanent per-update telemetry and per-evaluation metrics;
- sparse parameter, gradient, optimizer-state, and allocator summaries;
- exact protocol, data, parent/boundary, source-commit, package, and hardware
  lineage in snapshot manifests;
- the 48 frozen validation-only activation probes for later post-hoc analysis;
- the newest two atomic resume generations.

Raw activations are deliberately recomputed later from the seven frozen model
states. Ordinary changing-batch activations are not archived.

## VM verification

The private checkpoint-8I archive contains 89 verified payload files and
283,534,633 bytes at SHA-256
`c6aad4fa87a90e7080e13755dacad914cd3813d22660a2c595dde7262b5adddc`.
The VM whole-file hash and every embedded member passed. V7 test material is
absent. The no-model/no-training stage-1 preflight passed with 334.5 GiB free
and a conservative 145.3 GiB remaining evidence budget. Stage status reported
all three stages unstarted.

The projected stage durations and costs remain 2.84 h / `$7.43`, 1.05 h /
`$2.74`, and 0.19 h / `$0.49` under the frozen 1.25x allowance. These remain
qualification-based estimates; full validation, compilation, and snapshot I/O
can increase actual runtime.

## Verification boundary

Focused launch/bundle/trainer/execution tests pass 42/42. The complete suite
passes 1,077/1,077. The user remains the sole owner of actual training launches.
No training command was run during checkpoint 8I.
