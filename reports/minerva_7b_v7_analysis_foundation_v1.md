# Minerva 7B V7 Post-Training Analysis Foundation

## Status

The GPU-free analysis foundation is implemented and CPU tested. It prepares a
reproducible seven-state study of the three-stage Minerva 7B curriculum without
running post-hoc model inference, accessing V7 test data, or authorizing causal
interventions.

## Scientific Contract

The longitudinal axis contains the untouched parent plus midpoint and
validation-selected states from each of the three stages. The primary
comparisons are the six adjacent transitions and the cumulative parent-to-final
transition. State audits distinguish an expected missing future state, a
partially transferred directory, and an internally or cross-lineage invalid
state. Only complete, manifest-valid states are comparison-ready.

The foundation provides:

- immutable state and comparison registries with cross-stage lineage checks;
- tolerant parsing of an active JSONL writer's incomplete final line, with
  rejection of conflicting duplicate or non-contiguous update evidence;
- authoritative CSV/JSON exports for training, evaluation, gate, memory,
  throughput, timing, and cost evidence;
- chunked SafeTensors parameter-change planning and execution without
  materializing a complete model delta;
- revalidation and resource planning for the frozen 48-probe activation
  contract; and
- a fail-closed causal-proposal gate.

## Measured Dry-Run Bounds

The real stage-1 telemetry is recognized as exactly 2,065 of 2,065 updates. A
metadata-only scan of one retained Minerva state identifies 291 tensors and
13.78 GiB of SafeTensors input. With the default 64-MiB chunk request, the
largest projected arithmetic working set is 384 MiB. The code permits at most
one left and one right input chunk at a time, uses FP32 tensor arithmetic with
FP64 scalar reductions, and never constructs a full 7.4-billion-parameter
delta tensor.

The frozen 48-probe manifest passes its exact SHA-256, token, attention-mask,
selected-position, four-domain balance, and no-test checks. Retaining local BF16
hidden states plus the approved compact FP32 aggregates, bounded raw attention,
and top-logit summaries currently projects to approximately 23.89 GiB across
all seven states. This is an intentionally conservative planning estimate, not
a measured GPU runtime or final artifact size.

## Causal-Experiment Boundary

Parameter or representation drift is descriptive evidence, not causal proof.
Layer restoration, stage-delta removal, interpolation, and ablation are not
authorized by this checkpoint. A later causal proposal must cite a descriptive
finding, predeclare an intervention and negative control, predict adaptation
and preservation effects, freeze state comparisons, domains, and primary
metrics, preserve V7 test isolation, and set a stopping rule. Passing that
structural gate still records `execution_authorized=false` until the user gives
separate approval.

## Verification

The focused analysis suite passes 12 tests. The complete repository suite
passes 1,090 tests. Real integration dry runs validate the stage-1 evidence,
the 291-tensor checkpoint layout, and the frozen probe contract without hashing
an incomplete local transfer or altering any training artifact.

No model weights, private probe token IDs, local run artifacts, raw activations,
V7 test material, or rented-instance details are published in this report.
