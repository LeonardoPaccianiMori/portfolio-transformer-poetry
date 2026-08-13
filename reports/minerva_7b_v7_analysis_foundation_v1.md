# Minerva 7B V7 Post-Training Analysis Foundation

## Status

The GPU-free foundation and the bounded descriptive research suite are
implemented and CPU tested. They prepare a reproducible seven-state study of
the three-stage Minerva 7B curriculum without running post-hoc model inference,
accessing V7 test data, or authorizing causal interventions.

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

The descriptive suite adds:

- a manual-execution-only BF16 extractor for the frozen 48 probes, retaining
  raw hidden states, compact pooled/selected representations, attention
  summaries, bounded raw attention, and top-logit evidence;
- matched BF16 generation with a frozen 24-prompt by 3-seed grid per state,
  including exact conditioning and output token IDs;
- paired representation, attention, logit, embedding, and LM-head comparisons;
- a private, hash-pinned surface-copy reference decoded from exactly the 19,899
  V7 training sonnets, with no validation or test pool access;
- automatic behavioral summaries and a deterministic model-blinded human
  review scaffold; and
- hash-verified, relative-path completion records so private outputs remain
  portable between the GPU host and laptop.

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
and top-logit summaries projects to 3.51 GiB per state and 24.59 GiB across all
seven states. This corrected estimate counts 34 streams: embedding output, 32
block outputs, and final norm. It is a byte-level planning estimate before
filesystem and manifest overhead, not a measured GPU runtime or final artifact
size.

The private memorization-reference dry run verified the frozen document-index
and token-shard hashes, decoded all 19,899 `sonnets_train` records / 3,551,021
tokens, and wrote a 17.34-MB hash-pinned local export. The scorer bounds later
work by first building a generated-40-gram lookup and scanning the training
records once, then running exact longest-common-substring work only for
candidates sharing a generated 40-gram.

The frozen 24-prompt by 3-seed grid remains the confirmatory dataset. A larger,
separately labeled exploratory generation tier is now implemented so the
qualified GPU can produce substantially more samples without changing the
confirmatory contract. It freezes 120 validation-only openings, eight seeds,
and three predeclared conservative/balanced/creative decoding recipes: 2,880
outputs per state and 20,160 across all seven states. The prompt builder verifies
the validation document index, token shard, and tokenizer before deterministically
balancing historical period and limiting author/work concentration.

The high-volume runner uses batched BF16 cached decoding, one independent RNG
stream per output, hash-bound relative output paths, batch-level progress, and
safe resume. A separate one-batch qualification measures usable batch size,
throughput, runtime, storage, and cost before a full user-launched job. The
analysis reports per-state/recipe estimates and paired changes for all six
adjacent transitions plus parent-to-final, with 5,000 clustered bootstrap
resamples over prompts. A deterministic 504-output blinded subset covers 24
prompt clusters, all seven states, and all three recipes without exposing model
or recipe identities in the review sheet.

The GPU research package is intentionally independent of legacy corpus-builder,
scraping, and adapter-training imports. Its frozen chat prompt, line-completion
counter, sampling filters, and bounded memorization primitives live in the
research package itself. This keeps model-state analysis deployable beside the
minimal full-weight training bundle without installing unrelated acquisition
dependencies or changing the scientific prompt/sampling contract.

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

The focused research/analysis suite passes 32 tests, the adjacent training-
regression suite passes 47 tests, and the complete repository suite passes
1,110 tests. Real dry runs validate stage-1 state accounting, the 291-tensor
checkpoint layout, the frozen probe contract, 48-probe/72-output execution
plans, the 2,880-output-per-state high-volume plan, and the private training-
only memorization export without altering any training artifact.

No model weights, private probe token IDs, local run artifacts, raw activations,
V7 test material, or rented-instance details are published in this report.
