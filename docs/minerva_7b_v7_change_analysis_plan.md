# Minerva 7B V7 Change-Analysis Plan

## Purpose

After the full-weight Minerva 7B curriculum is complete, this study will ask
what each stage changed rather than judging only the final poems. The stage
names are:

1. historical-general adaptation;
2. historical non-sonnet poetry adaptation;
3. sonnet specialization.

This plan does not authorize the later analysis run. It freezes what evidence
must survive training so that the study remains possible and reproducible.

## Model states to preserve

Use seven states:

1. untouched Minerva parent at the pinned published revision;
2. stage-1 midpoint;
3. validation-selected stage-1 endpoint;
4. stage-2 midpoint;
5. validation-selected stage-2 endpoint;
6. stage-3 midpoint;
7. validation-selected stage-3 endpoint.

The six new model-only snapshots use BF16 safetensors and immutable manifests,
without optimizer state. At 7,399,542,784 parameters, one raw BF16 state is
about 14.8 GB and six are about 88.8 GB before filesystem or compression
overhead. The untouched parent is referenced from its pinned cached revision
instead of needlessly duplicated. Resume checkpoints are a separate artifact:
they include optimizer, scheduler, RNG, sampler, and counters and exist for
crash recovery, not for most analyses.

Each snapshot manifest records the stage and update, parent/preceding snapshot,
data and protocol hashes, validation and preservation metrics, every weight-file
hash, source commit, package versions, and hardware topology.

Checkpoint 8I implements this as stage-scoped single-H100 lineage. A selected
endpoint is both a permanent analysis state and the only valid parent boundary
for the following stage. The boundary carries the untouched-parent baseline,
selected metrics, complete validation history, protocol/data hashes, launch
contract hash, and preceding-state identity. Within-stage resume checkpoints
also retain preservation and early-stopping selection state, so an interruption
cannot silently change which endpoint would be promoted.

Checkpoint 8F also retains compact process evidence that weights cannot
reconstruct by themselves. One permanent JSONL row per optimizer update records
the exact window-range digest, loss, pre-clipping gradient norm, learning rate,
throughput, elapsed/estimated time, per-rank memory, and cumulative cost when a
provider rate is available. Every evaluation retains per-pool and per-instruction
losses plus promotion decisions. Stage midpoints and validation-selected endpoints
retain per-module parameter, gradient, optimizer-state, and allocator summaries.
These summaries store norms, shapes, dtypes, maxima, and sparsity—not full gradient
or optimizer copies.

## Frozen probe design

Before the long run, create a held-out probe suite with 8–16 examples from each
of four domains:

- modern instruction following;
- historical general Italian;
- historical non-sonnet poetry;
- standard sonnets.

Freeze the source split and identity, exact token IDs and attention mask,
selected token positions, tokenizer hash, model revision, module/layer names,
extraction dtype and pooling, and random seed. Validation material may supply
the probes. V7 test sonnets remain unopened until the final stage-3 checkpoint
has been selected.

The frozen suite uses 12 probes per domain / 48 total. Historical-general
validation contains only 11 documents and only nine long enough for the bounded
probe policy, so that domain uses nine documents plus three additional disjoint
excerpts from those held-out documents. Historical non-sonnet poetry and standard
sonnets use 12 distinct held-out documents each; modern instructions use all 12
preservation prompts. This limits claims about work-level diversity in the
historical-general activation results.

## Behavioral and loss analysis

Evaluate all seven model states with identical inputs and settings:

- loss on every fixed V7 broader and sonnet validation pool;
- teacher-forced loss on the PAISÀ and instruction-preservation sets;
- identical fixed generation prompts and decoding parameters;
- automatic form, collapse, topic, and memorization controls;
- blinded human review without revealing checkpoint identity.

This separates intended specialization from regression. For example, stage 2
should improve held-out non-sonnet poetry without erasing stage-1 historical
behavior, and stage 3 should improve held-out sonnets without unacceptable
modern-instruction or broader-literary damage.

## Weight-delta analysis

For every adjacent pair and every state versus the untouched parent, compute:

- absolute and relative L2/Frobenius weight-delta norms by layer and module;
- cosine similarity between flattened parameters and between update vectors;
- update-to-weight ratios;
- attention, MLP, embedding, and normalization update concentration;
- singular-value spectra or low-rank approximations of weight deltas;
- correlations between layerwise change and behavioral/loss change.

These measurements show where optimization spent its capacity. They do not by
themselves prove a causal mechanism, so interpretations must be paired with the
behavioral and activation evidence.

## Activation and attention analysis

Run the frozen probes post hoc with every model in `eval()` mode. Capture the
embedding output and every transformer-block residual output at preselected
token positions. Do not archive activations from ordinary training batches:
their changing inputs and training-mode effects make comparisons less clean,
and storage would grow without a clear scientific benefit.

Planned activation measurements are:

- linear CKA between model states, layer by layer;
- cosine similarity and relative activation-norm shift;
- singular values and effective rank;
- frozen-split linear probes for domain separability;
- next-token logit and probability shifts for archaic words, verse endings,
  rhyme anchors, and instruction responses.

Attention is quadratic in sequence length. Store per-head entropy and attention-
distance summaries for the full probe suite and raw attention matrices only for
a small bounded sample. Raw hidden-state and attention tensors remain local.

Each frozen state also produces compact top-20 token/logit summaries, log-sum-exp,
and entropy at the selected positions. The fixed generation prompt set and decoding
seed are retained so deterministic behavioral samples can be regenerated before a
remote instance is released. These outputs are small enough to preserve alongside
the permanent telemetry and evaluation history.

## Reproducible artifact structure

The authoritative analysis should be implemented as deterministic Python CLIs
that emit JSON/CSV and plots. A Jupyter notebook may then load those frozen
outputs for explanation, visual comparison, and portfolio presentation. The
notebook must not be the only place where metrics are calculated.

Planned later artifacts are:

- a snapshot and probe manifest;
- a behavioral/loss comparison CLI and JSON/CSV report;
- a weight-delta CLI and per-layer table;
- an activation-extraction CLI writing bounded local tensors;
- an activation-comparison CLI writing aggregate matrices and plots;
- one explanatory notebook consuming those outputs;
- a technical report that distinguishes observation, correlation, and causal
  inference.

## Acceptance criteria

The study is complete when all seven states are hash-pinned, all measurements
use the same frozen probes, raw tensors remain local, aggregate results are
reproducible from scripts, and the report can state which changes were gradual,
which appeared at stage boundaries, which abilities improved, and which
regressed. Results must not be described as mechanistic causality unless a
separate intervention supports that claim.
