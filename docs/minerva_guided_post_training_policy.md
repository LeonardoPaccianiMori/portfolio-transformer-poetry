# Minerva-Guided DPO And GRPO Policy

Decision approved: 2026-08-08.

Status update: **current after completion of the bounded Minerva repair
programme.** The judge gate remains mandatory before DPO or GRPO. See
[`minerva_repair_policy.md`](minerva_repair_policy.md). No final-test isolation
or branch-independence rule below is changed.

## Purpose And Order

The corpus-only from-scratch track remains a completed experimental result. The
project has finished and evaluated the fixed Minerva 3B QLoRA comparison. The
validation-only diagnostic in
[`minerva_sanity_audit.md`](minerva_sanity_audit.md) completed without a
qualifying QLoRA adapter strength. The later repair-policy experiment is now
complete: the selected Minerva 7B V6 adapter passed structure, topic, and
memorization controls but missed the grammar and collapse thresholds. The
project may now test whether feedback from the untouched Minerva 3B base model
improves the selected from-scratch model through two bounded post-training
methods: DPO and GRPO.

These experiments do not replace or retroactively redefine the corpus-only
from-scratch result. They must be reported as Minerva-guided post-training of a
model whose weights were originally trained from scratch.

## Required Judge Gate

The judge is the untouched
`sapienzanlp/Minerva-3B-base-v1.0` revision
`129ae5366bae3611a1c9f8c68606c38b7de8b055`, with no sonnet adapter active.

Before DPO or GRPO training, a fixed judge protocol must test whether Minerva
can distinguish and rank:

- genuine sonnets;
- outputs from the selected from-scratch model;
- deliberately corrupted sonnets;
- examples already assessed under the project's human qualitative rubric.

Judge development may use training and validation prompts only. The fixed final
test openings and their outputs remain unavailable until final evaluation. The
judge protocol and agreement thresholds must be recorded before its results
are examined. If the gate fails, neither DPO nor GRPO runs; the failure becomes
the result of this project phase.

## Independent Branches

If the gate passes, both branches initialize independently from:

```text
runs/sonnet_task_format_paisa_historical_rescue_v1_v5_12k_001/best_validation.pt
```

The DPO branch uses a fixed dataset of Minerva-ranked preferred and rejected
continuations. The GRPO branch uses group-relative Minerva rewards while
retaining a supervised next-token-loss component and a penalty against drifting
too far from the shared parent.

The DPO model must not initialize the GRPO model, and the GRPO model must not
initialize the DPO model. This preserves attribution: any difference can be
associated with the post-training method rather than with sequential training.

## Experiment Boundaries

- Permit hardware and memory calibration before training.
- Freeze each complete recipe and compute budget before its full run.
- Run exactly one full DPO experiment and one full GRPO experiment.
- Do not perform a result-driven hyperparameter, reward, decoding, or checkpoint
  search.
- Use identical final prompts, seeds, decoding controls, memorization checks,
  and qualitative criteria for the shared parent, DPO, and GRPO outputs.
- Compare all three against the separately fine-tuned Minerva QLoRA model.

The exact judge score, validation-only cases, and pass thresholds are now
frozen in [`minerva_3b_judge_gate_protocol.md`](minerva_3b_judge_gate_protocol.md).
Candidate counts, DPO and GRPO hyperparameters, compute budgets, and checkpoint
selection rules still require a separate predeclared design checkpoint, and
only if the judge gate passes. This policy authorizes those two bounded
branches; it does not silently choose their unmeasured implementation details.
