# Minerva-Guided DPO And GRPO Policy

Decision approved: 2026-08-08.

Status update: **completed negative result.** The bounded Minerva repair
programme completed, then the mandatory judge gate failed. Under this policy,
neither DPO nor GRPO is authorized. See
[`minerva_3b_judge_gate_protocol.md`](minerva_3b_judge_gate_protocol.md) and
[`minerva_repair_policy.md`](minerva_repair_policy.md).

## Purpose And Order

The corpus-only from-scratch track remains a completed experimental result. The
project has finished and evaluated the fixed Minerva 3B QLoRA comparison. The
validation-only diagnostic in
[`minerva_sanity_audit.md`](minerva_sanity_audit.md) completed without a
qualifying QLoRA adapter strength. The later repair-policy experiment is now
complete: the selected Minerva 7B V6 adapter passed structure, topic, and
memorization controls but missed the grammar and collapse thresholds. The
project then tested whether the untouched Minerva 3B base model was reliable
enough to guide two bounded post-training methods. The mandatory gate failed,
so no DPO or GRPO training followed.

Had they been authorized, these experiments would not have replaced or
retroactively redefined the corpus-only from-scratch result. They would have
been reported as Minerva-guided post-training of a model whose weights were
originally trained from scratch.

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
selection rules would have required a separate predeclared design checkpoint
if the judge gate passed. The policy did not silently choose those unmeasured
implementation details.

The fixed gate subsequently failed three of six requirements. In particular,
the score preferred every from-scratch generation to its authentic sonnet and
strongly preferred human-labelled collapsed outputs to non-collapsed outputs.
This makes the signal unsuitable for optimization. DPO and GRPO are therefore
cancelled rather than run with a reward that points in the wrong direction.
