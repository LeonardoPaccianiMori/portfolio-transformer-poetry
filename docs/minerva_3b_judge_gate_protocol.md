# Minerva 3B Judge Gate Protocol

Decision authorized: 2026-08-08. Exact gate frozen: 2026-08-09, before GPU
scores were computed.

## Purpose

This gate tests whether the untouched `sapienzanlp/Minerva-3B-base-v1.0`
checkpoint can supply a defensible quality signal for the bounded DPO and GRPO
experiments. Passing this gate does not establish that Minerva is equivalent to
a human reviewer. It establishes only that its fixed score separates the
project's declared controls well enough to justify one experiment with each
post-training method.

The checkpoint is a base language model rather than a reliable instruction
follower. The gate therefore does not prompt it to print a subjective rating.
It computes the mean next-token negative log-likelihood of each thirteen-line
continuation conditioned on its exact opening line. The judge score is the
negative of that loss, so higher is better. The model is loaded in unquantized
FP16 with no adapter and no parameter updates.

## Validation-Only Cases

The fixed cases use the eight prompts in
`configs/minerva_3b_validation_sanity_prompts.json`. Every prompt belongs to the
V6 validation split. No final-test opening, poem body, or generated output may
be loaded by the gate.

For each prompt, the first control family contains:

- the genuine held-out sonnet;
- one output from the selected from-scratch PAISA-rescue task model, generated
  with seed 2027, temperature 0.8, top-k 50, and decoder-controlled thirteen
  continuation lines;
- a deterministic corruption of the genuine continuation that reverses word
  order independently within every line while preserving the opening,
  vocabulary, and fourteen-line count.

The second family contains all 56 validation-only outputs judged blindly in
`reports/minerva_3b_validation_sanity_blinded_judgments.md`. Their fixed grammar,
topic, and collapse labels were written before this gate was designed. Human
ordinal quality is `2 * grammar + topic + 2 * noncollapse`.

Authentic sonnets may have appeared in Minerva's external pretraining sources,
so genuine-versus-control accuracy alone is not sufficient. The independent
agreement checks against generated, human-labelled outputs are mandatory.

## Pass Thresholds

All six requirements must pass:

| Check | Required |
| --- | ---: |
| Genuine score above corrupted | at least 7/8 |
| Genuine score above from-scratch generated | at least 6/8 |
| From-scratch generated score above corrupted | at least 5/8 |
| Grammar AUROC on 56 human controls | at least 0.70 |
| Non-collapse AUROC on 56 human controls | at least 0.65 |
| Pairwise concordance with human ordinal quality | at least 0.65 |

Tied judge scores receive half credit. AUROC is computed directly from all
positive-negative pairs; no threshold is fitted to observed scores.

If any requirement fails, the judge gate fails and neither DPO nor GRPO is run.
The failure becomes the result of this phase. If all pass, the exact scoring
function may be used as one component of the separately frozen DPO ranking and
GRPO reward recipes. Mechanical prompt, line-count, repetition, memorization,
and parent-drift controls remain separate; likelihood is not allowed to stand
in for all aspects of sonnet quality.

The complete machine-readable lock is
`configs/minerva_3b_judge_gate.json`.

## Completed Result

The fixed gate ran on 2026-08-09 using the local RTX 3060 with the same FP16
weights split between GPU and CPU memory. It passed genuine-over-corrupted,
generated-over-corrupted, and grammar AUROC. It failed the other three checks:

- genuine-over-generated accuracy: `0.0000` against `0.7500` required;
- non-collapse AUROC: `0.1241` against `0.6500` required;
- human ordinal concordance: `0.4003` against `0.6500` required.

Mean NLL was `3.4274` for from-scratch generations and `3.8121` for genuine
sonnets, so every generated control outranked its authentic counterpart.
Collapsed human controls also received lower mean NLL (`2.0091`) than
non-collapsed controls (`2.7601`). This is a reward-misalignment failure, not a
borderline sample-size decision. Under the frozen policy, both DPO and GRPO are
cancelled. The public result is `reports/minerva_3b_judge_gate.md`.
