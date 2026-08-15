# Minerva 7B Quality-Recovery Diagnostic Protocol

Decision approved: 2026-08-09. Configuration frozen before recovery GPU
generation or judge scoring.

## Purpose

The completed project selected Minerva 7B Stage B epoch four as its strongest
model, but that adapter failed the fixed grammar and collapse thresholds. This
bounded extension diagnoses that failure before authorizing more data,
adaptation, preference optimization, or rented-GPU work. It does not erase or
reinterpret the completed final result.

The diagnostic asks two separate questions:

1. Did untouched Minerva, historical Stage A, or sonnet Stage B introduce the
   observed language defects?
2. Can an untouched Minerva 7B Instruct checkpoint agree with existing blinded
   human labels well enough to merit a definitive remote FP16 judge test?

No training occurs in this checkpoint. V6 final-test poems and openings remain
unavailable.

## Validation Prompt Lock

`configs/minerva_7b_quality_recovery_prompts.json` contains 12 previously
unused V6 validation poems. It uses 12 distinct authors across all five corpus
centuries, distributed 3/3/4/1/1 from the thirteenth through eighteenth
centuries. Runtime validation requires each opening to match the processed
validation poem exactly.

The set is disjoint from both the earlier eight-prompt Minerva validation audit
and the ten-prompt final acceptance set. File hashes and the V6 manifest hash
are frozen in `configs/minerva_7b_quality_recovery.json`.

## Lineage And Decoding Comparison

All generation loads the same pinned Minerva 7B Instruct revision in 4-bit NF4
on the local GPU. Quantization, prompts, task instruction, 512-token ceiling,
13-line continuation target, and seed 2029 remain fixed. The conditions are:

| Condition | Model state | Temperature | Top-k | Top-p | Repetition penalty |
| --- | --- | ---: | ---: | ---: | ---: |
| `untouched_control` | untouched parent | 0.80 | 50 | 1.00 | 1.00 |
| `stage_a_control` | historical step 4,000 | 0.80 | 50 | 1.00 | 1.00 |
| `stage_b_control` | sonnet epoch 4 | 0.80 | 50 | 1.00 | 1.00 |
| `stage_b_conservative` | sonnet epoch 4 | 0.65 | 40 | 0.92 | 1.05 |
| `stage_b_low_temperature` | sonnet epoch 4 | 0.55 | 30 | 0.90 | 1.05 |
| `stage_b_anti_repeat` | sonnet epoch 4 | 0.70 | 50 | 0.92 | 1.10 |
| `stage_b_nucleus` | sonnet epoch 4 | 0.70 | none | 0.90 | 1.05 |

The first three conditions isolate model lineage under identical decoding. The
remaining four isolate decoding while keeping Stage B fixed. Repetition
penalties apply only to tokens already generated in the continuation, not the
instruction or supplied opening. The complete set has 84 outputs.

Outputs are condition-blinded before qualitative review. Review records the
same three binary labels used previously: generally grammatical Italian,
seven-line topic continuity, and severe collapse. Automatic form and repetition
measurements remain diagnostics and cannot replace qualitative judgments.

Within each comparison, selection order is: highest grammatical count, lowest
collapse count, highest topic count, then lowest mean repeated-character
4-gram ratio. This diagnostic may recommend a later confirmation; it cannot
open final test or authorize training by itself.

## Untouched Minerva 7B Judge Preflight

The judge uses the untouched NF4 Instruct model and the 56 outputs whose labels
were completed blindly before the earlier Minerva 3B gate. It receives one
complete output and returns integer scores from zero to four for:

- `grammatica`: grammatical reliability;
- `tema`: sustained topic continuity;
- `stabilita`: absence of repetition collapse.

The response must be a JSON object with exactly those keys. A Markdown JSON
fence is tolerated but counted as parseable only when its contents satisfy the
exact schema. The model sees no condition ID or human label.

All five requirements must pass before the 7B judge can be used as a reward or
ranking signal:

| Check | Required |
| --- | ---: |
| Parseable responses | at least 0.98 |
| Grammar AUROC | at least 0.75 |
| Topic AUROC | at least 0.70 |
| Non-collapse AUROC | at least 0.75 |
| Human ordinal concordance | at least 0.65 |

The local result authorizes one remote FP16 confirmation only when parse rate
and non-collapse AUROC pass and at least four of the five total checks pass.
Otherwise the rented GPU VM remains off. Even a passing local result does not
authorize DPO, GRPO, or training; those require a separately frozen recipe and
a definitive FP16 judge result.

## Completion Criteria

This checkpoint is complete when:

- prompt, manifest, adapter, selection, and human-control hashes pass;
- all 84 generation outputs and metadata are written with flushed progress;
- a blinded review and automatic comparison report are complete;
- all 56 judge cases are scored or explicitly recorded as schema failures;
- the local judge gate and remote-authorization decision are reported;
- no final-test material or optimizer update has been used.
