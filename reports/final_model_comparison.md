# Final Model Comparison

## Scope

This comparison closes the model-development phase. It contrasts the final
from-scratch system with the two adapted Minerva systems that received complete
20-output task evaluations.

The V5 and V6 final prompt set is preserved across these runs. V6 removes an
editorial non-sonnet and duplicate leakage without moving retained final-test
prompts, so the output counts below are directly useful for behavioral
comparison. The systems still differ substantially in parent scale, external
pretraining, tokenizer, and compute; the table is not a controlled architecture
ablation.

## System Lineage

| System | Parent scale | Updated parameters | Adaptation path | Training hardware |
| --- | ---: | ---: | --- | --- |
| PAISÀ historical from-scratch rescue | 70.1M | all 70.1M | PAISÀ pretraining, historical adaptation, V5 sonnet and task-format fine-tuning | local RTX 3060 6 GiB |
| Minerva 3B QLoRA | 2.89B | 26.2M rank-16 adapters | V5 opening-line continuation QLoRA | local RTX 3060 6 GiB |
| Minerva 7B staged LoRA | 7.4B | 6.82M rank-8 adapters | historical prose plus PAISÀ replay, then V6 instruction-formatted sonnets | remote RTX 8000 48 GiB |

The Minerva base weights were frozen. The 7B training path was unquantized FP16
LoRA; 4-bit NF4 is used only to make local demo inference fit the laptop GPU.

## Shared Acceptance Results

| System | Form | Grammar | Topic | Severe collapse | High-risk overlap | Full gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| From-scratch rescue | 20/20 | 0/20 | 0/20 | 20/20 | 0/20 | fail |
| Minerva 3B QLoRA | 20/20 | 2/20 | 13/20 | 7/20 | 0/20 | fail |
| Minerva 7B staged LoRA | 20/20 | 8/20 | 20/20 | 5/20 | 0/20 | fail |
| Required | at least 18 | at least 12 | at least 10 | at most 2 | exactly 0 | all checks |

`Form` means exact opening preservation and decoder-controlled fourteen-line
output. It does not establish metre or rhyme.

## What Improved

The progression is meaningful even though no model passed the complete gate:

1. The final from-scratch model learned the task interface and line production,
   but not reliable sentence-level Italian or semantic development.
2. Minerva 3B QLoRA retained topic in 13 outputs and reduced collapse from 20
   outputs to 7, showing clear transfer from external Italian pretraining.
3. Historical adaptation plus V6 specialization of Minerva 7B retained a topic
   in every output and produced generally grammatical Italian in 8 outputs.
4. None of the systems showed high-risk long-span overlap with its recorded
   sonnet training split under the project's surface-copying heuristic.

The strongest gain comes from pretrained model capacity and language knowledge,
not from a single transformer component. RMSNorm, RoPE, SwiGLU, schedule,
gradient-clipping, tokenizer, and corpus-scaling experiments were informative,
but none closed the language-quality gap in the small from-scratch branch.

## What Still Failed

The selected 7B adapter misses the grammar threshold by four outputs and exceeds
the collapse limit by three. Its failures often begin with plausible historical
diction and then lose clause structure or repeat an increasingly narrow frame.
This makes automatic line count and topic metrics necessary but insufficient.

The untouched Minerva 3B judge experiment also failed. Negative continuation
NLL detected grammar and obvious word-order corruption, but preferred every
from-scratch generation to its genuine-sonnet control and assigned higher
likelihood to collapsed human controls. DPO and GRPO were therefore cancelled
under the predeclared exit policy rather than optimized against a misaligned
reward.

## Compute And Data Efficiency

- The from-scratch rescue required hundreds of thousands of optimizer updates
  and multi-day local training, yet remained unable to generate coherent
  continuations reliably.
- Minerva 3B changed only 26.2M adapter parameters and learned visible task
  behavior in three selected epochs, but its broad adapter scope damaged syntax.
- Minerva 7B changed only 6.82M attention-adapter parameters. Stage A selected
  update 4,000; Stage B selected epoch 4/update 744. It delivered the strongest
  result with a much narrower trainable surface.

Data efficiency is not directly comparable: both Minerva systems inherit large
external pretraining corpora that are not part of the project's own token
budget. The comparison demonstrates practical transfer, not superiority under
equal data or compute.

## Final Decision

`minerva_7b_v6_selected_epoch_04` is the selected demonstration model because it
is the strongest validated system. It remains explicitly labelled experimental
and failed. No additional from-scratch, DPO, GRPO, or Minerva tuning run is
authorized by the project's exit policies.

The final project contribution is therefore the complete engineering and
evaluation pipeline: corpus provenance, from-scratch implementation, modern
component experiments, resumable training, pretrained adaptation, fixed prompt
evaluation, memorization checks, negative-result reporting, and a local demo.

## Evidence

- From scratch: `reports/sonnet_task_format_paisa_historical_rescue_v1_v5_best_evaluation.md`
- Minerva 3B: `reports/minerva_3b_base_vs_qlora_evaluation.md`
- Minerva 7B: `reports/minerva_7b_v6_final_evaluation.md`
- Judge gate: `reports/minerva_3b_judge_gate.md`
