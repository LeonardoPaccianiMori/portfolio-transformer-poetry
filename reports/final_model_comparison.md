# Final Model Comparison

## Scope

This comparison closes model development across the final from-scratch system,
Minerva 3B QLoRA, the earlier Minerva 7B V6 staged LoRA, the full-weight V7
Stage-3 model, and its bounded AI-judged DPO adapter.

The older systems used the V5/V6 20-output acceptance evaluation. V7 uses a
larger author/work-isolated validation and one-time V7 test, so its metrics are
reported separately rather than forced into the older table. Model scales,
external pretraining, tokenizers, datasets, and compute differ; this is a
behavioral project comparison, not a controlled architecture ablation.

## System Lineage

| System | Parent scale | Updated parameters | Adaptation path | Hardware |
| --- | ---: | ---: | --- | --- |
| PAISÀ historical from-scratch rescue | 70.1M | all 70.1M | PAISÀ, historical, V5 sonnet, task-format | RTX 3060 6 GiB |
| Minerva 3B QLoRA | 2.89B | 26.2M LoRA | V5 continuation QLoRA | RTX 3060 6 GiB |
| Minerva 7B V6 staged LoRA | 7.4B | 6.82M LoRA | historical prose/replay, V6 sonnets | RTX 8000 48 GiB |
| Minerva 7B V7 Stage 3 | 7.4B | full model | historical, poetry, V7 sonnets | H100 80 GB |
| Minerva 7B V7 DPO | 7.4B | 20.0M rank-8 LoRA parameters | Stage 3 plus AI-judged preferences | H100 80 GB |

All authoritative V7 training and evaluation used unquantized BF16. The local
demo's 4-bit NF4 loading is a deployment approximation only.

## Legacy Shared Acceptance Results

| System | Form | Grammar | Topic | Severe collapse | High-risk overlap | Full gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| From-scratch rescue | 20/20 | 0/20 | 0/20 | 20/20 | 0/20 | fail |
| Minerva 3B QLoRA | 20/20 | 2/20 | 13/20 | 7/20 | 0/20 | fail |
| Minerva 7B V6 staged LoRA | 20/20 | 8/20 | 20/20 | 5/20 | 0/20 | fail |
| Required | at least 18 | at least 12 | at least 10 | at most 2 | exactly 0 | all checks |

`Form` means exact opening preservation and decoder-controlled fourteen-line
output. It does not establish metre or rhyme.

## V7 Final Comparison

Stage 3 and DPO were compared under identical openings, seeds, prompting,
decoding, and stopping. The final protocol was frozen before opening the V7
test. All 1,244 test openings and two seeds yielded 2,488 outputs/system.

| V7 test metric | Stage 3 | DPO | Paired DPO change (95% interval) |
| --- | ---: | ---: | ---: |
| Opening preserved | 100.00% | 100.00% | `0.00` |
| Fourteen lines | 99.96% | 99.92% | `-0.04` points (`-0.20`, `+0.08`) |
| Meta-text free | 86.33% | 87.78% | `+1.45` points (`-0.28`, `+3.18`) |
| Terminal punctuation | 17.60% | 20.46% | `+2.85` points (`+0.72`, `+4.86`) |
| Automatic surface screen | 15.07% | 17.60% | `+2.53` points (`+0.52`, `+4.50`) |
| High-risk overlap | 0/2,488 | 0/2,488 | `0` |

On matched validation, DPO also improved the surface screen by 5.00 points
(95% interval `+0.63` to `+9.38`) and genuine completion in the frozen blind
sample from 12/40 to 20/40. The one-time test therefore replicates a modest
targeted gain.

In the final 200-output AI-analyst blind review, DPO improved mean historical
register from `2.88` to `3.09` (paired `+0.21`; 95% interval `+0.04` to
`+0.38`). Differences in grammar, poetic quality, sonnet/form, volta, and
visible completion all had intervals crossing zero. DPO supplied 3/100
moderate-clean outputs versus 0/100, but both systems supplied 0/100
strict-good outputs. The final evidence therefore does not show a broad
literary-quality breakthrough.

## What Improved

1. External Italian pretraining sharply improved coherence over the 70M
   from-scratch branch.
2. The V6 7B staged adapter was the strongest legacy system but still failed
   grammar and collapse thresholds.
3. Full-weight V7 adaptation learned the broader historical/poetry curriculum
   while keeping modern and instruction losses inside preservation gates.
4. A validation-selected no-label/prose prompt reduced meta-text.
5. AI-judged DPO delivered small replicated improvements in terminal behavior
   without detected high-risk copying or material preservation loss.

## What Still Failed

No evidence supports a claim of consistently acceptable sonnets. Outputs often
contain plausible historical diction while losing syntax, rhyme organization,
argument, or genuine closure. Decoder-controlled line count remains much easier
than sonnet structure.

The DPO labels also failed human calibration: AI-majority preferences agreed
with the user's review on 12/20 pairs. The final system is therefore accurately
described as AI-judge distillation, not human alignment. Its gains are bounded
to the measured behavior.

## Compute And Data Efficiency

- The from-scratch branch required multi-day training and still collapsed.
- V6 LoRA updated only 6.82M parameters and delivered the strongest legacy
  result.
- V7 full-weight training used 2,960 updates across three stages and retained
  six intermediate/boundary states for model-change analysis.
- DPO added only 61 optimizer updates and 148.6 seconds of H100 training, then
  required matched validation, preservation evaluation, and the one-time final
  grid to establish its modest effect.

These costs are not directly comparable because pretrained systems inherit
large external corpora.

## Final Decision

`minerva_7b_v7_stage_3_ai_judged_dpo` is the final experimental system. It was
selected before V7 test access because it improved the targeted completion and
surface behavior while preserving prior domains. Selection does not mean it
passed a complete sonnet-quality gate.

No retuning, rescue branch, or test rerun is permitted. The project ends with a
mixed result: a rigorous, reproducible improvement in a narrow behavior and a
negative result for reliable literary quality.

## Evidence

- From scratch: `reports/sonnet_task_format_paisa_historical_rescue_v1_v5_best_evaluation.md`
- Minerva 3B: `reports/minerva_3b_base_vs_qlora_evaluation.md`
- Minerva 7B V6: `reports/minerva_7b_v6_final_evaluation.md`
- V7 model-change study: `reports/minerva_7b_v7_post_training_study.md`
- V7 preference branch: `reports/minerva_7b_v7_ai_judged_dpo.md`
