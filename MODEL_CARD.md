# Model Card: Minerva 7B V7 Stage 3 + AI-Judged DPO

## Model Summary

This is the repository's final experimental sonnet-continuation system. It
starts from pinned `sapienzanlp/Minerva-7B-instruct-v1.0`, applies three stages
of full-weight BF16 adaptation, and attaches one rank-8 LoRA-DPO adapter trained
from AI-judged preferences.

The system increased automatic terminal-punctuation and surface-screen rates
over the Stage-3 comparator. The meta-text point estimate and blind visible-
completion difference remained uncertain. It is not a consistently good
sonnet generator and is not human-aligned. The three AI judges agreed with the
user's separate 20-pair review on only 12/20 pairs (60%).

## Identification

- Project model: `minerva_7b_v7_stage_3_ai_judged_dpo`
- Parent: `sapienzanlp/Minerva-7B-instruct-v1.0`
- Parent revision: `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`
- Stage-3 selected update: 120 of 135
- Stage-3 state identity:
  `478d5979e25a78375d7af0434db6a5432678762fac2d142af2d4798dda53a474`
- DPO adapter: rank 8, alpha 16, attention and MLP projections
- DPO adapter SHA-256:
  `72aa174b2ef87e021a367b0f7e786fce8c3437bb5ca1f8c7f9c5b13588620822`
- Context length: 2,048 for full-weight training; 1,024 for DPO preference
  scoring
- Primary language: Italian, including historical orthography

The authoritative Stage-3 checkpoint is retained as full BF16 weights. All V7
validation, preservation, and final-test evidence used unquantized BF16. The
optional laptop demo loads a transient 4-bit NF4 deployment approximation only
because the 14.8 GB BF16 checkpoint cannot fit in 6 GB VRAM; demo outputs may
differ slightly and are not authoritative research evidence.

## Intended Task

Input is one exact opening line. Output is intended to be a complete
fourteen-line classical-Italian sonnet. The decoder preserves the input line
and stops after thirteen continuation lines.

Fourteen-line stopping does not prove learned rhyme, hendecasyllabic metre,
octave/sestet structure, volta, grammar, or literary quality.

## Training Lineage

### Stage 1: Historical And Literary Italian

- 2,065 full-weight BF16 updates
- peak learning rate `1e-5`, warmup then cosine decay
- selected historical-general loss `2.8466`
- historical, poetry, sonnet, modern, and instruction gates passed

### Stage 2: Historical Non-Sonnet Poetry

- 760 full-weight BF16 updates
- peak learning rate `5e-6`, warmup then cosine decay
- selected poetry loss `2.8475`
- all adaptation and preservation gates passed

### Stage 3: V7 Sonnets

- 135 full-weight BF16 updates; update 120 selected
- peak learning rate `2e-6`, warmup then cosine decay
- selected V7 sonnet validation loss `3.1103`
- update 120 preserved instruction behavior better than terminal update 135

Across all three completed V7 stages, deterministic PAISÀ
modern-preservation replay supplied exactly 5% of target-token exposure. This
describes training-token exposure, not 5% of unique documents, corpus size, or
examples. The replay is distinct from the earlier prospective PAISÀ rescue
curriculum.

### AI-Judged DPO

- 4,096 Stage-3 candidates from V7 training-only openings
- 534 preference pairs, each judged blindly by three AI judges
- 482 prompt-disjoint training and 52 validation pairs
- one epoch / 61 optimizer updates, beta `0.1`
- held-out preference accuracy `65.38%`
- one H100 runtime 148.6 seconds; peak VRAM 14.81 GiB

Human-reviewed calibration pairs were excluded from training. This branch
distills the frozen AI-majority rubric; it is not RLHF or human-calibrated DPO.

## Evaluation

### Matched Validation

Across 960 matched outputs, DPO increased the automatic surface-screen rate
from 13.96% to 18.96% (paired `+5.00` points; 95% interval `+0.63` to `+9.38`).
In a frozen 80-output blind review, genuine terminal completion was 20/40 for
DPO and 12/40 for Stage 3. Neither system produced a strict-good reviewed
output.

### One-Time V7 Test

All 1,244 sealed V7 test openings were generated with two seeds and both
systems, for 4,976 outputs. The system and protocol were frozen before access;
no post-test tuning or rerun is allowed.

| Metric | Stage 3 | DPO | Paired DPO change (95% interval) |
| --- | ---: | ---: | ---: |
| Opening preserved | 100.00% | 100.00% | `0.00` |
| Fourteen lines | 99.96% | 99.92% | `-0.04` points (`-0.20`, `+0.08`) |
| Meta-text free | 86.33% | 87.78% | `+1.45` points (`-0.28`, `+3.18`) |
| Terminal punctuation | 17.60% | 20.46% | `+2.85` points (`+0.72`, `+4.86`) |
| Automatic surface screen | 15.07% | 17.60% | `+2.53` points (`+0.52`, `+4.50`) |
| High-risk memorization | 0/2,488 | 0/2,488 | `0` |

The frozen 200-output final blind review used 100 matched prompts. DPO improved
mean historical register from `2.88` to `3.09` (paired `+0.21`; 95% interval
`+0.04` to `+0.38`). Grammar, poetic quality, sonnet/form, volta, and visible
completion differences remained uncertain. DPO produced 3/100 moderate-clean
outputs versus 0/100, but both systems produced 0/100 strict-good outputs. The
full review is reported in `reports/minerva_7b_v7_post_training_study.md`.

## Preservation

DPO-minus-Stage-3 loss changes were:

- historical-general `-0.00003`;
- historical poetry `+0.00004`;
- V7 sonnet validation `+0.00063`;
- modern Italian `+0.00014`;
- instruction validation `+0.01334`.

The adapter therefore preserved prior domains closely, with a small but
reported instruction regression.

## Intended Uses

- Reproducing the repository's controlled sonnet evaluation
- Studying full-weight curriculum adaptation and preference optimization
- Inspecting successful and failed historical-Italian generations
- Demonstrating deployment-aware 4-bit inference on a laptop GPU
- Educational comparison with the project's from-scratch and earlier Minerva
  systems

## Out-of-Scope Uses

- Publishing outputs as authentic historical poetry
- Claims of reliable rhyme, metre, sonnet structure, or literary authorship
- Human-alignment claims
- Literary scholarship, attribution, grading, translation, or textual criticism
- Production deployment without substantially stronger evaluation

## Limitations

- Literary quality remains unreliable even when surface checks pass.
- Rhyme and metre were not certified.
- AI judges are correlated and failed the human-calibration threshold.
- Historical-looking language can conceal broken syntax or incoherence.
- Automatic punctuation is weaker than genuine syntactic closure.
- Memorization checks detect long surface overlap, not every form of recall.
- The parent may have unknown external-pretraining overlap.
- Local 4-bit deployment can differ from authoritative BF16 behavior.

## Licensing And Distribution

The Minerva parent is recorded as Apache-2.0 and must retain its model ID,
revision, source link, Sapienza NLP attribution, and license notice.

Training lineage includes public-domain material, Italian Wikisource,
Liber Liber CC BY-NC-SA editions, and PAISÀ CC BY-NC-SA replay. The full-weight
checkpoints and DPO adapter are withheld under a conservative release policy
pending separate artifact-specific specialist review. That policy is not a
legal conclusion that the PAISÀ corpus license necessarily governs model
weights. This card grants no rights beyond the underlying model and dataset
terms.

## Local Demo

Install the Minerva extras from `requirements/minerva_qlora.txt`, retain the
local Stage-3 archive and DPO adapter, then run:

```bash
.venv/bin/python -u scripts/serve_sonnet_demo.py --device cuda:0
```

The loader validates the frozen system, adapter SHA-256, and Stage-3 state
identity before allocating the 4-bit NF4 deployment approximation. Use
`--legacy-v6` to load the prior V6 demo instead.

## Related Evidence

- `reports/minerva_7b_v7_post_training_study.md`
- `reports/minerva_7b_v7_ai_judged_dpo.md`
- `reports/minerva_7b_v7_full_weight_protocol_v1.md`
- `reports/minerva_7b_v7_analysis_foundation_v1.md`
- `reports/final_model_comparison.md`
