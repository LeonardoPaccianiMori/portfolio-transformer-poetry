# Minerva 7B V7 AI-Judged DPO

## Scope

This experiment applies one bounded LoRA-DPO update to the validation-selected
Stage-3 full-weight Minerva 7B model. It targets two observed generation
failures: meta-text around the poem and incomplete terminal syntax.

This is **AI-judged preference optimization**, not human-calibrated or
human-aligned DPO. Three AI judges supplied the controlling training labels.
Their majority agreed with the user's separate 20-pair calibration review on
only 12/20 pairs (60%). Those human-reviewed pairs remained validation-only and
were never used for DPO training.

## Preference Data

The Stage-3 model generated 4,096 training-only candidates from 512 V7 training
openings and eight seeds. Candidate screening retained exact opening lineage,
fourteen-line completion, repetition, meta-text, terminal punctuation, and
memorization evidence. The frozen builder produced:

- 173 ordinary literary comparisons;
- 361 terminal-completion contrasts;
- 534 total preference pairs;
- zero high-risk memorization candidates.

Each pair received three blind votes under the unchanged six-dimension rubric.
All 1,602 votes were present, and every pair had a decisive majority. Of 534
majorities, 437 (81.84%) were unanimous. Judge disagreement is retained rather
than discarded.

One judge used `candidate_a` and `candidate_b` instead of `A` and `B` for 294
score maps. The original files were preserved; only those equivalent keys were
normalized before schema validation and aggregation.

## Training

The implementation uses direct PyTorch and PEFT rather than TRL. A single
shared Stage-3 base computes policy scores with the rank-8 adapter enabled and
reference scores with it disabled. Only response-token log likelihood enters
the DPO objective.

| Setting | Value |
| --- | --- |
| Parent | Stage-3 validation-selected full-weight Minerva 7B |
| Training / validation pairs | 482 / 52, prompt-disjoint |
| Adapter | rank 8, alpha 16, attention and MLP projections |
| DPO beta | 0.1 |
| Epochs / optimizer updates | 1 / 61 |
| Microbatch / accumulation | 1 / 8 |
| Optimizer | AdamW, gradient clipping, warmup plus cosine decay |
| Hardware | one H100 80 GB |
| All-in runtime | 148.6 seconds |
| Peak VRAM | 14.81 GiB |
| Estimated run cost | $0.093 |

A disposable one-update qualification preceded training and retained no mutated
weights. The authoritative run saved best, final, five periodic, and resumable
optimizer-state artifacts.

Held-out preference loss was `0.6629`; held-out preference accuracy was
`65.38%`. This is evidence that the adapter learned some of its AI-judge target,
not proof of improved human literary quality.

## Validation Selection

Stage 3 and DPO generated 960 matched validation outputs: 120 held-out openings,
four seeds, and two systems. Prompt, decoding, stopping, input, and RNG seed were
identical; adapter enablement was the intended difference.

| Metric | Stage 3 | DPO | Paired DPO change |
| --- | ---: | ---: | ---: |
| Exact opening preservation | 100.00% | 100.00% | 0.00 points |
| Fourteen lines | 100.00% | 100.00% | 0.00 points |
| Meta-text free | 80.83% | 85.42% | +4.58 points |
| Automatic surface screen | 13.96% | 18.96% | +5.00 points |
| Terminal punctuation | 18.13% | 20.83% | +2.71 points |
| High memorization risk | 0/480 | 0/480 | 0 |

The prompt-cluster bootstrap 95% interval for the surface-screen change was
`+0.63` to `+9.38` percentage points. The intervals for terminal punctuation
and most continuous surface measures crossed zero.

An 80-output sample (40 matched prompts) was frozen and reviewed before system
identities were revealed. The AI analyst review found terminally complete poems
in 20/40 DPO outputs and 12/40 Stage-3 outputs. DPO was slightly higher in
historical register, poetic quality, and volta/argument, while grammar and
sonnet-form scores were approximately unchanged. Neither system produced a
strict-good output under the five-dimension clean-completion rubric. The
experiment therefore improved a bounded behavior without solving sonnet
quality.

## Preservation

Losses were recomputed on the frozen historical, poetry, sonnet, modern-Italian,
and instruction domains with the adapter disabled and enabled.

| DPO minus Stage 3 | Loss change |
| --- | ---: |
| Historical-general bridge | -0.00003 |
| Historical non-sonnet poetry | +0.00004 |
| V7 sonnet validation | +0.00063 |
| Modern Italian | +0.00014 |
| Instruction validation | +0.01334 |

The changes are small relative to the preserved baseline losses. The modest
instruction regression is reported rather than treated as zero.

## Selection And Limitations

DPO was selected over Stage 3 for the one-time V7 final comparison because it
combined held-out preference learning, a positive surface-screen interval,
better blinded completion, zero detected high-risk copying, and small
preservation changes. Selection does not override the following limitations:

- human--AI calibration failed at 60%;
- the judges are correlated AI systems, not independent human literary experts;
- no strict-good validation output was identified;
- fourteen-line control does not establish rhyme, metre, stanza structure, or
  genuine sonnet quality;
- automatic terminal punctuation is weaker than syntactic and argumentative
  closure;
- memorization screening cannot detect every kind of recall.

The one-time final-test protocol was hash-frozen before test access. Final-test
results may be analyzed and reported but may not trigger retuning, a rescue
branch, or a rerun.

## One-Time Final Test

The frozen final protocol evaluated every one of the 1,244 V7 test openings
with two seeds and both Stage 3 and DPO, for 4,976 matched outputs. Generation
used the authoritative unquantized BF16 Stage-3 checkpoint plus the frozen DPO
adapter. It took 2,970.6 seconds on one H100 and cost approximately `$1.967`.

| Metric | Stage 3 | DPO | Paired DPO change (95% interval) |
| --- | ---: | ---: | ---: |
| Exact opening preservation | 100.00% | 100.00% | `0.00` |
| Fourteen lines | 99.96% | 99.92% | `-0.04` points (`-0.20`, `+0.08`) |
| Meta-text free | 86.33% | 87.78% | `+1.45` points (`-0.28`, `+3.18`) |
| Terminal punctuation | 17.60% | 20.46% | `+2.85` points (`+0.72`, `+4.86`) |
| Automatic surface screen | 15.07% | 17.60% | `+2.53` points (`+0.52`, `+4.50`) |
| High memorization risk | 0/2,488 | 0/2,488 | `0` |

The sealed-test automatic result therefore replicates a modest DPO advantage
on the targeted surface/completion behavior. The meta-text interval crosses
zero, and punctuation is not equivalent to genuine syntactic closure. The
predeclared 200-output blind literary review remains the controlling quality
assessment.

The blind review used 100 matched prompts and one frozen seed per prompt. An AI
qualitative analyst scored all outputs before identities were opened. Mean
DPO-minus-Stage-3 changes were grammar `-0.01` (95% prompt-bootstrap interval
`-0.20` to `+0.18`), historical register `+0.21` (`+0.04` to `+0.38`), poetic
quality `+0.06` (`-0.12` to `+0.23`), sonnet/form `+0.09` (`-0.03` to `+0.22`),
and volta/argument `0.00` (`-0.21` to `+0.21`). Visible completion was 41/100
for DPO and 35/100 for Stage 3; that six-point difference was uncertain
(`-9` to `+21`). DPO supplied 3/100 moderate-clean outputs versus 0/100, but
neither system supplied a strict-good output.

Thus, only the historical-register interval excluded zero. The final evidence
supports a narrow surface improvement and a small register shift, not a broad
or reliable literary-quality improvement.

## Reproducibility Artifacts

- `configs/minerva_7b_v7_ai_judged_dpo.json`
- `src/sonnet_training/minerva_v7_ai_dpo.py`
- `scripts/train_minerva_v7_ai_judged_dpo.py`
- `src/sonnet_analysis/minerva_v7_dpo_preferences.py`
- `src/sonnet_analysis/minerva_v7_dpo_validation.py`
- `scripts/analyze_minerva_v7_dpo_validation.py`
- `scripts/evaluate_minerva_v7_dpo_preservation.py`
- `src/sonnet_analysis/minerva_v7_final_evaluation.py`
- `scripts/generate_minerva_v7_one_time_final.py`
- `scripts/analyze_minerva_v7_one_time_final.py`

Large candidates, votes, adapters, raw generations, mappings, and resume state
remain local research artifacts and are not committed as public repository
data.
