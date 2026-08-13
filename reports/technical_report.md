# Building And Evaluating Classical-Italian Sonnet Language Models

## Abstract

This project implements a GPT-style causal transformer from first principles in
PyTorch, builds provenance-aware historical-Italian and sonnet corpora, and
compares from-scratch training with parameter-efficient and full-weight
adaptation of open Minerva language models. The work began under a 6 GiB
laptop-GPU target and later used rented 48 GiB and 80 GB GPUs for the bounded
Minerva 7B experiments.

The final from-scratch system reliably followed the opening-line and line-count
interface but failed all qualitative language criteria. Minerva adaptation
substantially improved topic continuity, grammar, and collapse rates, although
no system passed the complete acceptable-quality gate. The final V7 experiment
applied three stages of full-weight BF16 curriculum adaptation, retained six
intermediate/boundary states, and studied weight, embedding, representation,
loss, and behavioral change. A bounded AI-judge-distillation DPO adapter then
produced small replicated gains in automatic surface behavior and historical
register without establishing reliable sonnet quality. Neither final system
produced a strict-good output in the frozen 100-prompt literary comparison.

## 1. Research Question

The initial question was whether a tiny, inspectable transformer could learn
enough Italian language and classical sonnet structure to produce acceptable
opening-line-conditioned poems. A second comparison question asked how much a
pretrained Italian model changes the result when corpus and evaluation controls
remain as consistent as practical.

The project prioritizes honest evidence over selected impressive samples. A
model must pass form, grammar, topic continuity, collapse, and memorization
checks together. Decoder-enforced line count is never treated as proof of metre
or rhyme.

## 2. Data

### Sonnet Corpus

The final V6 sonnet corpus contains 1,868 poems: 1,481 train, 190 validation,
and 197 final test. V6 removed one editorial apparatus page and six duplicate or
mislabeled V5 records while preserving all frozen validation and test prompts.
It has no exact-text or cross-split duplicate groups under the recorded checks.

Sources cover multiple authors and centuries, with form verified from source
metadata and structural checks. Original spelling and punctuation are
preserved. Source, author, URL, revision, license, and cleaning metadata are
retained in the manifest and attribution index.

### Broader Italian Corpora

Historical pretraining uses 36 prose sources from public-domain and explicitly
licensed archives. The final data path also includes PAISÀ modern-Italian web
text for scale and replay. Raw, interim, processed, encoded, and local-only
artifacts are separated to preserve provenance and prevent accidental
publication of restricted material.

Licensing is not limited to public domain. Compatible Creative Commons and
other non-commercial permissions are accepted when attribution, share-alike,
notice, and distribution restrictions are recorded. PAISÀ-derived checkpoints
are not published under the project's repository policy.

## 3. From-Scratch Transformer

The repository's transformer implementation does not hide core logic behind a
high-level model library. It includes:

- Unicode character and BPE tokenizers;
- token and positional representations;
- causal scaled dot-product attention and multi-head attention;
- residual blocks and LayerNorm or RMSNorm;
- learned positions or RoPE;
- conventional feed-forward or SwiGLU layers;
- optional weight tying;
- explicit next-token loss, optimization, validation, checkpointing, resume,
  and autoregressive generation.

Tests protect tensor shapes, masks, shifted labels, normalization behavior,
positional encoding, checkpoint reconstruction, data splitting, and generation
controls. Long-running jobs record flushed progress, elapsed time, ETA,
validation, learning rate, and checkpoints.

## 4. From-Scratch Experiments

The project advanced from character models to BPE and broader-corpus
pretraining. Controlled experiments isolated RMSNorm conversion versus native
pretraining, RoPE, SwiGLU, learning-rate scheduling, gradient clipping, corpus
scaling, model scaling, and checkpoint-neighborhood selection.

The strongest from-scratch branch used a 70,055,900-parameter model with
PAISÀ-first modern-Italian pretraining, historical-Italian adaptation, V5
sonnet specialization, and task-format post-training. The expensive curriculum
was intentionally allowed to run for multiple days because output quality was
the priority.

The result passed mechanical form and memorization controls but produced 0/20
generally grammatical outputs, 0/20 topic-continuous outputs, and 20/20 severe
collapses. Under the frozen exit policy, this closed corpus-only from-scratch
development. The result is evidence about this scale, data, and compute budget,
not a claim that from-scratch language modeling is impossible.

## 5. Minerva 3B QLoRA

The first pretrained comparison used
`sapienzanlp/Minerva-3B-base-v1.0`. The 4-bit NF4 base remained frozen while
26,214,400 rank-16 adapter parameters were trained on V5 continuation examples.
Prompt tokens were masked from the loss. Epoch 3/update 558 was selected by
validation and training stopped after epoch 6 through patience.

QLoRA changed task behavior decisively: controlled form improved to 20/20,
topic continuity to 13/20, and severe collapse fell to 7/20. Grammar remained
only 2/20. A blinded validation audit across adapter scales showed that reducing
adapter strength did not recover grammar; the recipe had learned historical
surface cues and form while damaging syntax.

## 6. Minerva 7B Staged LoRA

The repair branch used the pinned Minerva 7B Instruct model. A remote RTX 8000
allowed the 7.4B base to remain unquantized FP16 during training. Only 6,815,744
rank-8 attention-adapter parameters were updated.

Stage A mixed seven historical 512-token windows with one PAISÀ replay window
per update. Selection required historical improvement without excessive modern
Italian or instruction loss. Update 4,000 passed and reduced historical
validation loss from `3.262937` to `3.187692`.

Stage B continued the same adapter with a fresh AdamW optimizer and complete
chat-formatted V6 sonnet targets. Epochs 3, 4, and 5 were compared using frozen
validation generation. Epoch 4 was selected for lower repetition and stronger
qualitative stability while remaining close to minimum validation loss.

Only after selection was hash-frozen was the final test opened. The selected
adapter reached final-test loss `3.212791`, 20/20 form, 8/20 grammar, 20/20
topic, 5/20 collapse, and 0/20 high-risk overlap. It failed the complete gate
but materially outperformed every prior system.

## 7. Minerva 7B V7 Full-Weight Curriculum

V7 starts from the pinned `sapienzanlp/Minerva-7B-instruct-v1.0` parent and
updates the complete model in BF16 on one H100 80 GB. The three-stage curriculum
contains 2,065 historical/general updates, 760 historical non-sonnet-poetry
updates, and 135 V7-sonnet updates. Each stage uses warmup followed by cosine
decay, fixed 2,048-token context, 32,768 target tokens per optimizer update,
validation/preservation gates, atomic resume state, and validation-selected
checkpoint retention.

All stages passed their declared gates. Stage 1 selected update 2,065 at
historical-general loss `2.8466`; Stage 2 selected update 760 at poetry loss
`2.8475`; Stage 3 selected update 120 of 135 at sonnet loss `3.1103`. Update 120
was effectively tied with update 135 on sonnet loss while preserving instruction
behavior better. The run retained midpoint and selected BF16 states for every
stage, rather than retaining only the final model.

The global parent-to-Stage-3 relative parameter displacement was `0.03302`.
Most movement occurred in Stage 1, with progressively smaller changes in Stages
2 and 3. Across 48 fixed probes, parent-to-final mean hidden-state drift was
`0.2394`, minimum linear CKA was `0.9219`, and standard-sonnet probes moved most.
The late half of Stage 3 had only `0.00371` mean drift and minimum CKA
`0.999997`. These are descriptive localization results: they show where and how
much the model changed, not that any component caused a particular behavior.

## 8. Evaluation Design

Generation protocols fix prompts, seeds, temperature, top-k, token ceiling,
line target, and special-token suppression. Reports include complete grids,
not selected favorable examples. V7 evaluation uses author/work-isolated
validation material for development and an untouched 1,244-opening test split
opened only after the final system, comparator, decoder, stopping rule, metrics,
blind sample, and selection record were hash-frozen.

Automatic checks cover exact opening preservation, non-empty line count,
boundary-marker leakage, repetition, meta-text, terminal punctuation, and
character-level overlap with training poems. Frozen qualitative reviews score
grammar, historical register, poetic quality, sonnet/form coherence, and
volta/argument, plus visible truncation, meta-text, and collapse. Reviews in the
V7 research programme were performed by an AI qualitative analyst and are
identified as such; they are not presented as independent human judgments.
Dimensions remain separate so decoder-controlled line count cannot conceal
language or structural failure.

Checkpoint choice uses validation loss plus neighboring-checkpoint generation
where appropriate. Final-test bodies and prompts remain unavailable until the
selection record is frozen and hashed.

## 9. Preference Learning: Two Different Outcomes

### Cancelled From-Scratch Reward Branch

The project considered DPO and GRPO for the selected from-scratch parent, using
untouched Minerva 3B as an AI feedback source. Before policy optimization, a
fixed validation-only gate tested negative continuation NLL against authentic,
generated, word-order-corrupted, and previously human-assessed controls.

The signal detected grammar and obvious corruption, but it failed as a general
reward. Every from-scratch output outranked its authentic sonnet, non-collapse
AUROC was `0.1241`, and ordinal agreement with human quality was `0.4003`.
Collapsed outputs received substantially higher model likelihood than
non-collapsed outputs. Optimizing this reward could make repetition worse.

The gate therefore failed and both DPO and GRPO were cancelled for that
from-scratch lineage. This negative result demonstrates why reward validation
is a prerequisite.

### Bounded V7 AI-Judged DPO

A later and distinct V7 branch used direct pairwise AI judgments rather than
the failed likelihood reward. Stage 3 generated 4,096 candidates from 512
training-only openings. Deterministic screening and three blind AI votes per
pair yielded 534 chosen/rejected comparisons; 437 majorities were unanimous.
Human--AI calibration failed: AI-majority labels agreed with the user's separate
20-pair review on only 12 pairs (60%). The branch is therefore AI-judge
distillation, not human-calibrated DPO, RLHF, or evidence of human alignment.

One rank-8 LoRA-DPO epoch trained 20.0M adapter parameters for 61 optimizer
updates. Held-out preference accuracy was `65.38%`. Preservation loss changes
were at most `+0.01334`, on instruction validation. The adapter was selected
before test access because matched validation showed a positive
surface-screen interval and better blind completion, with no high-risk copying.

The one-time V7 test generated 4,976 outputs: all 1,244 test openings, two seeds,
and matched Stage-3/DPO systems. DPO increased the automatic surface screen from
`15.07%` to `17.60%` (paired `+2.53` points, 95% prompt-bootstrap interval
`+0.52` to `+4.50`) and terminal punctuation from `17.60%` to `20.46%`
(`+2.85` points, `+0.72` to `+4.86`). Meta-text-free output rose from `86.33%`
to `87.78%`, but its interval crossed zero. Neither system produced a high-risk
memorization match.

The frozen literary review used 100 matched prompts and one seed per prompt.
DPO improved mean historical register from `2.88` to `3.09` (paired `+0.21`,
95% interval `+0.04` to `+0.38`). Intervals for grammar, poetic quality,
sonnet/form, volta, and visible completion crossed zero. DPO supplied 3/100
moderate-clean outputs versus 0/100, but both systems supplied 0/100 strict-good
outputs. This is a narrow improvement, not a solved sonnet generator.

## 10. Engineering Outcomes

The repository includes reproducible source acquisition, corpus audits,
tokenizer reports, hardware benchmarks, atomic resume checkpoints, candidate
selection records, exact artifact hashes, fixed evaluations, and a local web
demo. It also contains V7 state registries, memory-bounded checkpoint-delta
analysis, embedding and representation probes, high-volume generation,
prompt-intervention analysis, direct PyTorch/PEFT DPO, preservation evaluation,
and one-time final-test tooling. The complete local suite passes more than
1,150 CPU tests.

Training and evaluation scripts provide flushed progress by default. GPU jobs
record device, token budget, update count, learning rate, validation losses,
preservation gates, and selection state. Long remote runs were copied locally
and hash-verified before the temporary GPU was stopped.

## 11. Limitations

- No tested model passed the complete acceptable-quality gate.
- Sonnet metre and rhyme were not certified; blind form scores remained low.
- The strongest system inherits unknown external-pretraining overlap.
- Historical orthography complicates grammatical review.
- V7 qualitative judgments were made by an AI analyst, and the DPO judges are
  correlated systems whose majority failed the 20-pair human-calibration gate.
- The memorization heuristic cannot detect every form of recall.
- The systems differ in scale, tokenizer, pretraining, and hardware, so the
  final comparison is behavioral rather than causally controlled.
- The selected adapter is not a public weight release because its lineage
  includes non-commercial PAISÀ replay and other licensed source obligations.
- The laptop demo's transient 4-bit NF4 load is a deployment approximation;
  all authoritative V7 evidence used the full BF16 checkpoint on the H100.

## 12. Conclusion

At laptop scale, the from-scratch branch succeeded as an educational and
engineering implementation but not as an acceptable sonnet generator.
Pretrained Italian models delivered large improvements. Full-weight V7
adaptation provides a documented model-change case study: most change occurred
during historical adaptation, poetry training added smaller domain gains, and
sonnet specialization was narrow. AI-judged DPO then produced a statistically
supported surface/completion gain and a small historical-register shift, but
not a reliable improvement across grammar, poetic quality, form, or volta.

The project therefore ends with a rigorous negative-to-mixed result rather than
an inflated success claim. Its strongest portfolio evidence is the complete
experimental method: implementation from first principles, licensed data
engineering, controlled comparisons, failure analysis, reward validation,
reproducible artifacts, and a working local demo.

## Primary Evidence

- `DATA_SOURCES_AND_ATTRIBUTION.md`
- `reports/final_model_comparison.md`
- `reports/sonnet_task_format_paisa_historical_rescue_v1_v5_best_evaluation.md`
- `reports/minerva_3b_base_vs_qlora_evaluation.md`
- `reports/minerva_7b_v6_final_evaluation.md`
- `reports/minerva_3b_judge_gate.md`
- `reports/minerva_7b_v7_post_training_study.md`
- `reports/minerva_7b_v7_ai_judged_dpo.md`
- `MODEL_CARD.md`
