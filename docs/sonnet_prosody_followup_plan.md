# Checker-First Sonnet Reasoning Follow-Up

## Status and scope

Leonardo approved the prosody checker as the first technical direction on
2026-09-08. On 2026-09-13 he approved the Phase A1 checkpoint plan and the
resolved decisions below. The A1 checker module, command-line entry point, and
tests are implemented. The later reasoning-training work is not committed. No
experiment execution, time budget, model change, checkpoint release, or remote
publication is approved by this document.

This plan keeps the detailed technical backlog in the implementation
repository. The private Career Center keeps the career purpose, owner
decisions, milestones, evidence state, and public-readiness state.

For this follow-up, approved repository work is committed locally. It must not
be pushed to the remote unless Leonardo gives an explicit instruction to push.
Any new model, checkpoint, or adapter release requires a separate licence and
publication review.

## Why the checker comes first

The existing automatic screen in
`src/sonnet_analysis/minerva_v7_quality.py` checks surface form. It checks for
14 lines, meta-text, terminal punctuation, lines longer than 120 characters,
and a repetition ratio below 0.35. It computes the 4+4+3+3 stanza pattern but
does not include that pattern in its pass criterion. It does not measure metre
or rhyme.

The published Stage-3 and DPO systems therefore have no measured metre or
rhyme result. A prosody checker can supply:

- evaluation metrics;
- a best-of-N selector;
- a preference or reinforcement-learning reward;
- diagnostics for a bounded revision loop;
- a corpus-derived rhyme lookup tool.

For a per-sample pass rate `q`, `N` samples give
`P(at least one pass) = 1 - (1-q)^N`. If `q = 5%`, `N = 64` gives about 96%.
This is a hypothesis about form selection. It does not show literary quality.
The actual value of `q` must be measured.

## Relevant repository evidence

These facts were inspected on 2026-09-08. Verify each source before an
experiment uses it.

- The V7 corpus has 19,899 training sonnets and 3,551,021 training tokens,
  with 1,247 validation and 1,244 test documents. Its 90/5/5 split is
  author-disjoint and work-disjoint.
- The corrected V8 corpus renders each sonnet as 4+4+3+3. No model was trained
  on V8 at the time of this plan.
- Staged supervised fine-tuning used plain-text language-model continuation.
  It did not use an instruction or chat training format.
- The Minerva stages used a hand-written, full-weight BF16 PyTorch trainer
  with PagedAdamW8bit on one H100. The three-stage cost was a qualification-
  based projection of about $10.65.
- DPO used a hand-written loss and rank-8 LoRA through PEFT. It did not use
  TRL. The recorded runtime was 148.6 s and the estimated cost was $0.093.
- The repository has no reinforcement-learning infrastructure.
- Generation uses a chat-templated Italian instruction and force-appends the
  opening line. A line-count stop produces 13 continuation lines. This is not
  constrained decoding.
- The sealed recipe used temperature 0.85, `top_p = 0.95`,
  `no_repeat_ngram = 4`, at most 512 new tokens, and seeds 6200 and 6201.
- The sealed evaluation has 4,976 outputs: 1,244 openings, two seeds, and two
  systems.
- DPO candidate generation has 4,096 candidates: 512 openings, two recipes,
  and four seeds. Three imported AI-judge vote sets produced 534 majority
  preference pairs. The judge identities were not confirmed in the inspected
  files and must be confirmed before reuse.
- Human and AI calibration failed at 12/20. The frozen blind literary review
  found 0/100 strict-good outputs for Stage 3 and 0/100 for DPO. Only the
  historical-register comparison had an interval that excluded zero.
- `src/sonnet_analysis/minerva_v7_prompt_intervention.py` provides a precedent
  for a bounded deterministic retry arm. It permits at most three attempts and
  uses the existing surface screen.

## Phase A: prosody checker

Phase A is the approved first direction. The A1 code checkpoint was approved on
2026-09-13 and is implemented. A2 and later checkpoints still require the
decisions listed near the end of this document.

### A1. Checker module

Implemented in `src/sonnet_evaluation/sonnet_prosody.py`, with the
command-line entry point `scripts/check_sonnet_prosody.py` and tests in
`tests/test_sonnet_prosody.py`. It covers:

- structure: 14 lines and a 4+4+3+3 stanza pattern;
- metre: poetic syllable count, sinalefe, diphthongs, elision, final stress at
  position 10, and `verso piano`, `verso tronco`, and `verso sdrucciolo`;
- rhyme: phonetic normalization of line-final words, rhyme classes, scheme
  extraction, comparison with target schemes, perfect rhyme, and a softer
  similarity result for historical conventions.

### A2. Checker validation

Validate the checker before it becomes an experiment metric or reward:

- create a public-domain ground-truth set with known-metre lines from Dante
  and Petrarch;
- freeze an accuracy gate and the manual-review sample size before evaluation;
- measure the line-level and poem-level error types;
- run full-corpus statistics;
- manually review a frozen sample of flagged lines;
- keep uncertain linguistic cases visible instead of forcing a pass or fail.

### A3. Retroactive evaluation

Score the existing 4,976 sealed outputs. Compare Stage 3 and DPO for metre,
rhyme, rhyme scheme, and stanza structure. Preserve the sealed outputs and
generation recipe. Report uncertainty and paired comparisons where the design
supports them.

### A4. Best-of-N pilot

Generate `N` Stage-3 candidates for each opening in a frozen sample. Select by
prosody score without new training. Measure:

- the observed per-sample pass rate `q`;
- success as `N` increases;
- metre, rhyme, and structure failure rates;
- differences between quatrains and tercets;
- the compute and time cost per accepted output;
- literary-quality risks introduced by selection for form alone.

Freeze the opening sample, values of `N`, generation recipe, selection order,
and stop rule before execution.

### A5. Rhyme lexicon

Build a rhyme lexicon from line-final words in the training sonnets. Preserve
source counts and training-split boundaries. The lexicon can later support a
`get_rhyme()` tool, constrained sampling, or plan generation.

## Phase A decision gate

Use the checker validation and best-of-N results to select the next problem:

- If checker accuracy is insufficient, improve or limit the checker before
  using it as a selector, label source, or reward.
- If `q` is high at a moderate `N`, shift the follow-up from form correction
  toward coherence and literary quality.
- If `q` is low and failure is mainly formal, test form-targeted training.
- If selection improves form but harms literary review, do not report the
  selector as an overall quality improvement.

## Phase B: contingent reasoning experiments

Phase B is not approved. Its order depends on Phase A evidence.

1. Test a bounded inference-time check-and-revise loop on current models.
   Prefer re-sampling with constraints when direct revision repeats the same
   defect.
2. Test verifier-labelled DPO round 2. Use the checker to select chosen and
   rejected outputs. Consider a separate V8 Stage-3 retraining decision first.
3. Test plan-then-poem SFT with compact, machine-checkable fields. Pre-commit
   rhyme words and measure whether the poem follows the plan. Compare a
   scheme-and-rhyme plan with a scheme-rhyme-theme plan.
4. Consider LoRA-GRPO with form rewards through TRL only if the earlier
   experiments show useful headroom.
5. Consider a larger base model as a separate experiment and licence decision.

Training tool use into the model is not the first route. Start with an
inference pipeline of generate, check, and bounded revise. A historical-word
attestation checker needs a diachronic lexicon and belongs in a later phase.

Plans derived from existing poems describe completed poems. They do not prove
that a model learns prospective planning. Any plan-then-poem experiment must
measure plan and poem consistency and report this distribution difference.

## Model options and compute assumptions

The model landscape below was checked on 2026-09-08. Recheck model versions,
licences, Italian tokenizer behavior, and hardware requirements before a model
decision.

- Minerva has no model larger than 7B.
- Velvet has 2B and 14B variants. It is multilingual and Apache-2.0. Italian
  tokenizer fertility was not checked.
- Qwen3 has 14B and 32B dense variants. It is multilingual and Apache-2.0.
- Other candidates include Mistral Small 24B, Gemma 3 27B, Llama 3.3 70B,
  EuroLLM-22B, LLaMAntino-3-ANITA 8B, and Modello Italia 9B. Recheck each
  licence before use.

For one 80 GB H100, a full-weight BF16 run with 8-bit Adam has about 6 bytes
of static memory per parameter. A 7B model uses about 42 GB. A 14B model uses
about 84 GB before activations. A 32B model uses about 190 GB. Full-weight 14B
therefore needs heavy offloading or multiple GPUs, and full-weight 32B is not a
one-H100 route.

BF16 LoRA or QLoRA is practical for 14B. QLoRA is the practical one-H100 route
for 24B to 32B. A 70B model is QLoRA-only on this hardware and will make
generation loops slow. A broad register shift can require rank 64 to 128 on
all linear layers and can require trained embeddings or an LM head. Treat rank
and target modules as measured choices. A 4-bit base can weaken the learned
register, so compare it with BF16 LoRA if a 14B arm reaches a final stage.

## Resolved decisions (2026-09-13)

Leonardo approved these decisions with the A1 checkpoint:

- Module, command, and test paths: `src/sonnet_evaluation/sonnet_prosody.py`,
  `scripts/check_sonnet_prosody.py`, and `tests/test_sonnet_prosody.py`.
- Ground truth: 30 sonnets from the corrected V8 corpus, 15 by Dante and 15 by
  Petrarch, using the recorded attribution. Freeze the set before scoring.
- Accuracy gate: at least 95% line-level metre accuracy and at least 90% rhyme
  agreement. Report ambiguous cases separately.
- Manual review: 100 flagged lines in one owner review session.
- Ambiguity: return `uncertain`, never force a pass or fail, and use the
  documented conventions for sinalefe, dialefe, elision, and stress.
- Poem score order: structure, metre, and rhyme scheme are hard gates; rhyme
  quality is a soft score after them.

## Open decisions before A2 and A4

- The ground-truth file paths and their licence records.
- The frozen manual-review sample and its selection rule.
- The A4 opening sample and the values of `N`.
- The Phase A time budget and the A4 compute budget.

Later decisions include the base model, a possible teacher model for thematic
plan fields, LoRA configuration, GPU budget, and release scope. Record the
actual provider, exact model, reasoning effort, date or range, role, completed
contribution, and evidence for all AI-assisted project work.
