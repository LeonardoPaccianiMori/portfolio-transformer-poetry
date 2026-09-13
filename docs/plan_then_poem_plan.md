# Plan-Then-Poem Pilot Plan

## Status and scope

Leonardo approved this direction on 2026-09-13 after the verifier-labelled
DPO produced a partial signal: metre sharpened, rhyme scheme stayed at zero,
and no valid sonnet appeared in any system. The working diagnosis is that
rhyme consistency across lines needs an explicit plan rather than more
training or a larger corpus.

This plan covers two steps:

1. Build the rhyme lexicon from the corrected V8 train corpus (Phase A5,
   already approved in the checker-first plan).
2. An inference-only plan-adherence test: pre-commit the line endings, ask the
   model to follow them, and measure adherence and form.

A plan-then-poem supervised fine-tuning arm is not approved by this document
and requires a later proposal with its own budget.

## Question

Can the model follow pre-committed line endings, and does following them
raise rhyme scheme compliance and full-form validity?

## Step 1: rhyme lexicon (Phase A5)

- Source: the V8 corrected sonnet train split, the same frozen corpus as the
  retrain.
- For every line-final word, record the checker exact rhyme key and soft
  (Sicilian) key, their frequencies, and the most frequent words per key.
- Output: `data/metadata/rhyme_lexicon_v1.json` with the source manifest hash,
  the sonnet and line counts, and the per-key word lists.
- This step is CPU only and takes minutes.

## Step 2: inference-only plan test

- Openings: the 120 recorded validation prompts, unchanged.
- Plans: one target scheme per opening, restricted to schemes observed in the
  training corpus. Endings are sampled deterministically from the lexicon
  with a recorded seed, with constraints: no word repeats across classes,
  the single most frequent word of a class is not used, and each class has at
  least five candidate words. Every line ending is fixed before generation.
- Conditions (generated in the same run, same model, prompt builder, recipe,
  and seeds):
  - model: the verifier-DPO adapter on Stage-3, the best metre model so far,
    identical across conditions;
  - planned: the numbered list of planned final words is appended to the
    frozen instruction prompt;
  - format control: the same appended list shape with non-rhyming placeholder
    endings, to separate schema confusion from planning failure;
  - baseline: the existing verifier-DPO no-plan outputs at the paired seeds,
    paired by opening and seed.
- Generation: 120 openings, seeds 5200 and 5201 (the seeds of the existing
  verifier-DPO baseline), the frozen recipe, and the recorded
  13-continuation-line stop rule. The planned list does not change the stop
  rule. The sealed test set is not accessed.
- Adherence metrics, defined on the line-final word of each generated line:
  - `word_match`: the final word is identical to the planned word;
  - `key_match`: the final word has the same checker exact rhyme key as the
    planned word;
  - `scheme_adherence`: the quatrain and tercet rhyme groups match the
    planned scheme.
  Uncertain or missing final words are reported separately and never counted
  as matches.
- Form metrics: accepted hendecasyllable lines, failed and uncertain lines,
  quatrain and tercet scheme compliance, rhyme score, with paired comparisons
  against the baseline by opening and seed.

## Pre-registered reading for Step 2

- The gates use `key_match` adherence, not only identical words.
- If `key_match` adherence is at least 40% and scheme compliance exceeds zero,
  proceed to a plan-then-poem SFT proposal.
- If `key_match` adherence is below 20%, the model cannot follow plans under
  this prompt; the next experiment must first train plan following.
- Between 20% and 40%, or when adherence is high but scheme compliance stays
  zero, run a mechanical-format audit and propose a small plan-following SFT
  arm before any larger training.
- The format control must be reported next to the planned condition; a
  planned-versus-control difference is the evidence for plan following.

## Budget

- Step 1: CPU only.
- Step 2: about 20 to 30 minutes of H100, roughly $1, plus scoring time.

## Limits

- Form only. The checker's definite coverage limit (87.1% on the ground
  truth) applies.
- Plan adherence can be met with archaic, rare, or semantically odd endings;
  the report must separate adherence from coherence and must not claim a
  quality gain.
- The plan is machine-built from corpus frequencies; it does not model theme
  or meaning.
- No release, no checkpoint publication, no public wording before a separate
  review.

## Review outcome (2026-09-13)

The independent review found no problem with the inference-first ordering and
required three changes, now included: a format-control condition so a novel
prompt schema is not confused with a failure to plan; precise adherence
metrics at the line-final-word and key level; and paired comparisons by
opening and seed. The review also recommended separating word matching from
scheme compliance, which the revised gates do.
