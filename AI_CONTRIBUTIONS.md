# AI Contributions

Leonardo Pacciani-Mori conceived and directed the project, set its learning and
research goals, made executive decisions, approved the research plan, reviewed
outputs, and sometimes ran GPU work.

Codex 5.5 and later Codex 5.6 Sol helped design the research plan and
substantially assisted implementation, tests, execution, and analysis. Their
contribution includes source and test drafting, experiment and evaluation
tooling, debugging, evidence synthesis, and documentation support.

Leonardo remained responsible for accepting or rejecting decisions and for the
public release. The project must not be described as independently designed or
independently implemented by Leonardo. Generated or AI-assisted material was
reviewed against tests, frozen protocols, source artifacts, and reported
limitations; that review does not turn AI assistance into independent human
authorship of every implementation detail.

## 2026-09-08 correction work

| Provider and model | Reasoning effort | Role | Completed contribution |
| --- | --- | --- | --- |
| OpenAI GPT-5.6 Sol | high | Corpus correction implementation | Implemented and tested the V8 source-backed authorship corrections, exact line recovery, derived stanza rendering, strict held-out isolation, broader overlap quarantine, and `ſ` to `s` normalization. |
| OpenAI GPT-5.6 Sol | high | Memorization correction implementation | Corrected the reference-ranking rule, added the regression test, and rechecked all 4,976 saved final outputs. |
| OpenAI GPT-5.6 Sol | high | Independent correction review | Reviewed the integrated V8 and memorization correction artifacts before final validation and local commits. |

## 2026-09-09 follow-up plan placement

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenAI GPT-6 | unknown | Technical plan drafting | Moved the detailed checker-first prosody and contingent reasoning plan into the implementation repository under Leonardo's approved repository-boundary correction. | `docs/sonnet_prosody_followup_plan.md` and this dated record. |

## 2026-09-13 Phase A1 prosody checker

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Checker module, command-line entry point, and test implementation | Implemented the rule-based `sonnet_prosody` checker for sonnet structure, hendecasyllable metre, stress type, rhyme keys, scheme comparison, and uncertainty flags, under Leonardo's approved A1 checkpoint plan. | `src/sonnet_evaluation/sonnet_prosody.py`, `scripts/check_sonnet_prosody.py`, `tests/test_sonnet_prosody.py` |

## 2026-09-13 Phase A2 checker validation

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Ground-truth build, validation tooling, and checker hardening | Built the frozen Dante and Petrarch ground truth from the V6 corpus, implemented the validation metrics and review packet, read the rhyme classes from the frozen texts, and fixed diacritic, silent-h, and rhyme-equivalence defects found by the validation. | `data/metadata/sonnet_prosody_ground_truth_v1.json`, `src/sonnet_evaluation/sonnet_prosody_validation.py`, `scripts/validate_sonnet_prosody.py`, `reports/sonnet_prosody_validation_v1.md` |

## 2026-09-13 Phase A3 retroactive scoring

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Sealed-output scoring and paired analysis | Implemented and ran the retroactive scoring of the 4,976 sealed Stage-3 and DPO outputs, with paired comparisons and discordant-outcome tests. | `src/sonnet_evaluation/sonnet_prosody_sealed.py`, `scripts/score_sealed_sonnet_prosody.py`, `reports/sonnet_prosody_retroactive_v1.md` |

## 2026-09-13 A2 packet review

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Packet review at Leonardo's request | Rechecked all 54 metre flags and the one rhyme-scheme mismatch with word-level diagnostics; confirmed every flag is conservative, found no definite checker error, and recorded one reason-set limitation. | `reports/sonnet_prosody_review_outcome_v1.md` |

## 2026-09-13 Form-targeted LoRA pilot implementation

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Pilot implementation and tests | Wrote the approved pilot plan, the V8 train encoder, the one-arm LoRA trainer, the matched validation generation, the scoring pipeline, and their tests. | `docs/form_targeted_lora_pilot_plan.md`, `src/sonnet_training/form_targeted_data.py`, `src/sonnet_training/form_targeted_lora.py`, `src/sonnet_analysis/form_targeted_validation.py`, `scripts/encode_form_targeted_v8_data.py`, `scripts/train_form_targeted_lora.py`, `scripts/generate_form_targeted_validation.py`, `scripts/score_form_targeted_validation.py` |

## 2026-09-13 Form-targeted LoRA pilot run

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Pilot run, verification, and corrected scoring | Ran the V8 encoding, LoRA training, matched validation generation, and scoring on the prepared H100; found and fixed a paired-direction bug in the scoring script; recorded the null result. | `reports/form_targeted_lora_pilot_v1.md`, `reports/form_targeted_lora_pilot_v1.json` |

## 2026-09-13 V8 Stage-3 retrain implementation

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Full-weight retrain implementation | Wrote the retrain plan, the window-plan and safeguarded trainer modules, candidate generation and merged evaluation, the scripts, and the tests, under the review requirements. | `docs/v8_stage3_retrain_plan.md`, `src/sonnet_training/v8_stage3_retrain_data.py`, `src/sonnet_training/v8_stage3_retrain.py`, `scripts/train_v8_stage3_retrain.py` |

## 2026-09-13 V8 Stage-3 retrain run

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Retrain run, defect diagnosis, and reports | Ran the Stage-2 download, encoding, safeguarded full-weight training, candidate generation, and scoring; diagnosed and fixed the causal-label alignment defect that invalidated the LoRA pilot; recorded the null result. | `reports/v8_stage3_retrain_v1.md`, `reports/v8_stage3_retrain_v1.json` |

## 2026-09-13 Verifier-labelled DPO implementation

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Preference build and pipeline implementation | Wrote the verifier preference builder, the verifier DPO config, loader, and scripts, and the tests under the approved plan; built and audited the frozen 425-pair dataset from the existing candidates. | `docs/verifier_labelled_dpo_plan.md`, `src/sonnet_evaluation/verifier_preferences.py`, `src/sonnet_training/verifier_labelled_dpo.py`, `scripts/build_verifier_preferences.py` |

## 2026-09-13 Verifier-labelled DPO run

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Verifier DPO run, scoring, and report | Ran the adapter training and the matched validation generation on the H100, scored the grid locally, found the partial signal and its failed-line caveat, and recorded them. | `reports/verifier_labelled_dpo_v1.md`, `reports/verifier_labelled_dpo_v1.json` |

## 2026-09-13 Rhyme lexicon and plan-then-poem plan

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Plan drafting, review revision, and rhyme lexicon | Wrote the plan-then-poem pilot plan, incorporated the independent review changes, implemented and built the V8 train rhyme lexicon (1,220 keys from 228,164 endings), and added its tests. | `docs/plan_then_poem_plan.md`, `src/sonnet_evaluation/rhyme_lexicon.py`, `data/metadata/rhyme_lexicon_v1.json` |

## 2026-09-13 Plan-then-poem inference test implementation

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Implementation and tests | Wrote the planned and control prompt generation, the plan-adherence metrics, the scoring pipeline, and the tests under the reviewed plan. | `src/sonnet_analysis/plan_then_poem_validation.py`, `scripts/generate_plan_then_poem_validation.py`, `scripts/score_plan_then_poem_validation.py` |

## 2026-09-13 Plan-then-poem inference test run

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Inference test run and scoring | Ran the planned and format-control generation on the H100, scored adherence and form locally, and recorded the train-plan-following-first reading. | `reports/plan_then_poem_validation_v1.md`, `reports/plan_then_poem_validation_v1.json` |

## 2026-09-13 Plan-following SFT implementation

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Plan, implementation, and tests | Revised the plan-following plan after the independent review (mismatched control, echo detection, position-wise adherence, fixed retry rules); implemented the response-masked trainer with merged-model save and the extended evaluation; added the tests. | `docs/plan_following_sft_plan.md`, `src/sonnet_training/plan_following_sft.py`, `scripts/train_plan_following_sft.py`, `scripts/score_plan_then_poem_validation.py` |

## 2026-09-13 Plan-following SFT run

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | SFT run, evaluation, and analysis | Ran the SFT and the three-condition evaluation, scored adherence and form, identified the opening-line plan-construction flaw and the blind-copying signature, and recorded the reading. | `reports/plan_following_sft_v1.md`, `reports/plan_following_sft_v1.json` |

## 2026-09-13 Constrained plan validation implementation

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Implementation and tests | Wrote the opening-anchored plan builder, the deterministic ending repair, the generation and scoring scripts, and the tests under the approved step. | `docs/constrained_plan_validation_plan.md`, `src/sonnet_analysis/constrained_plan_validation.py`, `scripts/generate_constrained_plan_validation.py` |

## 2026-09-13 Constrained plan validation run

| Provider and model | Reasoning effort | Role | Completed contribution | Supporting evidence |
| --- | --- | --- | --- | --- |
| OpenCode Go deepseek-v4.1-flash | unknown | Run and scoring | Built the anchored plans, ran the constrained generation and repair, scored scheme compliance, adherence, repair distance, and form against the baseline, and recorded the form-objective-met reading. | `reports/constrained_plan_validation_v1.md`, `reports/constrained_plan_validation_v1.json` |
