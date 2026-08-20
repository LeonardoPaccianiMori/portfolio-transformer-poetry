# Minerva 7B Historical Adaptation And V6 Specialization: Final Evaluation

Evaluation completed: 2026-08-09.

## Lineage

1. Pinned `sapienzanlp/Minerva-7B-instruct-v1.0` loaded unquantized in FP16.
2. Rank-8 attention LoRA adapted on historical Italian prose with PAISÀ replay.
3. Historical step 4,000 selected after passing preservation gates.
4. The same adapter was specialized on V6 train sonnets with a fresh AdamW
   optimizer and chat-formatted full-sonnet targets.
5. Epoch 4 selected from epochs 3, 4, and 5 using frozen validation prompts.
6. Final test opened only after the epoch-4 checkpoint and candidate evidence
   were hash-frozen.

The base model remained frozen throughout both training stages.

## Quantitative Result

- Stage B starting V6 validation loss: `3.493399`.
- Selected epoch-4 validation loss: `3.171254`.
- V6 final-test loss across 197 poems: `3.212791`.
- Selected adapter SHA-256: `aff3f2c4d193ce880ec9c7a6df6373f433001662c3ca78d7f915890733cb0df3`.
- Final generation artifacts: 20 outputs from 10 fixed prompts and two seeds.

## Acceptance Result

| Requirement | Result | Threshold | Status |
|---|---:|---:|---|
| Exact prompt and 14-line form | 20/20 | at least 18/20 | pass |
| Generally grammatical Italian | 8/20 | at least 12/20 | fail |
| Seven-line topic continuity | 20/20 | at least 10/20 | pass |
| Severe collapse | 5/20 | at most 2/20 | fail |
| High-risk memorization | 0/20 | exactly 0/20 | pass |

The model does not pass the complete predeclared acceptable-quality gate. Its
failure is narrower than the from-scratch models: structure, conditioning,
topic continuity, and memorization all pass; remaining defects are persistent
grammar errors and occasional repetition collapse.

## Evidence

- Selection: `reports/minerva_7b_v6_candidate_selection.md`
- Form controls: `reports/minerva_7b_v6_final_acceptance_controls.md`
- Generation metrics: `reports/minerva_7b_v6_final_generation_metrics.md`
- Memorization: `reports/minerva_7b_v6_final_memorization.md`
- Completed output-level qualitative review: retained locally and excluded
  from the public tree
- Frozen selection record: `configs/minerva_7b_v6_selected_adapter.json`
