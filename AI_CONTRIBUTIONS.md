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
