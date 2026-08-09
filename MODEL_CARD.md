# Model Card: Minerva 7B Historical-Italian V6 Sonnet LoRA

## Model Summary

This is the selected experimental model from the repository's pretrained-model
branch. It adapts the pinned
[`sapienzanlp/Minerva-7B-instruct-v1.0`](https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0)
checkpoint to historical Italian prose and then to opening-line-conditioned
sonnet composition.

The base model remains frozen. The trainable component is a rank-8 LoRA adapter
on `q_proj`, `k_proj`, `v_proj`, and `o_proj`, containing 6,815,744 parameters.
Training used unquantized FP16 base weights. The local demo may load the same
frozen base in 4-bit NF4 for inference without changing the selected adapter.

This model is the strongest system tested in the project, but it did not pass
the complete acceptable-quality gate. It is an experimental portfolio and
research artifact, not a production sonnet generator.

## Identification

- Project model name: `minerva_7b_v6_selected_epoch_04`
- Parent: `sapienzanlp/Minerva-7B-instruct-v1.0`
- Parent revision: `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`
- Parent scale: approximately 7.4 billion parameters
- Adapter type: rank-8 attention LoRA, alpha 16
- Selected Stage A checkpoint: update 4,000
- Selected Stage B checkpoint: epoch 4, update 744
- Selected adapter SHA-256:
  `aff3f2c4d193ce880ec9c7a6df6373f433001662c3ca78d7f915890733cb0df3`
- Selection record: `configs/minerva_7b_v6_selected_adapter.json`
- Context length used for adaptation: 512 tokens
- Primary language: Italian, including historical orthography

## Intended Task

Input is one exact opening line. Output is a continuation intended to produce a
fourteen-line classical-Italian sonnet. The published evaluation and demo stop
decoding after thirteen continuation lines.

Decoder-controlled line count is not evidence that the model learned
hendecasyllabic metre, rhyme, stanza structure, or literary quality. Those
properties were not certified by this project.

## Training Lineage

### Stage A: Historical Italian Adaptation

- Data: `pretraining_historical_italian_v2`, 36 prose sources
- Mixture per update: seven 512-token historical windows and one 512-token
  PAISÀ modern-Italian replay window
- Optimizer: AdamW with only adapter parameters trainable
- Peak learning rate: `2e-5`
- Schedule: 3 percent warmup, cosine decay to `2e-6`
- Completed updates: 4,000, stopped by patience
- Historical validation loss: `3.262937` to `3.187692`
- Modern-Italian and instruction-preservation gates: passed

### Stage B: V6 Sonnet Specialization

- Data: 1,481 V6 training sonnets and 190 validation sonnets
- Supervision: complete assistant sonnet; system/user prompt tokens masked
- Optimizer: fresh AdamW; base model still frozen
- Peak learning rate: `1e-5`
- Schedule: 5 percent warmup, cosine decay to `1e-6`
- Training stopped after epoch 7 through patience
- Candidate selection: epochs 3, 4, and 5 compared on frozen validation prompts
- Selected epoch 4 validation loss: `3.171254`

Epoch 4 was selected for lower repetition and better qualitative stability than
the neighboring candidates while remaining within `0.0063` validation loss of
epoch 5. Final-test material was unavailable until this choice was hash-frozen.

## Evaluation

The unopened V6 final test contains 197 poems. Generation evaluation used ten
fixed openings, two seeds, temperature 0.8, top-k 50, and 20 outputs.

| Requirement | Result | Required | Status |
| --- | ---: | ---: | --- |
| Exact opening and controlled 14-line form | 20/20 | at least 18/20 | pass |
| Generally grammatical Italian | 8/20 | at least 12/20 | fail |
| Topic continuity for at least seven lines | 20/20 | at least 10/20 | pass |
| Severe repetition or collapse | 5/20 | at most 2/20 | fail |
| High-risk training overlap | 0/20 | exactly 0/20 | pass |

Final-test loss across all 197 poems was `3.212791`. Full evidence is in
`reports/minerva_7b_v6_final_evaluation.md`.

## Intended Uses

- Reproducing the repository's controlled evaluation
- Studying LoRA domain adaptation and task specialization
- Demonstrating local quantized inference with a frozen adapter
- Inspecting successful and failed historical-Italian generations
- Educational comparison with the repository's from-scratch transformer

## Out-of-Scope Uses

- Publishing generated text as authentic historical poetry
- Literary scholarship, attribution, translation, or textual criticism
- Automated grading of Italian grammar or poetry
- Production deployment without stronger safety and quality evaluation
- Claims of learned metre or rhyme based only on fourteen-line output

## Limitations

- The complete quality gate failed on grammar and collapse.
- Five of twenty final outputs show severe repetition or degeneration.
- Historical-looking spelling can conceal malformed syntax.
- The training corpus is concentrated in a limited set of authors and digital
  editions.
- The model may reproduce biases, errors, or modern editorial choices inherited
  from its external pretraining and project corpora.
- Memorization checks detect long surface overlap, not every form of recall.
- Genuine sonnets may have appeared in the parent model's external pretraining.
- Final generation controls test line count and conditioning, not metre or rhyme.

## Licensing And Distribution

The Minerva parent is recorded as Apache-2.0 and must retain its model ID,
revision, source link, Sapienza NLP attribution, and license notice.

Project adaptation data includes public-domain text, Italian Wikisource
material with CC BY-SA/GFDL records, Liber Liber editions under CC BY-NC-SA
4.0, and PAISÀ replay under CC BY-NC-SA terms. Required credits are indexed in
`DATA_SOURCES_AND_ATTRIBUTION.md`.

The selected adapter is not committed to the public repository. Stage A used
PAISÀ replay, and the project policy does not publish PAISÀ-derived checkpoints.
The locally retained adapter is for the project's non-commercial research and
demonstration workflow. This model card does not grant rights beyond the
underlying model, dataset, and repository terms.

## Local Demo

With the authenticated Minerva cache and retained local adapter artifacts, run:

```bash
.venv/bin/python -u scripts/serve_sonnet_demo.py --device cuda:0
```

Cached startup is typically 20 seconds to several minutes. A 6 GiB RTX 3060
generated the verified 205-token acceptance sample in 18.4 seconds. Open
`http://127.0.0.1:8000` after the terminal prints `demo | ready`.

## Related Evidence

- `docs/minerva_7b_staged_lora_protocol.md`
- `reports/minerva_7b_historical_fp16_lora_result.md`
- `reports/minerva_7b_v6_candidate_selection.md`
- `reports/minerva_7b_v6_final_evaluation.md`
- `reports/final_model_comparison.md`
