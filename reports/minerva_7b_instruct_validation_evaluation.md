# Minerva 7B Instruct Validation Baseline Evaluation

## Question

This validation-only baseline tested whether the published Minerva 7B Instruct
checkpoint is already a credible Italian-sonnet quality parent under the
project's fixed opening-line continuation task. It did not use the final test
set and did not train or modify the model.

## Protocol

- Model: `sapienzanlp/Minerva-7B-instruct-v1.0`, revision
  `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`.
- Loading: 4-bit NF4 quantization on the local RTX 3060 Laptop GPU.
- Data: eight frozen V5 validation openings from eight authors and five
  centuries.
- Conditioning: the published chat template, an explicit fourteen-line sonnet
  request, and the exact opening line as assistant prefill.
- Decoding: seed 4242, temperature 0.8, top-k 50, 512-token ceiling, and
  decoder stopping after thirteen completed continuation lines.
- Qualification threshold: at least 5/8 generally grammatical outputs, at
  least 5/8 outputs sustaining a topic for seven generated lines, and no more
  than 1/8 severe collapses.

The fixed per-output judgments and raw samples are retained locally and
excluded from the public tree.

## Results

| Measurement | Result | Required |
| --- | ---: | ---: |
| Exact opening and controlled 14-line form | 6/8 | diagnostic |
| Outputs reaching the 512-token ceiling | 2/8 | diagnostic |
| Generally grammatical Italian | 2/8 | at least 5/8 |
| Topic or argument sustained for seven lines | 7/8 | at least 5/8 |
| Severe repetition or generation collapse | 2/8 | no more than 1/8 |
| Mean repeated character 4-gram ratio | 0.2941 | diagnostic |

Quantized inference peaked at 4,445.9 MiB allocated and 4,568.0 MiB reserved.
This establishes that the checkpoint fits for local 4-bit inference. It does
not establish that a representative QLoRA optimizer update will fit.

The strongest outputs, especially Alfieri and Dante, are coherent enough to
show meaningful instruction-following and ordinary Italian competence. Most
outputs sustain the opening topic, and six stop in the intended controlled
form. However, Giacomo, Stampa, and Colonna contain persistent grammatical
defects, while several outputs add forbidden labels such as `Secondo verso`.

The two token-ceiling failures are substantial. Petrarca abandons the poem
after three malformed continuation lines and repeats a prose appraisal of the
supposed sonnet. Andreini produces eleven malformed poetic lines, then repeats
near-identical explanatory paragraphs. These are generation collapse rather
than minor formatting errors.

## Decision

**The prompt-only 7B Instruct quality-parent gate fails.**

The model passes topic continuity but misses both decisive language criteria:
2/8 outputs are generally grammatical rather than the required 5/8, and 2/8
collapse rather than the permitted maximum of 1/8. The failure is large enough
that changing one borderline judgment would not reverse the decision.

This result does not reject Minerva 7B Instruct as a fine-tuning parent. It is
materially more coherent and controllable than the tested 3B Base conditions,
and its failures are concentrated enough to motivate conservative adaptation
on a corrected sonnet corpus. The next hardware checkpoint remains the one
predeclared rank-8 QLoRA calibration. Full 7B training is permitted only if
that optimizer-step calibration leaves at least 512 MiB of measured CUDA
headroom. Any later full recipe must use the corrected V6 corpus and be frozen
before training.
