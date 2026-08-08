# Minerva 7B Instruct Unquantized FP16 Validation Evaluation

## Question

The local baseline loaded Minerva 7B Instruct in 4-bit NF4 and failed the
prompt-only quality gate. This remote validation-only control tests the exact
same model revision, eight validation openings, prompt, seed, and decoding
settings with unquantized FP16 weights. It isolates quantization as the changed
factor and does not use the final test set.

## Protocol

- Model: `sapienzanlp/Minerva-7B-instruct-v1.0`, revision
  `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`.
- GPU: Quadro RTX 8000 with 48 GiB VRAM.
- Weight loading: unquantized FP16 parameters and FP16 computation.
- Data and prompt: the same frozen eight V5 validation openings and published
  chat-template instruction used by the NF4 baseline.
- Decoding: seed 4242, temperature 0.8, top-k 50, 512-token ceiling, and
  decoder stopping after thirteen completed continuation lines.
- Quality gate: at least 5/8 generally grammatical outputs, at least 5/8
  seven-line topic continuations, and no more than 1/8 severe collapses.

## Results

| Measurement | NF4 | FP16 | FP16 requirement |
| --- | ---: | ---: | ---: |
| Controlled 14-line form | 6/8 | 8/8 | diagnostic |
| 512-token ceiling | 2/8 | 0/8 | diagnostic |
| Mean repeated character 4-gram ratio | 0.2941 | 0.2262 | diagnostic |
| Generally grammatical Italian | 2/8 | 1/8 | at least 5/8 |
| Seven-line topic continuation | 7/8 | 8/8 | at least 5/8 |
| Severe collapse | 2/8 | 1/8 | no more than 1/8 |

FP16 inference completed in 89 seconds including the initial model download.
Peak CUDA allocation was 14,191.8 MiB and peak reservation was 14,320.0 MiB.

Unquantized loading improves mechanical control in this fixed sample: every
output reaches fourteen lines, none runs to the token ceiling, and surface
repetition decreases. It does not recover reliable grammar. Giacomo is the one
generally grammatical continuation. The other outputs contain persistent
agreement, reference, or clause-construction defects. Dante abandons the poem
for fabricated analysis, several outputs add forbidden verse labels, and
Petrarca repeats heart-and-breeze clauses into overt collapse.

## Decision

**The unquantized FP16 prompt-only quality-parent gate fails.**

The decisive grammatical count is 1/8 rather than the required 5/8. Because
the FP16 outputs differ substantially from NF4 but do not improve grammar, the
earlier failure cannot reasonably be assigned to 4-bit quantization alone.

The result does not reject 7B Instruct as an adaptation parent. FP16 provides
more reliable stopping and the separate LoRA calibration leaves ample memory.
The next model-training decision must therefore concern the data and adapter
recipe, not a different quantizer. No full run is authorized until the
corrected V6 corpus is built and the exact conservative FP16 LoRA protocol is
predeclared.
