# Minerva QLoRA Hardware And License Gate

Date assessed: 2026-08-03

## Decision

Use `sapienzanlp/Minerva-3B-base-v1.0` as the pretrained-model comparison
candidate. Do not attempt to fine-tune `sapienzanlp/Minerva-7B-base-v1.0` on
this laptop.

The 3B selection is conditional on the next environment gate: its Hugging Face
repository requires authenticated access, and the required QLoRA packages are
not yet installed. No weights were downloaded and no environment was changed
while making this decision.

## Measured Local Capacity

| Resource | Measured value | Relevance |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU | Single local training GPU |
| GPU memory | 6,144 MiB total; 5,700 MiB free at assessment | Limits quantized base weights, activations, and training workspace |
| Driver | 580.173.02 | CUDA-capable PyTorch is available |
| System memory | 62 GiB total; 43 GiB available | Sufficient for model download, preprocessing, and ordinary CPU staging |
| Swap | 19 GiB total; effectively unused | Emergency headroom only, not a training-memory solution |
| Disk | 1.1 TiB free | Sufficient for model snapshots, local caches, checkpoints, and outputs |

The existing Python environment has PyTorch, but does not yet have
`transformers`, `peft`, `bitsandbytes`, `accelerate`, or `datasets` installed.
The missing packages are a later, explicit environment decision; they are not a
hardware failure.

## Official Model Evidence

| Candidate | Official revision at assessment | Published license and access | Architecture and scale |
| --- | --- | --- | --- |
| [`Minerva-7B-base-v1.0`](https://huggingface.co/sapienzanlp/Minerva-7B-base-v1.0) | `ff16836b81e75ae299c01fd6c797115c9935907d` | Apache-2.0; public, not gated | Mistral causal LM; 7,399,018,496 published F16 parameters; 4,096-token context |
| [`Minerva-3B-base-v1.0`](https://huggingface.co/sapienzanlp/Minerva-3B-base-v1.0) | `129ae5366bae3611a1c9f8c68606c38b7de8b055` | Apache-2.0; repository reports `gated: auto`, so authenticated access/acceptance is required before download | Mistral causal LM; 2,894,236,160 published BF16 parameters; 16,384-token context |

Both are Italian-English base models from Sapienza NLP. The 7B model card
reports 1.14 trillion Italian pretraining tokens, 1.14 trillion English tokens,
and 200 billion code tokens. The comparison should therefore be described as a
pretrained-model baseline, not as another from-scratch model.

Model metadata sources, accessed on 2026-08-03:

- https://huggingface.co/api/models/sapienzanlp/Minerva-7B-base-v1.0
- https://huggingface.co/sapienzanlp/Minerva-7B-base-v1.0/raw/main/README.md
- https://huggingface.co/api/models/sapienzanlp/Minerva-3B-base-v1.0

Any released adapter, model card, or comparison report must retain the model
identifier, revision, Apache-2.0 license notice, and source link. The 3B
repository's access conditions must also be recorded at download time.

## QLoRA Fit Assessment

QLoRA stores the frozen base model in 4-bit form and trains only small LoRA
adapter matrices. This avoids storing full gradients and AdamW moments for all
base-model weights, but it still requires GPU memory for quantized weights,
dequantization workspace, activations, logits, and the adapter optimizer.

| Candidate | 4-bit base-weight lower bound | Assessment on 6 GiB VRAM |
| --- | ---: | --- |
| Minerva 7B | about 3.45 GiB before quantization metadata and runtime workspace | **Reject.** Real 4-bit runtime storage plus training activations and temporary buffers leaves too little headroom for reliable backpropagation. CPU/device offloading would not make this a practical local QLoRA training configuration. |
| Minerva 3B | about 1.35 GiB before quantization metadata and runtime workspace | **Conditional pass.** A conservative run with NF4 4-bit quantization, batch size 1, 512-token sequences, gradient accumulation, gradient checkpointing, and a paged 8-bit adapter optimizer should fit. A one-batch GPU calibration is still required before committing to a full run. |

The 7B decision is a memory decision, not a quality judgment. It remains the
stronger candidate in principle, but its training workspace does not fit this
GPU reliably. The 3B model preserves the project’s key comparison: a locally
fine-tuned, Italian-pretrained open model versus the completed from-scratch
model under the same sonnet corpus and evaluation protocol.

## Next Gate

Before any model download or fine-tuning:

1. The user accepts the Minerva-3B Hugging Face access terms and authenticates
   locally.
2. Add the explicitly approved QLoRA dependencies in a project-scoped
   environment or documented existing environment.
3. Run a one-batch 4-bit QLoRA calibration at batch size 1 and context length
   512, with no corpus or recipe sweep.
4. Record the actual peak GPU memory and select the resulting fixed QLoRA
   training configuration before starting a long run.

## Calibration Result

The approved local calibration completed on 2026-08-03 after the authorized
Minerva 3B repository access and local Hugging Face authentication. It loaded
the exact recorded revision in 4-bit NF4, attached rank-16 adapters to all seven
recorded projection types, enabled gradient checkpointing, and completed one
forward pass, backward pass, and `PagedAdamW8bit` adapter update at batch size
one and context length 512.

| Measurement | Result |
| --- | ---: |
| Peak allocated CUDA memory | 2,462.1 MiB |
| Peak reserved CUDA memory | 2,818.0 MiB |
| Trainable LoRA parameters | 26,214,400 |
| Trainable fraction of reported quantized representation | 1.68% |
| Calibration loss | 3.2253 |

The calibration passed without an out-of-memory error and leaves substantial
VRAM headroom relative to the 6,144 MiB GPU. `sentencepiece 0.2.2` was needed
by the Minerva tokenizer and is now pinned in
`requirements/minerva_qlora.txt`. The next checkpoint may define one fixed
QLoRA sonnet fine-tuning recipe; it does not authorize a hyperparameter sweep.
