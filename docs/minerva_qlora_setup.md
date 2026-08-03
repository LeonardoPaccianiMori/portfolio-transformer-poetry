# Minerva 3B QLoRA Setup

The local pretrained-model comparison uses
`sapienzanlp/Minerva-3B-base-v1.0`, revision
`129ae5366bae3611a1c9f8c68606c38b7de8b055`. Its use and local-hardware
selection are recorded in
[`../reports/minerva_qlora_hardware_gate.md`](../reports/minerva_qlora_hardware_gate.md).

## Environment

Create `.venv` with `--system-site-packages`. This keeps the project-local
dependencies isolated while reusing the verified CUDA-enabled PyTorch already
installed on the laptop. Do not install another PyTorch wheel as part of this
setup.

The pinned supplementary dependencies live in
[`../requirements/minerva_qlora.txt`](../requirements/minerva_qlora.txt).

Setup checkpoint completed on 2026-08-03: `.venv` was created with
`--system-site-packages`, retaining the existing `torch 2.10.0+cu128`
installation. The pinned `transformers 4.57.1`, `peft 0.17.1`,
`accelerate 1.10.1`, `bitsandbytes 0.48.1`, `safetensors 0.6.2`, and
`huggingface_hub 0.34.4` packages import successfully with the Mistral and
QLoRA classes. `sentencepiece 0.2.2` is also required by the Minerva tokenizer
and is pinned in the requirements file. The actual one-batch CUDA calibration
must run on the laptop, not in the assistant sandbox.

## Authentication

Before model download, authenticate the local Hugging Face client using an
account that has accepted the Minerva 3B repository conditions. Use a read-only
access token. Never put that token in the repository, a command line, a config
file, or a report.

## Calibration

`scripts/calibrate_minerva_qlora.py` is a GPU-only one-batch gate, not a
fine-tuning run. It loads the exact recorded Minerva revision in 4-bit NF4,
uses double quantization and float16 computation, enables gradient
checkpointing, attaches rank-16 LoRA adapters to all Mistral attention and MLP
projection layers, and takes one paged-8-bit AdamW update.

The script locks batch size 1 and context length 512. It writes only an ignored
local JSON report with the loss, package versions, trainable adapter count, and
peak CUDA memory. The next long training run must not begin until this report
shows reliable VRAM headroom.
