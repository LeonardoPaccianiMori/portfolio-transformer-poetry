# Teaching a Tiny Transformer to Write Classical Italian Poetry

This project builds a small GPT-style causal language model from scratch in PyTorch and trains it to generate classical Italian poetry.

The goal is educational and experimental: understand the internals of language modeling, transformer training, tokenization, evaluation, and generation under local laptop-scale GPU constraints. The model is not intended to be a general-purpose language model.

## Project Goals

- Build an inspectable causal transformer implementation in PyTorch.
- Construct a reproducible corpus of early Italian prose and poetry from reusable public sources.
- Compare tokenization, architecture, training, and decoding choices on a small model.
- Evaluate generated text with fixed prompts, automatic metrics, qualitative notes, and memorization checks.
- Document the process clearly enough to support a technical portfolio artifact.

## Research Question

Can a tiny transformer first learn historical Italian from prose and then acquire poetic style through focused post-training on 13th- and 14th-century Italian poetry?

## Planned Scope

The project will progress in stages:

1. Corpus source audit and data provenance.
2. Reproducible raw/interim/processed data pipeline.
3. Character-level baseline language model.
4. Classic GPT-style causal transformer.
5. BPE tokenization experiments.
6. Modern transformer components such as RMSNorm, RoPE, SwiGLU, weight tying, and improved training schedules.
7. Staged prose pre-training followed by poetry post-training.
8. Repeatable evaluation harness and comparison reports.
9. Lightweight local generation demo.

## Constraints

- Python and PyTorch.
- From-scratch model implementation for the core transformer.
- Local training target: NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM.
- Preserve original spelling and punctuation unless a specific experiment says otherwise.
- Keep raw, cleaned, and processed data conceptually separate.
- Track source provenance and licensing before using texts for training.

## Status

Initial repository setup. No model, data pipeline, or training artifacts have been added yet.
