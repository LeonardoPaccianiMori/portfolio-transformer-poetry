# Teaching a Tiny Transformer to Write Classical Italian Sonnets

This project builds a small GPT-style causal language model from scratch in PyTorch and trains it to generate classical Italian sonnets.

The goal is educational and experimental: understand the internals of language modeling, transformer training, tokenization, evaluation, and generation under local laptop-scale GPU constraints. The model is not intended to be a general-purpose language model.

## Project Goals

- Build an inspectable causal transformer implementation in PyTorch.
- Construct a reproducible corpus of Italian sonnets from reusable public sources.
- Compare tokenization, architecture, training, and decoding choices on a small model.
- Evaluate generated text with fixed prompts, automatic metrics, qualitative notes, and memorization checks.
- Document the process clearly enough to support a technical portfolio artifact.

## Research Question

Can a tiny transformer learn enough early Italian poetic language and sonnet structure from a curated sonnet corpus to generate plausible classical Italian sonnets?

## Planned Scope

The project will progress in stages:

1. Corpus source audit and data provenance.
2. Reproducible raw/interim/processed data pipeline.
3. Character tokenizer, random batching, and PyTorch tensor/data-loading exercises.
4. Character-level from-scratch base pretraining with a classic GPT-style causal transformer.
5. BPE tokenization experiments and direct comparison against the character baseline.
6. Modern transformer components such as RMSNorm, RoPE, SwiGLU, weight tying, and improved training schedules.
7. Corpus mixture and curriculum experiments, including core, expanded, and conditioned sonnet variants.
8. Lightweight task-format post-training for sonnet continuation or metadata-controlled generation.
9. Repeatable evaluation harness, fixed-prompt samples, memorization checks, and comparison reports.
10. Local open-source pretrained causal LM fine-tuning as a comparison baseline.
11. Lightweight local generation demo.

## Constraints

- Python and PyTorch.
- From-scratch model implementation for the core transformer.
- Local training target: NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM.
- Preserve original spelling and punctuation unless a specific experiment says otherwise.
- Track processed corpus files in the repo with source attribution; keep raw/interim extraction files temporary.
- Track source provenance and licensing before using texts for training.

## Status

Corpus-builder implementation is in progress. The current generated corpus includes processed sonnet files and metadata for the planned Wikisource sources.
