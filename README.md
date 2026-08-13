# Teaching Transformers To Write Classical Italian Sonnets

This repository is a learning-by-doing language-model project: implement a
GPT-style causal transformer from scratch in PyTorch, build licensed Italian
corpora, train under laptop-scale constraints, and compare the result with
parameter-efficient adaptation of open Minerva models.

The project is complete as an experimental pipeline. No tested model passed the
full acceptable-quality gate. The final system is a three-stage full-weight
Minerva 7B V7 model plus one bounded AI-judged DPO adapter. It produced modest,
replicated completion/surface gains but remains an unreliable sonnet writer.

The V7 study retained midpoint and selected-boundary BF16 states, analyzed
weights, embeddings, hidden representations, losses, and more than 20,000
validation generations, then ran one hash-frozen test over all 1,244 held-out
openings. See the [V7 Post-Training Study](reports/minerva_7b_v7_post_training_study.md).

## Final Result

| System | Form | Grammar | Topic | Collapse | High-risk overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 70M from scratch | 20/20 | 0/20 | 0/20 | 20/20 | 0/20 |
| Minerva 3B QLoRA | 20/20 | 2/20 | 13/20 | 7/20 | 0/20 |
| Minerva 7B staged LoRA | 20/20 | 8/20 | 20/20 | 5/20 | 0/20 |
| Required | at least 18 | at least 12 | at least 10 | at most 2 | exactly 0 |

`Form` means exact opening-line preservation and decoder-controlled fourteen
lines. It does not prove metre or rhyme.

The newer V7 final test uses 2,488 outputs/system rather than the legacy
20-output rubric. DPO improved the automatic surface screen from 15.07% to
17.60% (paired +2.53 points; 95% interval +0.52 to +4.50) and terminal
punctuation from 17.60% to 20.46%. Neither system had a high-risk memorization
hit. In the frozen 200-output blind literary review, DPO showed a small
historical-register gain but no reliable broad quality improvement; DPO had
3/100 moderate-clean and 0/100 strict-good outputs, versus 0/100 and 0/100 for
Stage 3. The literary review is deliberately stricter than the surface metrics.

The earlier 7B V6 path adapted a frozen Minerva Instruct parent through
historical-Italian prose LoRA followed by sonnet specialization. V7 instead
updated the full BF16 model through historical, poetry, and sonnet stages before
adding DPO. The complete comparison is in
[Final Model Comparison](reports/final_model_comparison.md).

## What This Project Covers

- Character and Unicode BPE tokenization
- Causal masks, self-attention, multi-head attention, residual blocks, and loss
- LayerNorm, RMSNorm, RoPE, SwiGLU, and weight tying
- Optimizers, warmup/cosine schedules, gradient clipping, mixed precision,
  checkpointing, and interruption-safe resume
- Public-domain, Creative Commons, and other permitted non-commercial corpora
- Source attribution, split leakage, duplicate audits, and corpus versioning
- Fixed prompts, automatic controls, memorization heuristics, blinded review,
  and neighboring-checkpoint selection
- 4-bit QLoRA and unquantized FP16 LoRA on Minerva 3B and 7B
- A failed human-calibration gate plus honestly scoped AI-judge-distillation DPO
- Full-weight checkpoint, embedding, activation, and behavior-change analysis
- A standard-library local web server and responsive UI for the selected system

The core from-scratch transformer, training loop, and decoding logic remain
inspectable project code rather than wrappers around Hugging Face models.
Hugging Face, PEFT, Accelerate, and bitsandbytes are used only for the external
Minerva comparison.

## Local Demo

The demo requires the retained local Stage-3 BF16 archive and DPO adapter. It
loads Stage 3 transiently in 4-bit NF4 for inference on a 6 GiB GPU while
keeping the selected adapter unchanged. This is a deployment approximation:
the authoritative checkpoint remains full BF16, and all V7 research evidence
used unquantized BF16 on the H100.

Expected startup is several minutes because the local 14.8 GB checkpoint must
be read and quantized. Progress is printed during model loading; the ready URL
appears at the end.

```bash
.venv/bin/python -u scripts/serve_sonnet_demo.py --device cuda:0
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Enter one opening line,
choose temperature and seed, and generate a controlled fourteen-line output.
Use `--legacy-v6` to load the prior V6 demo.

For a UI-only check that does not allocate model weights:

```bash
python3 -u scripts/serve_sonnet_demo.py --static-only
```

Startup is normally under two seconds. Generation intentionally returns an
unavailable status in this mode.

## Verification

The complete CPU test suite normally finishes in 15–30 seconds:

```bash
python3 -m pytest
```

Long-running corpus, training, fine-tuning, benchmarking, and evaluation CLIs
print flushed progress, elapsed time, ETA, validation results, and checkpoint
events by default.

## Repository Map

| Path | Responsibility |
| --- | --- |
| `src/sonnet_model/` | From-scratch transformer modules and generation |
| `src/sonnet_training/` | Training, fine-tuning, checkpoints, and schedules |
| `src/sonnet_corpus/` | Acquisition, cleaning, manifests, splits, and encoding |
| `src/sonnet_evaluation/` | Metrics, memorization, selection, and Minerva evaluation |
| `src/sonnet_analysis/` | V7 checkpoint, dynamics, representation, and behavioral analysis |
| `src/sonnet_demo/` | Local selected-model web server |
| `scripts/` | Reproducible command-line entry points |
| `configs/` | Frozen prompts, selections, and experiment policies |
| `data/metadata/` | Corpus and attribution metadata |
| `reports/` | Public experiment evidence, including failed samples |
| `demo/` | Responsive local demo interface |
| `tests/` | Unit and integration tests |

## Primary Artifacts

- [Technical Report](reports/technical_report.md)
- [Model Card](MODEL_CARD.md)
- [Final Model Comparison](reports/final_model_comparison.md)
- [Minerva 7B V7 Post-Training Study](reports/minerva_7b_v7_post_training_study.md)
- [Minerva 7B V7 AI-Judged DPO](reports/minerva_7b_v7_ai_judged_dpo.md)
- [Minerva 7B V6 Final Evaluation](reports/minerva_7b_v6_final_evaluation.md)
- [Minerva Judge Gate](reports/minerva_3b_judge_gate.md)
- [Data Sources And Attribution](DATA_SOURCES_AND_ATTRIBUTION.md)

## Data And Licensing

The project does not restrict discovery to public-domain text. A source may be
used when its terms explicitly permit this non-commercial research/training
workflow and every attribution, share-alike, notice, source-link, or downstream
restriction is recorded.

The data and model lineage includes public-domain works, Italian Wikisource,
Liber Liber CC BY-NC-SA editions, PAISÀ CC BY-NC-SA text, and Apache-2.0 Minerva
parents. See [Data Sources And Attribution](DATA_SOURCES_AND_ATTRIBUTION.md) for
the exact records.

The final checkpoints and adapters are not committed publicly because their
lineage includes PAISÀ replay and the project policy withholds PAISÀ-derived
checkpoints. This
repository does not apply one blanket license over third-party data, models, or
generated artifacts; downstream users must follow each recorded source term.

## Honest Scope

This is not a production LLM and not a claim of solved poetry generation. The
from-scratch branch demonstrates transformer and data-pipeline engineering. The
Minerva branch demonstrates practical transfer, full-weight curriculum
adaptation, model-change analysis, and preference optimization. The fixed
evaluations show exactly where the systems improve and still fail.
