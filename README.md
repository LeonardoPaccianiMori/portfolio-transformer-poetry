# Teaching Transformers to Write Classical Italian Sonnets

This completed experimental project asks what can be learned by building a small
language model from first principles and then adapting an existing Italian 7B
model under a controlled curriculum. It has two distinct arcs:

1. a roughly 70-million-parameter GPT-style causal transformer implemented and
   trained from scratch in PyTorch; and
2. staged full-weight adaptation of the existing Minerva 7B parent, followed by
   one bounded AI-judged DPO experiment.

The final 7B system was **not** trained from scratch. It starts from the pinned
Minerva parent and updates it through historical prose, non-sonnet poetry, and
sonnet stages before attaching a small DPO adapter.

The experimental pipeline is complete. No tested system passed the complete
quality gate. The final DPO system replicated narrow automatic terminal-
punctuation and surface-screen improvements, while meta-text and blind visible-
completion differences remained uncertain. Both Stage 3 and DPO produced
`0/100` strict-good outputs in the sealed blind literary review. The project is
evidence of model engineering, evaluation design, and candid failure
analysis—not a solved-poetry claim.

## Two Project Arcs

### Transformer learning from scratch

The repository implements tokenization, batching, causal attention, multi-head
attention, residual blocks, normalization, training, checkpointing, and
autoregressive decoding directly in PyTorch. Controlled one-seed experiments
compare classic and modern components, including ReLU versus SwiGLU. Architecture
comparisons use five sampled validation batches per evaluation and are
descriptive rather than definitive.

### Minerva 7B staged adaptation

The second arc starts from `sapienzanlp/Minerva-7B-instruct-v1.0` at pinned
revision `d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`. Three full-weight BF16 stages
adapted the parent to historical/literary Italian, non-sonnet poetry, and V7
sonnets. Deterministic PAISÀ modern-preservation replay supplied exactly 5% of
target-token exposure in every stage; this is not 5% of unique documents,
examples, or corpus size.

The completed V7 path is distinct from the earlier prospective PAISÀ rescue
curriculum documented in `data/metadata/paisa_attribution.md`.

## Final Evidence

| Evidence | Result | Qualification |
| --- | --- | --- |
| Stage runtimes | 15,495 s; 7,547 s; 3,115 s | Measured completed runs |
| Three-stage cost | about $10.65 | Qualification-based projection, not the final measured bill |
| DPO runtime / peak VRAM | 148.6 s / 14.81 GiB | Measured |
| DPO cost | about $0.093 | Estimated |
| Sealed-test runtime / cost | 2,970.6 s / about $1.967 | Runtime measured; cost estimated |
| Human/AI calibration | 12/20 | Failed the calibration gate; the work remains AI-judged |
| Strict-good literary outputs | 0/100 for both final systems | Frozen blind review |

The one-time final test used all 1,244 sealed openings, two seeds, and both
systems: 4,976 outputs total. DPO increased the automatic surface screen from
15.07% to 17.60% (paired `+2.53` points; 95% interval `+0.52` to `+4.50`) and
terminal punctuation from 17.60% to 20.46%. In the 200-output blind literary
review, only the historical-register interval excluded zero. Grammar, poetic
quality, sonnet/form, volta, and visible-completion changes remained uncertain.

Fourteen-line output is decoder-controlled. It does not establish learned
rhyme, metre, stanza structure, grammar, or literary quality.

## Repository Map

| Path | Responsibility |
| --- | --- |
| `src/sonnet_model/` | From-scratch transformer modules and generation |
| `src/sonnet_training/` | Training, adaptation, checkpoints, and schedules |
| `src/sonnet_corpus/` | Acquisition, cleaning, manifests, splits, and encoding |
| `src/sonnet_evaluation/` | Metrics, memorization, selection, and evaluation |
| `src/sonnet_analysis/` | V7 weight, representation, and behavior analysis |
| `src/sonnet_demo/` and `demo/` | Local static and selected-model demo |
| `scripts/` | Reproducible command-line entry points |
| `configs/` | Frozen experiment policies and selections |
| `data/metadata/` | Corpus and attribution metadata |
| `reports/` | Experiment evidence and aggregate reports |
| `release/` | Fail-closed public-tree and history review records |
| `tests/` | Unit, integration, release-scope, and hygiene tests |

## Public Verification Quick Start

Use Python 3.12. The public CPU verification environment is separate from the
historical GPU training environment.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/verify_public_test_scope.py
.venv/bin/python -m pytest -m "not local_artifact"
```

The public-clone suite contains 1,159 tests. Ten additional tests are explicitly
catalogued as local-only because they validate intentionally withheld research
artifacts; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

For the static-only demo check:

```bash
.venv/bin/python -u scripts/serve_sonnet_demo.py --static-only
```

Static-only generation intentionally returns an unavailable response. Full
local generation requires withheld model artifacts and is not part of public
verification. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Artifact Availability

| Artifact | Public status |
| --- | --- |
| Software, tests, configs, CI, reports, aggregate evidence | Candidates for inclusion only after affirmative manifest approval; licensing remains file-specific |
| Processed corpora | Candidates only where the release manifests affirmatively approve redistribution |
| Source and attribution metadata | Candidates subject to affirmative approval and source-specific notices |
| Gutendex catalog JSON/CSV summaries | Candidate non-authoritative third-party generated metadata with recorded provenance; not Leonardo's analysis |
| Model weights, checkpoints, adapters | Withheld pending artifact-specific specialist review |
| Raw generations, poems/openings used in evaluation, preferences, votes, annotations, mappings, tensors | Not release artifacts |

The repository has an approximately 434 MiB loose Git-object footprint before
cleanup; rights-approved processed corpora make cloning heavier than a normal
software repository. No retroactive Git LFS migration is planned.

## Data, Licensing, and Redistribution

This is a mixed-rights repository. After affirmative manifest approval,
Apache-2.0 covers only the identified Leonardo-owned software and executable
configuration described in [LICENSE.md](LICENSE.md). CC BY 4.0 covers only
affirmatively approved Leonardo-owned prose, reports, aggregate evidence,
tables, and plots. Neither grant automatically covers third-party
corpora or metadata, pretrained-model material, generated poems, model outputs,
preferences, annotations, checkpoints, adapters, or embedded third-party
passages.

PAISÀ and several other sources have source-specific terms. Check
[DATA_SOURCES_AND_ATTRIBUTION.md](DATA_SOURCES_AND_ATTRIBUTION.md),
[NOTICE](NOTICE), and the release manifests before reuse. Withholding the final
checkpoint and adapter is a conservative release policy pending separate
artifact-specific review; it is not a legal conclusion that the PAISÀ corpus
license necessarily governs model weights.

## AI Contribution

Leonardo conceived and directed the project, made executive decisions, approved
the research plan, reviewed outputs, and sometimes ran GPU work. Codex 5.5 and
later Codex 5.6 Sol substantially assisted research design, implementation,
tests, execution, and analysis. The project must not be described as
independently designed or independently implemented by Leonardo. See
[AI_CONTRIBUTIONS.md](AI_CONTRIBUTIONS.md).

## Citation and Limitations

Use [CITATION.cff](CITATION.cff) for the software citation. Primary technical
evidence is in the [V7 post-training study](reports/minerva_7b_v7_post_training_study.md),
[AI-judged DPO report](reports/minerva_7b_v7_ai_judged_dpo.md),
[model card](MODEL_CARD.md), and [final comparison](reports/final_model_comparison.md).

This is an educational research artifact, not a production LLM, literary
authority, human-aligned system, or reliable sonnet generator. The sealed test
limits the final claim more strongly than the headline improvement does.
