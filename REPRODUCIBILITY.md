# Reproducibility

## Public CPU verification

The supported public verification environment is Python 3.12 with the pinned
CPU packages in `requirements.txt`. It validates software behavior, corpus and
metadata contracts, aggregate evidence, release scope, and the static-only demo.
It does not reproduce the historical H100 training run or automatically
download and exercise the separately published model artifacts.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest
```

The full command runs 1,187 tests when the intentionally withheld local
research artifacts are present. A clean public clone first runs
`scripts/verify_public_test_scope.py`, then runs the complete 1,176-test public
subset with `python -m pytest -m "not local_artifact"`. The exact eleven local-only
node IDs are frozen in `release/local_only_test_allowlist.txt`; CI fails if a
marker is added, removed, or moved without updating that reviewed boundary.

Run the UI/server contract without model artifacts:

```bash
.venv/bin/python -u scripts/serve_sonnet_demo.py --static-only
```

The health endpoint reports static-only status and generation returns an
unavailable response by design.

## Historical training environment

`environments/v7-h100-reference.json` records the authoritative reported V7
environment: one H100 80 GB, PyTorch `2.12.0+cu126`, CUDA 12.6, and NCCL 2.29.3.
It is a provenance record, not a claim that public CPU verification reproduced
the GPU run.

## Evidence and determinism

`scripts/export_portfolio_evidence.py` validates pinned source hashes and emits
the aggregate evidence bundle plus ten Plotly JSON files. Repeated exports are
byte-identical. The public bundle excludes poems, openings, generations,
preferences, votes, annotations, private mappings, tensors, and corpus text.

`scripts/build_public_release_inventory.py` covers every indexed current-tree
path and every unique historical `(path, blob OID)` pair reachable from the
immutable commit recorded in `release/history_review_target.txt`. For a
preparatory commit, that target is its exact parent; the current manifest and
history manifest therefore cover the complete prospective public history
without using a moving remote ref. `--require-cleared` is fail-closed and cannot
pass while any rights, retention, privacy, authority, memo, or review-date field
remains pending.

## V8 corpus correction

The committed `data/processed/sonnets_expanded_v8/` directory is a derived
future-training view. It does not change the frozen V1 corpus, V7 split, trained
models, or published evaluation results. Its builder requires the retained
local BibIt TEI source cache, which is outside a clean public clone:

```bash
.venv/bin/python scripts/build_sonnets_expanded_v8.py
.venv/bin/python -m pytest tests/test_sonnet_v8_correction.py
```

Two complete local builds produced identical SHA-256 file fingerprints for all
11 generated corpus and report files. The committed
[reproducibility record](reports/sonnets_expanded_v8_reproducibility_v1.json)
lists these fingerprints. The builder verifies the pinned TEI source files,
the exact alias-evidence rows,
14-line structure, `4+4+3+3` rendering, `ſ` to `s` replacement, strict
author/work isolation from the fixed V7 held-out sets, and the broader-corpus
overlap quarantine. The committed correction report records the aggregate
results without reproducing held-out poem text.

## Local-only workflows

The canonical full-corpus verifier may run only when the retained local data is
available and cleared for that use. The selected Stage 1, Stage 2, and Stage 3
models and DPO adapter are separately public in the [Hugging Face release](https://huggingface.co/LPM93/teaching-transformers-classical-italian-sonnets),
but they are not embedded in a GitHub clone or the CPU quick-verification path.
Historical-run reconstruction, unselected or intermediate checkpoint
inspection, preference reconstruction, raw-output review, prompt-level
annotation parsing, and review-packet reconstruction remain local-only. Their
absence from the public repositories is intentional.
