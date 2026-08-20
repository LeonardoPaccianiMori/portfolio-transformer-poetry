# Hugging Face Release Preparation

## Status

Four Hugging Face artifacts are being prepared privately. This document and
the files under `release/huggingface/` do not authorize repository creation,
upload, or public distribution.

| Artifact | Planned repository | Form |
| --- | --- | --- |
| Stage 1 | `LPM93/minerva-7b-classical-italian-stage1` | Full BF16 model |
| Stage 2 | `LPM93/minerva-7b-classical-italian-poetry-stage2` | Full BF16 model |
| Stage 3 | `LPM93/minerva-7b-classical-italian-sonnets-stage3` | Full BF16 model, selected update 120 |
| DPO | `LPM93/minerva-7b-classical-italian-sonnets-dpo-adapter` | PEFT LoRA adapter for the exact Stage-3 repository |

The eventual collection title is **Teaching Transformers to Write Classical
Italian Sonnets**. It will be created only after all four repositories have
been separately verified in their intended public, ungated state.

## Layered licensing

Hugging Face metadata will use `license: cc-by-nc-4.0` for Leonardo-controlled
rights in the model modifications. The tag is not a repository-wide legal
conclusion. Every package contains a prominent `RIGHTS_SCOPE.md` explaining
that:

- Apache-2.0 continues to govern rights independently received in the pinned
  Minerva parent;
- CC BY-NC 4.0 applies only to copyright and similar rights Leonardo holds, if
  any, in his original modifications embodied in the identified weight or
  adapter files;
- existing project prose reused from the source repository retains its CC BY
  4.0 status;
- source disclosure grants no rights in corpora, generated outputs, or other
  third-party material.

Research transparency, inspection, reproducibility, and verification are the
intended purposes. CC BY-NC 4.0 also permits other uses that satisfy its own
NonCommercial definition. The project adds no research-only, production,
field-of-use, responsible-use, gating, or identical-terms restriction.

This structure is usable only under the unresolved assumption that
training-data licenses do not govern the weights. If incompatible ShareAlike
terms are determined to attach, distribution is not authorized under this
structure. Preparing files does not decide that question.

## Exact lineage

`scripts/reconcile_huggingface_lineage.py` reads the private deterministic
window indexes and emits only aggregate target-token evidence. It includes:

- all 33,040 windows consumed by selected Stage 1;
- all 12,160 windows consumed by selected Stage 2;
- the first 1,920 Stage-3 windows consumed through selected update 120, not the
  unselected terminal update 135;
- cumulative Stage-1-to-Stage-3 exposure for the DPO adapter, plus aggregate
  preference-training counts.

The output contains no poems, document identifiers, openings, preferences,
votes, annotations, generations, or token IDs.

## Local export

The exporter requires explicit local inputs and an absent or empty output
directory:

```bash
python3 scripts/reconcile_huggingface_lineage.py
python3 scripts/export_huggingface_artifacts.py \
  --stage-boundaries <selected-stage-boundaries> \
  --adapter-checkpoint <selected-best-adapter.pt> \
  --output-root <ignored-empty-output-directory> \
  --include-verified-upstream-files
python3 scripts/validate_huggingface_artifacts.py \
  --package-root <ignored-output-directory> \
  --adapter-checkpoint <selected-best-adapter.pt> \
  --certify-release-candidate
```

The three full-model exports are byte-identical copies of the selected
safetensors packages. The DPO exporter writes only the 448 LoRA tensors and a
standard PEFT configuration. Optimizer state, RNG state, histories, pair IDs,
preferences, and all other checkpoint fields are excluded.

Certification checks the complete package manifest and allowlists, exact
adapter tensors, safetensors metadata, clean local model loads, adapter
attachment, and fixed text-free token/logit equivalence against a PEFT model
reconstructed directly from the selected research checkpoint.

## Later publication gate

Before upload, a separate owner decision must name the exact package hashes
and accept the layered scope, cumulative notices, worldwide ungated
distribution, and the irrevocable CC BY-NC 4.0 grant for rights Leonardo can
license. It must also refresh the legal and EU AI Act assessment. Creating
private Hugging Face repositories, uploading, changing visibility, creating a
collection, and adding public links are outside this preparation checkpoint.
