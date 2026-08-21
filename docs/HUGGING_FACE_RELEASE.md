# Hugging Face Release Preparation

## Status

One Hugging Face repository containing four model artifacts is authorized for
private staging and, after the complete validation gate passes, public ungated
distribution. At this tracked checkpoint it has not yet been uploaded.

Planned repository:
`LPM93/teaching-transformers-classical-italian-sonnets`.

| Artifact | Subfolder | Form |
| --- | --- | --- |
| Stage 1 | `stage1` | Full BF16 model |
| Stage 2 | `stage2` | Full BF16 model |
| Stage 3 | `stage3` | Full BF16 model, selected update 120 |
| DPO | `dpo_adapter` | PEFT LoRA adapter for the exact Stage-3 subfolder |

The repository display title is **Teaching Transformers to Write Classical
Italian Sonnets**. A separate Hugging Face collection is unnecessary.

## Layered licensing

Hugging Face metadata will use `license: cc-by-nc-4.0` for Leonardo-controlled
rights in the model modifications. The tag is not a repository-wide legal
conclusion. Every artifact subfolder contains a prominent `RIGHTS_SCOPE.md`
explaining
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

The repository root contains the public model card and layered license texts.
The three full-model exports are byte-identical copies of the selected
safetensors packages. The DPO exporter writes only the 448 LoRA tensors and a
standard PEFT configuration. Optimizer state, RNG state, histories, pair IDs,
preferences, and all other checkpoint fields are excluded.

Certification checks the complete package manifest and allowlists, exact
adapter tensors, safetensors metadata, clean local model loads, adapter
attachment, and fixed text-free token/logit equivalence against a PEFT model
reconstructed directly from the selected research checkpoint.

## Publication procedure

Leonardo approved the single-repository topology, layered scope, worldwide
ungated distribution, and the irrevocable CC BY-NC 4.0 grant for rights he can
license. Publication remains fail-closed: create the repository privately,
upload only the allowlisted package, verify exact remote files and hashes,
re-download into a clean cache, load all three full-model subfolders, attach the
adapter from `dpo_adapter` to `stage3`, and only then change the one repository
to public and ungated. Any failed transition or public verification requires an
immediate return to private visibility.

PEFT's `adapter_config.json` has no standard base-subfolder field. It records
the single repository ID, while the model card and validation require callers
to load the base with `subfolder="stage3"` before loading the adapter with
`subfolder="dpo_adapter"`.
