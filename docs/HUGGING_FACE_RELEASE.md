# Hugging Face Release Record

## Status

One Hugging Face repository containing four model artifacts is public and
ungated after passing the complete private validation gate.

Published repository:
[`LPM93/teaching-transformers-classical-italian-sonnets`](https://huggingface.co/LPM93/teaching-transformers-classical-italian-sonnets).

The public Hugging Face commit is
`3581abbb1023c77f784b37aa152cdb6c0447fa73`. The 69-file package is exactly
44,492,691,499 bytes and its `package_manifest.json` has SHA-256
`cedc265ece2b0faf81cf03861ac2f64faa8dea90856dd919a5990e238f2746e2`.

| Artifact | Subfolder | Form |
| --- | --- | --- |
| Stage 1 | `stage1` | Full BF16 model |
| Stage 2 | `stage2` | Full BF16 model |
| Stage 3 | `stage3` | Full BF16 model, selected update 120 |
| DPO | `dpo_adapter` | PEFT LoRA adapter for the exact Stage-3 subfolder |

The repository display title is **Teaching Transformers to Write Classical
Italian Sonnets**. A separate Hugging Face collection is unnecessary.

## Layered licensing

Hugging Face metadata uses `license: cc-by-nc-4.0` for Leonardo-controlled
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

This release structure uses the unresolved assumption that
training-data licenses do not govern the weights. If incompatible ShareAlike
terms are determined to attach, distribution is not authorized under this
structure. Publishing the files does not decide that question.

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

## Completed publication procedure

Leonardo approved the single-repository topology, layered scope, worldwide
ungated distribution, and the irrevocable CC BY-NC 4.0 grant for rights he can
license. The repository was created privately, received only the allowlisted
package, and remained private while its exact remote paths and hashes were
checked. A clean re-download then passed package, safetensors, three-model-load,
adapter-attachment, selected-checkpoint, and fixed text-free logit-equivalence
checks. Only after those checks passed was the repository made public and
ungated. Anonymous API and file-resolution checks then confirmed the public
state, exact commit, model card, package manifest, Stage-3 configuration, and
DPO adapter configuration.

PEFT's `adapter_config.json` has no standard base-subfolder field. It records
the single repository ID, while the model card and validation require callers
to load the base with `subfolder="stage3"` before loading the adapter with
`subfolder="dpo_adapter"`.
