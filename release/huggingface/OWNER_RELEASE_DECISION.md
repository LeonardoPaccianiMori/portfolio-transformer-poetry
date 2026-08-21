# Owner Release Decision — Single Hugging Face Repository

- **Decision record:** `decision-2026-08-21-huggingface-single-repository-release`
- **Decision date:** 2026-08-21
- **Decision authority role:** repository owner and model publisher
**Status:** authorized for private staging and public ungated distribution after validation

## Authorized repository and package

Leonardo Pacciani-Mori authorized one Hugging Face model repository:

`LPM93/teaching-transformers-classical-italian-sonnets`

The authorized local release candidate contains 69 files in the repository
root and the four subfolders `stage1`, `stage2`, `stage3`, and `dpo_adapter`.
Its deterministic `package_manifest.json` has SHA-256:

`cedc265ece2b0faf81cf03861ac2f64faa8dea90856dd919a5990e238f2746e2`

The selected research identities are:

- Stage 1:
  `c3aba5b672e8634028477885ad96d7c25c48d2c60cc5597b6da730089212ac39`;
- Stage 2:
  `75817039c2392daac314d9f3365b4c0e1a7b6a5bdab33cf5f95d39ec1ee8397d`;
- Stage 3 selected update 120:
  `478d5979e25a78375d7af0434db6a5432678762fac2d142af2d4798dda53a474`;
- selected DPO research checkpoint:
  `72aa174b2ef87e021a367b0f7e786fce8c3437bb5ca1f8c7f9c5b13588620822`;
- exported DPO adapter:
  `7bed600bd710eee3c9983ab443de1258194033c105b24609e8cf54a17fcc658a`.

The package manifest records the exact SHA-256 and byte size of every public
file. No other local checkpoint or package is authorized by this record.

## Scope accepted by the owner

The owner authorizes worldwide public, ungated distribution through the named
Hugging Face repository after private remote validation. He accepts that:

- CC BY-NC 4.0 applies only to copyright and similar rights he holds, if any,
  in his original modifications embodied in the identified model and adapter
  files;
- independently received rights in the pinned Minerva parent retain the
  parent's Apache-2.0 designation;
- identified Leonardo-owned prose and aggregate documentation retain CC BY
  4.0;
- source disclosure grants no rights in corpus text, metadata beyond permitted
  reuse, prompts, openings, generations, preferences, votes, annotations, or
  third-party material;
- research, transparency, reproducibility, and verification are intended
  purposes, not restrictions beyond CC BY-NC 4.0;
- the applicable CC grant is irrevocable subject to its terms;
- public visibility cannot retract prior downloads, caches, or mirrors.

## Unresolved legal assumption

This is an owner publication decision, not legal advice or a specialist legal
conclusion. It deliberately accepts the documented unresolved risk that a
training-data ShareAlike or other source term might be argued to govern model
weights. The release structure assumes that training-data licenses do not
govern the weights. If incompatible terms are later determined to attach,
distribution is not authorized under this structure and remediation must be
assessed separately.

## Fail-closed publication procedure

The repository must be created privately. Only files covered by the package
manifest may be uploaded. Remote paths, hashes, safetensors serialization,
clean-cache downloads, all three `subfolder=` model loads, and exact
Stage-3-plus-DPO attachment must pass before public visibility. If upload,
visibility transition, or public verification fails, the repository must be
returned to private and its private state verified.
