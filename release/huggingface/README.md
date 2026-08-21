---
license: cc-by-nc-4.0
language:
  - it
library_name: transformers
pipeline_tag: text-generation
base_model: sapienzanlp/Minerva-7B-instruct-v1.0
tags:
  - minerva
  - historical-italian
  - italian-poetry
  - italian-sonnets
  - peft
  - research
---

# Teaching Transformers to Write Classical Italian Sonnets

This release package contains four selected artifacts intended for publication
from a controlled research study of staged literary adaptation of the existing
Minerva 7B Instruct model:

| Subfolder | Artifact | Selected endpoint |
| --- | --- | --- |
| `stage1` | Full BF16 model after historical/general Italian adaptation | Update 2,065 |
| `stage2` | Full BF16 model after non-sonnet poetry adaptation | Update 760 |
| `stage3` | Full BF16 model after V7 sonnet adaptation | Update 120 of 135 planned |
| `dpo_adapter` | Rank-8 PEFT LoRA adapter for the exact selected Stage-3 model | Update 61 |

The three stages form one sequential lineage. They ultimately begin from
`sapienzanlp/Minerva-7B-instruct-v1.0` at revision
`d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d`. The models were **not trained
from scratch**. The DPO adapter must be attached to the `stage3` subfolder in
this repository; it is not compatible with a generic Minerva checkpoint.

## Loading

Load each full model by naming its subfolder:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "LPM93/teaching-transformers-classical-italian-sonnets"

tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="stage3")
stage3 = AutoModelForCausalLM.from_pretrained(
    repo_id,
    subfolder="stage3",
    torch_dtype="auto",
    device_map="auto",
)
```

Load the adapter only after explicitly loading its exact Stage-3 base:

```python
from peft import PeftModel

dpo_model = PeftModel.from_pretrained(
    stage3,
    repo_id,
    subfolder="dpo_adapter",
)
```

Use `subfolder="stage1"` or `subfolder="stage2"` for the earlier full-model
states. PEFT's `adapter_config.json` records the repository ID but has no
standard field for a base-model subfolder, so automatic adapter-only loading is
not supported: load `stage3` explicitly as shown above.

## Result boundary

The final sealed test supports a narrow automatic surface/completion gain from
AI-judged DPO, not broad literary quality. Human/AI calibration was only
**12/20** and failed its gate. Both Stage 3 and DPO produced **0/100**
strict-good outputs in the frozen blind review. Fourteen-line output was
decoder-controlled. These artifacts are research evidence, not reliable poets,
human-aligned systems, or production language models.

## Rights and training-data boundary

Hugging Face metadata uses CC BY-NC 4.0 only for copyright and similar rights
Leonardo Pacciani-Mori holds, if any, in his original model modifications.
Rights independently received in the pinned Minerva parent retain the parent's
Apache-2.0 designation. Leonardo-owned model-card prose and aggregate
documentation retain CC BY 4.0 where identified.

No corpus text, token streams, prompts, openings, generations, preferences,
votes, annotations, mappings, optimizer state, or training logs are included.
Source disclosure does not grant corpus-redistribution rights or create one
package-wide data license. Read each subfolder's `RIGHTS_SCOPE.md`, `NOTICE.md`,
`TRAINING_CONTENT_SUMMARY.md`, and `lineage.json` before reuse.

The release uses the unresolved assumption that training-data licenses do not
govern the weights. If incompatible ShareAlike terms are determined to attach,
distribution is not authorized under this structure. The repository is
provided without a warranty of title or non-infringement, subject to the
included license texts.

## Ownership and AI contribution

Leonardo Pacciani-Mori conceived and directed the project, made executive
decisions, approved the research plan, reviewed outputs, and sometimes ran GPU
work. Codex 5.5 and later Codex 5.6 Sol substantially assisted research design,
implementation, tests, execution, and analysis. The study was not independently
designed or independently implemented by Leonardo.

Project source, evidence, and full limitations:
<https://github.com/LeonardoPaccianiMori/portfolio-transformer-poetry>
