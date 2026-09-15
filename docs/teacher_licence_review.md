# Teacher Licence Review v1

as_of: 2026-09-15
review_after: 2026-12-15
status: research note, owner decision pending

## Purpose

Record which open-weight models may be used to generate synthetic sonnets that
then train the project's model. The relevant question is not only the weights
licence but whether outputs may be used as training data for another model,
including a model that may be published. This is a research note, not legal
advice, and none of the clauses are court-tested.

## Finding

The cleanest route is to self-host a permissively licensed open-weight model
and generate locally. That avoids the hosted-gateway terms that made the
earlier `opencode-go` pilot unpublishable. Running weights locally is governed
by the weights licence alone.

## Shortlist (capable and clearly permissive)

| Model | Size | Licence | Output-training | Notes |
|---|---|---|---|---|
| Qwen3-32B, Qwen3-14B, Qwen2.5-14B/7B | 7-33B | Apache-2.0 | Yes, no output clause | 100+ languages; Qwen2.5 near the top of the Italian Evalita-LLM leaderboard; 14B fits 48 GB in bf16, 32B needs 4-bit |
| Gemma 4 (up to 31B, April 2026) | edge-31B | Apache-2.0 | Yes | First Gemma under plain Apache-2.0; Italian evidence not yet published, predecessor Gemma 3 led the Italian leaderboard |
| Mistral-Nemo-12B-Instruct | 12B | Apache-2.0 | Yes, no output clause | Self-reported MMLU-Italian 61.3; trivial LoRA on 48 GB |
| DeepSeek-R1-Distill-Qwen-32B | 32B | MIT | Yes, explicit: the model card names distillation for training other LLMs | Reasoning-flavoured and English/Chinese heavy; better as a second-stage teacher |
| Apertus-8B-Instruct | 8B | Apache-2.0 | Yes | Multilingual, EU documented; verse evidence thin |

Excluded on licence: Gemma 3 (custom terms with a synthetic-data derivative
clause), Llama 3.x (name-prefix condition on distilled models), Command-R
(non-commercial), Falcon (custom TII terms), Mistral Large (research licence
bars output use), Llama 2 (explicit ban), GLM-4-9B (registration and naming),
Qwen2.5-72B (custom Qwen agreement with naming conditions).

Excluded on capability: Phi-4 and OLMo 2 (English-centric).

## Route chosen

Generate the teacher corpus by self-hosting Qwen3-32B on the rented 48 GB
instance with 4-bit NF4 quantization, with Qwen3-14B in bf16 as the fallback
if quantization or throughput disappoints. Both are Apache-2.0 with no output
clause. Record the exact model revision, prompts, dates, and counts as
provenance, keep the corpus private, and treat any weight release as a new
owner decision under the publication-risk record.

The earlier `opencode-go` pilot outputs stay local and are not used for
training; they remain a private diagnostic only.

## Material uncertainties

- No clause here is judicially tested, and output copyright status is unsettled
  in the EU.
- Italian capability evidence measures comprehension, not prosody; every
  candidate must be piloted before it is trusted as a verse teacher.
- Model-card licence tags sometimes conflict with licence text; always check
  the exact repository revision used.
