# PAISÀ-To-Historical Rescue Curriculum

## Decision

`paisa_historical_rescue_v1` is the single permitted final from-scratch rescue
curriculum. It does not create a permanently mixed corpus. Its stages are
ordered so the model first learns broad modern Italian, then adapts to the
historical prose closest to the sonnet target, and only then enters the existing
V5 sonnet and task-format stages.

| Stage | Data | Maximum passes | Schedule role |
| --- | --- | ---: | --- |
| Modern Italian pretraining | PAISÀ train | 3 | Warmup, stable learning rate, then decay |
| Historical Italian annealing | Historical-prose train | 12 | Lower-rate continuation and decay |
| Sonnet fine-tuning | V5 sonnet train | Selected from the existing stable protocol | Target-form specialization |
| Task-format post-training | V5 task-format train | Selected from the existing protocol | Opening-line continuation behavior |

Three PAISÀ passes are below the hard four-pass broad-corpus limit. Twelve
historical passes are intentional because historical prose is much smaller and
must materially alter the modern stage before sonnet specialization. This is a
fixed curriculum, not a new corpus-mixture or optimizer sweep.

## Provenance And Splits

The curriculum locks the PAISÀ release checksum and the SHA-256 hashes of both
input build reports. Preparation fails if either report drifts.

- PAISÀ: 371,612 train documents and 3,776 document-level validation documents.
  Its exact duplicates were removed before the deterministic fingerprint split.
- Historical prose: 36 sources, split separately at a newline near each source's
  final 1%. This yields 47,559,476 training and 445,461 validation characters.

The historical suffix split excludes validation text from tokenizer fitting and
training, but it is weaker than PAISÀ's document-level isolation because the
same work can continue across the boundary. Reports must retain this limitation.

## Fresh Tokenizer Policy

The rescue uses a fresh 16,000-token BPE vocabulary with only
`<|endoftext|>` reserved as a special token. Its deterministic 12M-character
training sample contains:

- 8,016,457 characters from 2,113 PAISÀ train documents, selected by a stable
  content-hash rule; and
- 4,000,540 characters stratified across all 36 historical training sources.

No PAISÀ or historical validation text is included in the sample. The
historical component receives one third of tokenizer-fitting characters despite
being much smaller, so older spellings and punctuation remain well represented
in the vocabulary.

The machine-readable policy is
[`configs/paisa_historical_rescue_v1.json`](../configs/paisa_historical_rescue_v1.json).
The aggregate local preparation evidence is
[`reports/paisa_historical_rescue_v1_curriculum_report.json`](../reports/paisa_historical_rescue_v1_curriculum_report.json).

## Completed Tokenizer Gate

The train-only Unicode BPE fit completed with exactly 16,000 tokens: 7,217
base tokens and 8,783 learned merges. The serialized tokenizer sample contains
12,049,232 characters and 4,812,076 BPE tokens, or 2.5040 characters per
token. Its difference from the 12,016,997 selected text characters is the
required newline and `<|endoftext|>` document separators.

The PAISÀ validation split contained nine characters that were absent from both
training splits: one control character and eight rare CJK characters. The
original validation file remains unchanged. The tokenizer workflow derives a
separate local validation file which excludes only the six affected documents
(32,468 characters), leaving 3,770 of 3,776 documents. This prevents
validation-derived vocabulary entries and records the exclusion count and
codepoints without publishing PAISÀ text or document URLs.

The completed aggregate evidence is in
[`reports/paisa_historical_rescue_v1_tokenizer_report.json`](../reports/paisa_historical_rescue_v1_tokenizer_report.json).

## Next Scheduled Checkpoint

Implement a memory-safe streaming encoder for the four already fixed splits:
PAISÀ train, tokenizable PAISÀ validation, historical train, and historical
validation. It must not create another suffix or random split. Its measured
token counts determine the final rescue model size within the already approved
50–70M-parameter range, its update budgets, and the GPU benchmark configuration.
