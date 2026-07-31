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

## Completed Encoding Gate

The memory-safe encoder writes each of the four already fixed splits as a
little-endian `uint16` binary token stream. It does not create another random
or suffix split. The format uses two bytes per token, can be opened with
`torch.from_file` without loading the complete corpus into RAM, and allows the
trainer to convert only sampled training windows to `torch.long`.

The completed token budget is:

| Split | Documents | Characters | BPE tokens |
| --- | ---: | ---: | ---: |
| PAISÀ train | 371,612 | 1,407,676,908 | 563,880,445 |
| PAISÀ validation | 3,770 | 14,427,213 | 5,777,210 |
| Historical train | 36 | 47,559,476 | 19,187,647 |
| Historical validation | 36 | 445,461 | 177,094 |

The four streams contain 589,022,396 tokens and occupy 1,178,044,792 bytes
(1.10 GiB). The same IDs stored as `torch.long` would require approximately
4.39 GiB before serialization overhead. The encoder checkpoints complete
document boundaries, truncates any uncheckpointed output before resuming, and
publishes final files only after source consumption and document counts pass.

Implementation:

- `src/sonnet_corpus/paisa_historical_encoding.py`
- `scripts/encode_paisa_historical_rescue.py`
- `tests/test_paisa_historical_encoding.py`
- `reports/paisa_historical_rescue_v1_encoded_report.json`

## Completed Mapped-Training Gate

The shared batching and validation code now accepts either legacy
`torch.long` tensors or the rescue corpus's mapped `torch.uint16` streams.
For the latter, it maps the files with `torch.from_file`, checks their recorded
paths, split IDs, counts, tokenizer vocabulary, separator ID, and complete
token-ID range, then converts only each sampled input/target window to
`torch.long` for embedding lookup. This keeps the 1.10 GiB encoded corpus out
of the training process's regular tensor allocation.

The fixed rescue architecture is 70,055,900 parameters: a 16,000-token
vocabulary, width 640, 10 layers, 10 attention heads of dimension 64, SwiGLU
dimension 1,707, LayerNorm, learned absolute positions, untied embeddings, and
a 512-token context. The only remaining hardware calibration is the practical
microbatch/gradient-accumulation pairing, while preserving 4,096 target tokens
per optimizer update:

| Candidate | Microbatch | Accumulation | Tokens per update |
| --- | ---: | ---: | ---: |
| `rescue_upper_micro1` | 1 | 8 | 4,096 |
| `rescue_upper_micro2` | 2 | 4 | 4,096 |
| `rescue_upper_micro4` | 4 | 2 | 4,096 |

Implementation:

- `src/sonnet_corpus/batching.py`
- `src/sonnet_training/pretraining_run.py`
- `src/sonnet_training/pretraining_benchmark.py`
- `scripts/benchmark_paisa_historical_rescue.py`
- `tests/test_batching.py`
- `tests/test_training_steps.py`
- `tests/test_pretraining_run.py`
- `tests/test_pretraining_benchmark.py`

## Next Scheduled Checkpoint

Run the GPU microbatch calibration using
`scripts/benchmark_paisa_historical_rescue.py`. It measures the three fixed
microbatch options against the actual PAISÀ streams and writes a local JSON
report plus the public Markdown report. Select the largest option that fits
reliably, then calculate the fixed stage budgets for no more than three PAISÀ
passes and twelve historical passes before the one permitted rescue run begins.
