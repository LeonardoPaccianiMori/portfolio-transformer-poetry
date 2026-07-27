# Pretraining Mixture Policy

`pretraining_historical_italian_v2` combines the previously used 33-source
expanded Italian corpus with the three audited historical Wikisource works.
The eight medieval Liber Liber sources are already present in the 33-source
component, so they are not added a second time.

Each approved source is concatenated exactly once. The mixture does not use
synthetic oversampling, downsampling, or a hard cap. The builder reports source
and author concentration against a 15-percent work threshold and a 20-percent
author threshold, but these thresholds are warnings rather than automatic text
removal.

`ll_ramusio_navigazioni_viaggi` and author Giovan Battista Ramusio are explicit
approved exceptions. The project decision is to retain the complete work rather
than discard unique primary text merely to satisfy an arbitrary percentage.
Every other future concentration warning requires an explicit composition
decision before it becomes an exception.

The old BPE tokenizer is retained as experiment provenance only. The active
16,000-token BPE tokenizer is trained from a stratified 4M-character sample of
all 36 sources and then counted against the full combined public corpus. Its
JSON artifact and report are committed so its vocabulary reflects the new
Sarpi and Verri text reproducibly.

The active encoded dataset uses that tokenizer without further vocabulary
training. Each source is encoded independently; its final one percent of token
IDs is validation data and the preceding 99 percent is training data. The
builder appends the single-token `<|endoftext|>` separator between consecutive
source segments in both splits. The resulting 17,891,995 training-token and
180,745 validation-token tensors are reproducible local artifacts; their
configuration, source-level counts, and integrity-relevant metadata are public
in [`pretraining_historical_italian_v2_encoded_report.json`](../reports/pretraining_historical_italian_v2_encoded_report.json).
