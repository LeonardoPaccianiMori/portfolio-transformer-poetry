# Tokenizer Comparison

Dataset: `expanded_with_petrarch`

BPE tokenizer path: `data/metadata/bpe_tokenizer.json`

## Configuration

- Character tokenizer vocabulary size: `94`

- BPE target vocabulary size: `512`

- BPE actual vocabulary size: `512`

- BPE learned merge count: `417`

- BPE special tokens: `<|poem_end|>`

- BPE merge rules were learned from the training split only.

- Base character coverage used train, validation, and test splits to avoid unknown held-out characters.

## Split Token Counts

| Split | Poems | Char Tokens | BPE Tokens | Compression Ratio | Avg Char Tokens/Poem | Avg BPE Tokens/Poem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 736 | 383192 | 157475 | 0.4110 | 520.6 | 214.0 |
| validation | 93 | 48368 | 19878 | 0.4110 | 520.1 | 213.7 |
| test | 92 | 47861 | 19849 | 0.4147 | 520.2 | 215.8 |

## Interpretation

BPE reduces the number of tokens the transformer must process. A lower compression ratio means shorter sequences relative to character tokenization.

Shorter sequences should make full-sonnet context easier for the model, but this report does not prove better generation quality by itself. The next step is to encode BPE train/validation/test tensors and train a comparable BPE transformer.

