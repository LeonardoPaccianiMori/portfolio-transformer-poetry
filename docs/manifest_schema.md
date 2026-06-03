# Corpus Manifest Schema

The corpus manifest is a machine-readable table with one row per candidate poem.

Its job is to connect the source audit to the training data pipeline. The manifest should make it clear where every poem came from, why it is considered a sonnet, how it was cleaned, and which dataset versions may use it.

Recommended file format:

```text
data/metadata/poems_manifest.csv
```

CSV is a good first format because it is easy to inspect manually. JSONL can be added if nested metadata becomes necessary. Processed poem text and metadata are committed to the repo. Raw and interim extraction files are temporary builder artifacts and are deleted after a successful build.

## Core Rule

One manifest row equals one candidate poem.

Do not create one row per line, source page, stanza, or training chunk. Training chunks will be created later by tokenization and batching.

## Field Definitions

| Field | Type | Required | Description |
|---|---|---:|---|
| `poem_id` | string | yes | Stable internal ID, such as `giacomo_sonnet_020` or `petrarca_rvf_001`. |
| `title_or_first_line` | string | yes | Human-readable title or first line. |
| `author` | string | yes | Normalized author used for grouping and analysis. |
| `displayed_author` | string | yes | Author shown on the source page. May differ from `author` for correspondence poems. |
| `source_archive` | string | yes | Archive name, for example `Italian Wikisource`. |
| `source_collection` | string | yes | Main collection, for example `Poesie (Giacomo da Lentini)`. |
| `source_subcollection` | string | no | Section or cycle name, for example `Sonetti d'amore` or `Sonetti dei mesi`. |
| `source_url` | string | yes | Exact page URL used for extraction. |
| `source_revision_id` | string | no | Source revision ID if available from Wikisource/API metadata. |
| `source_revision_timestamp` | string | no | Source revision timestamp if available from Wikisource/API metadata. |
| `downloaded_at_utc` | string | yes | UTC timestamp when the source page was fetched. |
| `source_edition` | string | no | Edition or scan information, if known. |
| `license_notes` | string | yes | Reuse/license notes copied or summarized from the source audit. |
| `period` | string | yes | Approximate period, for example `XIII secolo` or `XIV secolo`. |
| `form` | string | yes | Expected to be `sonnet` for included poems. |
| `form_evidence` | string | yes | Why this poem is treated as a sonnet: explicit index section, Wikisource category, line count, canonical count, etc. |
| `count_method` | string | yes | Method used to identify/count the poem. See allowed values below. |
| `attribution_status` | string | yes | `secure`, `doubtful`, `correspondence`, `attributed`, or `unknown`. |
| `line_count_raw` | integer | no | Number of poem lines before cleaning, if measurable. |
| `line_count_clean` | integer | no | Number of poem lines after cleaning. Sonnets should normally be 14. |
| `raw_text_path` | string | no | Path to raw downloaded text, if stored locally. |
| `clean_text_path` | string | no | Path to cleaned poem text used for modeling. |
| `include_in_core_pre_petrarch` | boolean | yes | Whether this poem belongs to the core pre-Petrarch dataset version. |
| `include_in_expanded_with_petrarch` | boolean | yes | Whether this poem belongs to the expanded dataset version including Petrarca. |
| `include_in_training` | boolean | yes | Final inclusion flag for the active dataset build. |
| `split_core_pre_petrarch` | string | no | `train`, `validation`, `test`, or `excluded` for the core dataset. |
| `split_expanded_with_petrarch` | string | no | `train`, `validation`, `test`, or `excluded` for the expanded dataset. |
| `editorial_brackets_removed` | boolean | yes | Whether processed text removes editorial square brackets such as `ben[e]` -> `bene`. |
| `line_markers_removed` | boolean | yes | Whether displayed line-number markers were removed from processed text. |
| `cleaning_notes` | string | no | Human-readable cleaning notes. |
| `audit_notes` | string | no | Source, attribution, form, or extraction uncertainty notes. |

## Allowed Values

### `count_method`

Use one of:

- `explicit_index_section`
- `wikisource_category`
- `line_count_14`
- `canonical_external_count`
- `manual_exclusion`

If multiple methods apply, use the strongest primary method in `count_method` and explain the secondary evidence in `form_evidence` or `audit_notes`.

### `attribution_status`

Use one of:

- `secure`
- `doubtful`
- `correspondence`
- `attributed`
- `unknown`

### `split`

Use one of:

- `train`
- `validation`
- `test`
- `excluded`

Leave empty only before split assignment.

## Dataset Versions

The project should maintain at least two sonnet corpus versions.

### `core_pre_petrarch`

Purpose: train and evaluate a model on earlier sonnet sources before Petrarca dominates the distribution.

Included by default:

- Giacomo da Lentini
- Dante Alighieri
- Guido Cavalcanti
- Cino da Pistoia
- Cecco Angiolieri
- Folgore da San Gimignano
- Guittone d'Arezzo

Excluded by default:

- Francesco Petrarca

### `expanded_with_petrarch`

Purpose: train and evaluate a larger corpus that includes Petrarca and tests whether the larger dataset improves form quality while changing style distribution.

Included by default:

- all approved `core_pre_petrarch` poems
- approved Petrarca sonnets from the *Canzoniere*

## Split Policy

Splits must be assigned by poem.

Do not split one sonnet across train, validation, and test. That would leak phrasing, rhyme endings, and local structure into evaluation.

Recommended first split:

| Split | Share | Purpose |
|---|---:|---|
| `train` | 80% | Model fitting. |
| `validation` | 10% | Hyperparameter and checkpoint selection. |
| `test` | 10% | Final held-out evaluation. |

For small author subsets, preserve author/source balance as much as possible. For example, do not put all Folgore sonnets only in validation or only in test.

## Cleaning Policy

Raw text and processed text serve different purposes.

Raw text:

- should preserve the source as downloaded;
- should keep editorial brackets, line markers, and source-specific artifacts;
- is used for auditability during a build;
- is deleted after a successful build unless a debugging flag preserves temporary artifacts.

Processed text:

- is what the tokenizer and model see;
- should remove page chrome, navigation, headers, footers, and license boilerplate;
- should remove displayed line-number markers;
- should preserve poetic line breaks;
- should preserve spelling and punctuation except for explicitly approved cleaning rules;
- should remove square brackets around editorial letter expansions, such as `ben[e]` -> `bene`.

Every cleaning rule that changes poem text should be recorded in manifest fields.

## PyTorch Connection

The manifest is not fed directly into the transformer. It is the metadata layer that lets us build a reliable dataset.

The later pipeline will look like:

```text
manifest row
  -> clean_text_path
  -> poem text
  -> tokenizer.encode(text)
  -> token IDs
  -> training chunks shaped like (batch, time)
```

For a causal language model, each training chunk will be split into:

```text
x = token_ids[0:time]
y = token_ids[1:time+1]
```

The model sees `x` and learns to predict `y`, one next token at a time.

## Example Row

Example shown as key-value pairs for readability:

```text
poem_id: giacomo_sonnet_018a
title_or_first_line: Oi deo d'amore
author: Abate di Tivoli
displayed_author: Abate di Tivoli
source_archive: Italian Wikisource
source_collection: Poesie (Giacomo da Lentini)
source_subcollection: Sonetti
source_url: https://it.wikisource.org/wiki/Poesie_(Giacomo_da_Lentini)/Sonetti/Oi_deo_d%27amore
source_revision_id:
source_revision_timestamp:
downloaded_at_utc:
source_edition: Roberto Antonelli, Bulzoni Editore, Roma, 1979
license_notes: CC BY-SA / GFDL metadata on Italian Wikisource
period: XIII secolo
form: sonnet
form_evidence: explicit Sonetti section; cleaned line count should be 14
count_method: explicit_index_section
attribution_status: correspondence
line_count_raw:
line_count_clean:
raw_text_path:
clean_text_path:
include_in_core_pre_petrarch: true
include_in_expanded_with_petrarch: true
include_in_training: true
split_core_pre_petrarch:
split_expanded_with_petrarch:
editorial_brackets_removed: true
line_markers_removed: true
cleaning_notes: Remove editorial square brackets around letter expansions; remove displayed line markers.
audit_notes: Displayed author is Abate di Tivoli, not Giacomo da Lentini.
```

## Acceptance Criteria

The manifest schema is ready for implementation when:

- every source-audit decision can be represented by one or more fields;
- doubtful attribution and correspondence poems can be represented without losing information;
- both planned dataset versions can be selected from boolean fields;
- raw and clean text paths are separate;
- poem-level splitting is supported;
- the schema explains how records become PyTorch training examples.
