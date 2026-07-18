# Data Attribution

This document covers the sonnet corpus. The repository-wide source index and
licensing policy are in [`DATA_SOURCES_AND_ATTRIBUTION.md`](../DATA_SOURCES_AND_ATTRIBUTION.md).

The processed corpus files in `data/processed/` are derived from Italian Wikisource pages listed in `data/metadata/poems_manifest.csv`.

The builder records source URLs, archive names, edition notes where available, download timestamps, and license notes for each poem.

`sonnets_expanded_v2` is a separate versioned corpus with 966 processed poems:
the 921 active v1 poems plus 45 revision-pinned Vittorio Alfieri sonnets from
*Rime varie* (1912 edition). Its manifest, exact source revisions, and
source-specific attribution are in `data/metadata/sonnets_expanded_v2_manifest.csv`,
`data/metadata/wikisource_snapshots/ws_alfieri_rime_1912.json`, and
`data/metadata/sonnets_expanded_v2_attribution.md`.

Italian Wikisource pages used in this project generally expose Creative Commons Attribution-ShareAlike and/or GFDL metadata. The exact source page URL and license notes should be preserved in the manifest for every processed poem.

Relevant source-policy pages:

- https://wikisource.org/wiki/Wikisource:Copyright_policy
- https://en.wikisource.org/wiki/Wikisource:Reusing_Wikisource_content

The repository should commit:

- corpus-building code;
- processed poem text;
- metadata and attribution files;
- reports and evaluation artifacts.

The repository should not commit:

- temporary raw HTML downloads;
- interim extraction files;
- virtual environments;
- caches;
- model checkpoints or large model weights.
