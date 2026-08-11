# Italian Wikisource Resolved Corpus v1: Attribution And Reuse

## Source And License Layers

- Transcription source: [Italian Wikisource](https://it.wikisource.org/).
- Pinned source snapshot: Italian Wikisource `20260801` current-revision XML
  dump, SHA-1 `cacf8406058d3cadcf520a399962e9029352bddb`.
- Wikisource transcription terms: [Creative Commons Attribution-ShareAlike
  4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.it).
- Source-scan evidence: the exact Wikimedia Commons description URL, license
  metadata, credit, creator/editor/translator/illustrator fields, and source
  edition are recorded per scan in
  `italian_wikisource_source_rights_v1.csv`.

The two rights layers are evaluated separately. Wikisource hosting or an old
author does not establish that a source scan or modern editorial contribution
is reusable. The build therefore fails closed unless the exact scan is marked
public domain/non-copyrighted and the edition evidence is compatible.

## Audited Scope And Decision

Checkpoint 4D accounts for all 4,641 extracted work roots and all 2,095 prior
review rows. It audits exactly 1,282 distinct source scans:

- 1,175 scans pass the recorded rights gate;
- 47 are held for incomplete Index provenance;
- 30 are held for incompatible or unclear Commons scan status;
- 30 are held because a post-1930 edition names a modern editor, translator,
  or illustrator.

After rights, extraction-quality, canonicalization, protected-V6, and form
gates, the inactive processed build contains 2,452 broader records and 987
standard-sonnet candidates. Another 222 roots contribute eligible sonnets but
no broader record. The build has 2,674 root-level attribution rows because
every root contributing either artifact retains its own source and history
link. All conditioned-language and checkpoint-4B-held roots remain outside the
standard-Italian build.

## Required Attribution And Notices

For each redistributed record or derived dataset:

1. credit Italian Wikisource and link the stable work page and its contributor
   history URL from `attribution_manifest.csv`;
2. retain the CC BY-SA 4.0 name and license link;
3. retain the exact source-scan credit, Commons description URL, and any named
   edition contributors from the same attribution row;
4. state the modifications described below; and
5. preserve applicable ShareAlike obligations for redistributed adaptations.

The machine-readable `required_notice`, `modification_notice`, and
`downstream_note` columns are authoritative for individual records. Public
domain scan labels do not remove the separate Wikisource transcription
attribution and ShareAlike requirements.

## Transformations

The builder removes MediaWiki/ProofreadPage markup, identified navigation and
editorial apparatus, canonical duplicate spans, and protected held-out sonnet
spans. It preserves retained primary-text spelling and punctuation. Verified
sonnets are removed from broader-stage text and materialized separately.
Deterministic newline separators preserve retained-segment and record
boundaries; consequently, the 30,751,389 materialized broader characters are
2,061 characters greater than the 30,749,328 retained source-span characters.

## Canonical Evidence

- `data/metadata/italian_wikisource_source_rights_v1.csv`: per-scan rights and
  edition evidence, including all failed rows.
- `data/metadata/italian_wikisource_root_decisions_v1.csv`: final root decision
  and source lineage.
- `data/metadata/italian_wikisource_segment_decisions_v1.csv`: exact retained
  and excluded source spans.
- `data/metadata/italian_wikisource_sonnet_decisions_v1.csv`: poem-level form,
  duplicate, and protected-split decisions.
- `data/processed/italian_wikisource_resolved_v1/attribution_manifest.csv`:
  root-level attribution for every materialized broader or sonnet artifact.
- `data/processed/italian_wikisource_resolved_v1/build_report.json`: shard and
  manifest hashes plus final verification counts.

These artifacts are materialized but inactive. This checkpoint creates no V7
split or mixture weight and authorizes no model training.
