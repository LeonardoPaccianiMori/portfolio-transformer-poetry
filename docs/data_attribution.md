# Data Attribution

The processed corpus files in `data/processed/` are derived from Italian Wikisource pages listed in `data/metadata/poems_manifest.csv`.

The builder records source URLs, archive names, edition notes where available, download timestamps, and license notes for each poem.

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
