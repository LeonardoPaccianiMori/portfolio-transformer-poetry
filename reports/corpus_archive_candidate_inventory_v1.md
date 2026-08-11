# Corpus Archive Candidate Inventory V1

Checkpoint 6B completed a metadata/source inventory only. It did not acquire corpus text, activate records, create V7 splits, assign mixture weights, delete caches, or start GPU work.

The normalized public ledger contains **114,971 rows**. Exactly **4,634** rows are inactive candidates for a later bounded audit; **4,393** language-variety rows remain outside the standard-Italian queue.

## Archive accounting

| Archive | Raw | Published rows | Filtered/accounted | Candidates | Holds | Excluded | Conditioned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eltec_italian | 70 | 70 | 0 | 42 | 10 | 18 | 0 |
| internet_archive | 99,424 | 99,424 | 0 | 1,985 | 93,906 | 0 | 3,533 |
| gallica | 13,919 | 12,100 | 1,819 | 172 | 11,454 | 0 | 474 |
| internet_culturale | 291 | 291 | 0 | 136 | 134 | 0 | 21 |
| beic | 9,420 | 2,285 | 7,135 | 1,670 | 250 | 0 | 365 |
| midia | 801 | 801 | 0 | 629 | 171 | 1 | 0 |

## Planning projections and concentration

Inactive candidates carry **21,843,444 projected characters** and **8,672,574 projected word/occurrence units** where source metadata permits estimation. The character projection comes only from ELTeC's word counts using a documented six-characters-per-word planning multiplier; MIDIA contributes occurrence units rather than tokenizer tokens. Archives without size metadata remain unprojected.

Top contributor/institution proxies by normalized record count are:

- `eltec_italian`: De Marchi, Emilio — 4/70 records (5.71%).
- `internet_archive`: Il Maldicente — 812/99,424 records (0.82%).
- `gallica`: [s.n.] — 646/12,100 records (5.34%).
- `internet_culturale`: Biblioteca nazionale centrale - Firenze — 22/291 records (7.56%).
- `beic`: Accademia delle scienze di Torinoaut; Accademia delle scienze di Torinopbl — 144/2,285 records (6.30%).
- `midia`: Anonimo — 140/801 records (17.48%).

## Important boundaries

- Internet Archive candidates require explicit item rights, an advertised text/OCR format, and literary metadata; OCR quality and duplication are still unresolved.
- BEIC is exhaustively enumerated through its official Rosetta OAI set. Only Italian records dated no later than 1900 are published; every filtered record remains counted in the summary.
- Gallica's official SRU interface is either completely enumerated or represented by an explicit access blocker. A failed request is never interpreted as a zero-record inventory.
- ELTeC anomalies and post-1900 works fail closed. MIDIA period V remains held because its 1841-1947 bucket crosses the boundary.
- Internet Culturale rows describe collections, not reusable text items; partner terms, item formats, and item counts remain pending.

## Composition interpretation

Metadata counts and word projections are planning evidence, not cleaned characters or training tokens. No new characters are added to the frozen 626,379,622-character broader-pool subtotal by this checkpoint.

## Verification

Two complete cache-backed builds reproduced all four public artifact hashes byte-for-byte. The combined checkpoint-6A/6B focused suite passes 19 tests, and the complete repository suite passes 899 tests.
