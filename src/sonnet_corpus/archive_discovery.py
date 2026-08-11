"""Run checkpoint 6C's bounded, metadata-only archive discovery pass."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep as default_sleep
from typing import Any
from urllib.parse import urlencode

import requests


AUDIT_VERSION = "corpus_archive_discovery_v1"
USER_AGENT = "portfolio-transformer-poetry-archive-discovery/1.0"
BASE_CORPUS_CHARACTERS = 626_379_622

QUERY_FIELDS = (
    "query_id", "surface_id", "authority", "query_text", "endpoint_url",
    "result_boundary", "response_format", "purpose", "retrieval_date",
    "http_status", "content_type", "content_sha256", "result_count",
    "verification_status",
)

EVIDENCE_FIELDS = (
    "evidence_id", "candidate_id", "evidence_type", "authority",
    "source_url", "resolved_url", "retrieval_date", "http_status",
    "content_type", "content_sha256", "evidence_quote",
    "supports_decision", "limitation", "verification_status",
)

DECISION_FIELDS = (
    "candidate_id", "candidate_name", "discovery_query_ids",
    "source_archive", "landing_page", "assigned_corpus_role",
    "language_variety", "historical_period", "register_genre_form",
    "measured_scope", "projected_characters", "projected_works",
    "projected_sonnets", "materiality_basis", "license_status",
    "reuse_obligations", "bulk_access", "overlap_risk",
    "official_evidence_ids", "composition_decision", "final_status",
    "registry_archive_id", "next_action", "activation_status",
)

REGISTRY_FIELDS = (
    "archive_id", "archive_name", "landing_page", "corpus_roles",
    "coverage", "license_or_reuse_status", "bulk_access", "status",
    "next_action", "notes",
)

Progress = Callable[[str], None]
Sleep = Callable[[float], None]


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    surface_id: str
    authority: str
    query_text: str
    endpoint_url: str
    result_boundary: str
    response_format: str
    parser: str
    purpose: str


@dataclass(frozen=True)
class EvidenceSpec:
    evidence_id: str
    candidate_id: str
    evidence_type: str
    url: str
    quote: str
    needles: tuple[str, ...]
    supports: str
    limitation: str


@dataclass(frozen=True)
class ArchiveDiscoveryConfig:
    repo_root: Path
    registry_path: Path
    cache_dir: Path
    query_path: Path
    evidence_path: Path
    decision_path: Path
    json_report_path: Path
    markdown_report_path: Path
    request_timeout_seconds: float = 45.0
    request_delay_seconds: float = 0.25
    max_attempts: int = 3


def _url(base: str, **params: str | int) -> str:
    return f"{base}?{urlencode(params)}"


ZENODO_QUERIES = (
    '"historical Italian"', '"Italian literary"',
    '"Italian sonnet"', '"medieval Italian"',
    '"early modern Italian"', '"diachronic Italian"',
)
GITHUB_QUERIES = (
    "historical Italian corpus in:name,description,readme",
    "Italian literary corpus in:name,description,readme",
    "Italian drama corpus in:name,description,readme",
    "Italian poetry corpus in:name,description,readme",
)
CLARIN_IT_QUERIES = (
    "historical Italian", "Italian literature", "Italian literary",
    "medieval Italian", "old Italian", "Italian corpus",
    "sonnet Italian", "poetry Italian",
)
PHAIDRA_QUERIES = (
    '"letteratura italiana"', '"testi storici"', '"corpus italiano"',
    '"manoscritti italiani"', '"poesia italiana"',
    '"edizione digitale" AND Italiano',
)
DALIA_QUERIES = (
    "letteratura italiana", "testi storici", "corpus italiano",
    "manoscritti", "poesia",
)


def _query_specs() -> tuple[QuerySpec, ...]:
    specs: list[QuerySpec] = []
    for index, query in enumerate(("Italian language", "Italian literature", "digital humanities Italy"), 1):
        specs.append(QuerySpec(
            f"re3data_{index:02d}", "re3data", "curated_repository_registry",
            query, _url("https://www.re3data.org/api/beta/repositories", query=query),
            "all repository matches returned by the public beta API", "xml",
            "re3data", "Find independent repositories rather than individual works.",
        ))
    for index, query in enumerate(ZENODO_QUERIES, 1):
        specs.append(QuerySpec(
            f"zenodo_phrase_{index:02d}", "zenodo", "curated_dataset_repository",
            query, _url("https://zenodo.org/api/records", q=query, size=25),
            "first 25 ranked public metadata records", "json", "zenodo",
            "Find deposited historical, literary, and sonnet datasets.",
        ))
    for index, query in enumerate(GITHUB_QUERIES, 1):
        specs.append(QuerySpec(
            f"github_{index:02d}", "github", "public_code_repository_index",
            query, _url("https://api.github.com/search/repositories", q=query, per_page=20),
            "first 20 ranked public repositories", "json", "github",
            "Find published corpus repositories and derivative mirrors.",
        ))
    for index, query in enumerate(CLARIN_IT_QUERIES, 1):
        specs.append(QuerySpec(
            f"clarin_it_{index:02d}", "clarin_it", "official_national_clarin_repository",
            query, _url(
                "https://dspace-clarin-it.ilc.cnr.it/server/api/discover/search/objects",
                query=query, size=100,
            ), "all matches up to 100 records", "json", "dspace7",
            "Find machine-readable Italian historical resources in CLARIN-IT.",
        ))
    for index, query in enumerate(PHAIDRA_QUERIES, 1):
        specs.append(QuerySpec(
            f"phaidra_{index:02d}", "phaidra_unipd", "official_institutional_repository",
            query, _url(
                "https://phaidra.unipd.it/api/search/select", q=query,
                defType="edismax", wt="json", rows=25,
                fl="pid,dc_title,dc_description,dc_language,dc_subject,dc_rights,dc_date",
            ), "first 25 ranked records plus total count", "json", "solr",
            "Check an independent Italian university repository for primary-text editions.",
        ))
    for index, query in enumerate(DALIA_QUERIES, 1):
        specs.append(QuerySpec(
            f"dalia_{index:02d}", "dalia", "official_cnr_ckan_repository",
            query, _url(
                "https://dalia-bo.cnr.it/api/3/action/package_search",
                q=query, rows=100,
            ), "all matches up to 100 datasets", "json", "ckan",
            "Check the CNR DALIA open-data catalog for historical text datasets.",
        ))
    specs.extend((
        QuerySpec(
            "eurac_01", "eurac_clarin", "official_clarin_repository",
            "all repository sets", _url(
                "https://clarin.eurac.edu/repository/oai/request", verb="ListSets",
            ), "complete OAI-PMH set list", "xml", "oai_sets",
            "Check repository collection scope before any item inventory.",
        ),
        QuerySpec(
            "clarin_family_01", "clarin_resource_family", "official_clarin_curated_overview",
            "Italian literary corpora", "https://www.clarin.eu/resource-families/literary-corpora",
            "complete curated literary-corpora overview", "html", "clarin_family",
            "Check CLARIN's curated literary-corpus family for Italian coverage.",
        ),
        QuerySpec(
            "ota_01", "oxford_text_archive", "official_text_archive_catalog",
            "language equals Italian", _url(
                "https://llds.ling-phil.ox.ac.uk/llds/xmlui/discover", query="*",
                filtertype="language", filter_relational_operator="equals",
                filter="Italian", rpp=100,
            ), "complete Italian-language facet (up to 100 records)", "html",
            "ota_handles", "Enumerate the archive's bounded Italian-language catalog.",
        ),
    ))
    return tuple(specs)


QUERY_SPECS = _query_specs()


EVIDENCE_SPECS = (
    EvidenceSpec(
        "re3data_ilc_cnr", "ilc_cnr_repository", "repository_scope_and_terms",
        "https://www.re3data.org/api/beta/repository/r3d100012262",
        "re3data identifies ILC-CNR for CLARIN-IT as an open repository with CC0 repository metadata.",
        ("ILC-CNR for CLARIN-IT repository", "CC0", "open"),
        "Establishes the independent repository boundary and open metadata layer.",
        "Repository-level CC0 does not replace each deposit's content license.",
    ),
    EvidenceSpec(
        "ilc_rosmini", "ilc_cnr_rosmini", "item_scope_terms_and_access",
        "https://dspace-clarin-it.ilc.cnr.it/server/api/core/items/30dcd581-aa78-4470-9fc9-9f6bd385b97f",
        "Corpus Antonio Rosmini contains 4,311,182 Italian words and is CC BY-NC-SA 4.0 with one downloadable corpus archive.",
        ("Corpus Antonio Rosmini", "4311182", "Attribution-NonCommercial-ShareAlike 4.0", "Italian"),
        "Passes the broad-text materiality floor and establishes explicit non-commercial terms.",
        "It is a single-author nineteenth-century philosophical corpus with unresolved overlap and concentration.",
    ),
    EvidenceSpec(
        "ilc_libretti", "ilc_cnr_opera_libretti", "item_scope_terms_and_access",
        "https://dspace-clarin-it.ilc.cnr.it/server/api/core/items/5c8f43f9-d039-4475-b930-fa13d35b583c",
        "The CC BY 4.0 digital edition contains 56 Italian opera libretti from 1636-1705 in one downloadable XML resource.",
        ("Digital edition of opera libretti", "56", "1636 to 1705", "Attribution 4.0"),
        "Establishes a core-compatible, underrepresented historical verse/drama candidate.",
        "XML size includes markup; cleaned primary characters and sonnet boundaries remain unmeasured.",
    ),
    EvidenceSpec(
        "ilc_libretti_bitstream", "ilc_cnr_opera_libretti", "bitstream_format_and_size",
        "https://dspace-clarin-it.ilc.cnr.it/server/api/core/bitstreams/0422a142-25b6-4bf8-af9b-8ad17f18b31f",
        "The downloadable libretti XML bitstream is 2,737,739 bytes.",
        ("Edizione_digitale_dei_libretti", "2737739"),
        "Pins the machine-readable format and byte-scale materiality evidence.",
        "XML bytes include TEI markup and are not cleaned primary-text characters.",
    ),
    EvidenceSpec(
        "ilc_bellini", "ilc_cnr_bellini", "item_scope_terms_and_access",
        "https://dspace-clarin-it.ilc.cnr.it/server/api/core/items/2e6fe73a-db5c-48aa-ad0a-dcaecc0baddb",
        "Bellini Digital Correspondence exposes 40 Italian letters as TEI XML under CC BY-NC 4.0.",
        ("Bellini Digital Correspondence", "40", "Attribution-NonCommercial 4.0", "Italian"),
        "Establishes compatible terms and an underrepresented epistolary register.",
        "Forty letters do not pass the approved work threshold and cleaned characters are unmeasured.",
    ),
    EvidenceSpec(
        "ilc_bellini_bitstream", "ilc_cnr_bellini", "bitstream_format_and_size",
        "https://dspace-clarin-it.ilc.cnr.it/server/api/core/bitstreams/ca4201ea-d874-4169-983e-9e3e0feeac1c",
        "The downloadable Bellini correspondence TEI ZIP is 407,671 bytes.",
        ("BDC-XML.zip", "407671"),
        "Pins the machine-readable format and compressed byte size.",
        "Compressed bytes do not establish cleaned primary-text characters.",
    ),
    EvidenceSpec(
        "ilc_pelavicino", "ilc_cnr_codice_pelavicino", "item_scope_terms_and_access",
        "https://dspace-clarin-it.ilc.cnr.it/server/api/core/items/776f944f-c44b-4230-9960-f13a861593da",
        "Codice Pelavicino provides 536 CC BY 4.0 XML files spanning the tenth-thirteenth centuries in Italian and Latin.",
        ("536 XML files", "Italian", "Latin", "Attribution 4.0", "66000000"),
        "Establishes material scale for a separately conditioned medieval-document experiment.",
        "The source is mixed Italian/Latin and cannot enter the standard-Italian queue.",
    ),
    EvidenceSpec(
        "ilc_alfieri", "ilc_cnr_alfieri", "item_access_and_overlap",
        "https://dspace-clarin-it.ilc.cnr.it/server/api/core/items/8ed4e1fd-05b1-4fb5-b836-0f865a5794f6",
        "The Alfieri service lists 19 tragedies but publishes no downloadable files and overlaps Biblioteca Italiana lineage.",
        ("Vittorio Alfieri's tragedies", "local.files.count", "0", "Italian"),
        "Documents a named historical lead and its concrete access/overlap blockers.",
        "A service page is not a reusable bulk primary-text archive.",
    ),
    EvidenceSpec(
        "ilc_bruno", "ilc_cnr_bruno", "item_access_and_overlap",
        "https://dspace-clarin-it.ilc.cnr.it/server/api/core/items/9a62878c-a2f9-4544-b513-381094c825f7",
        "The Giordano Bruno corpus advertises 334,854 tokens but publishes no downloadable files and overlaps canonical editions already inventoried.",
        ("Dialoghi Italiani di Giordano Bruno", "334854", "local.files.count", "0"),
        "Documents material scale but no reusable bulk-access route.",
        "The repository record states no item content license and exposes no files.",
    ),
    EvidenceSpec(
        "ota_cc0_early_modern", "oxford_text_archive_italian", "item_terms_and_access",
        "https://llds.ling-phil.ox.ac.uk/llds/xmlui/handle/20.500.14106/A14166?show=full",
        "An early-modern Italian poetry item is public, downloadable as XML, and dedicated under CC0 1.0.",
        ("Rime di Petruccio", "publicdomain/zero/1.0", "A14166.xml"),
        "Confirms reusable keyboarded Italian primary text in the early-modern subset.",
        "Page images and supplementary files can retain separate restrictions.",
    ),
    EvidenceSpec(
        "ota_by_nc_sa_item", "oxford_text_archive_italian", "item_terms_and_access",
        "https://llds.ling-phil.ox.ac.uk/llds/xmlui/handle/20.500.14106/0302?show=full",
        "A canonical Italian item is publicly downloadable as text under CC BY-NC-SA 3.0.",
        ("Il libro del cortegiano", "by-nc-sa/3.0", "cortegiano-0302.txt"),
        "Confirms the archive's item-level compatible non-commercial license route.",
        "Canonical works have high BibIt/Gutenberg overlap and require item-level deduplication.",
    ),
    EvidenceSpec(
        "itadracor_readme", "itadracor", "scope_lineage_and_reuse_notice",
        "https://raw.githubusercontent.com/dracor-org/itadracor/main/README.md",
        "ItaDraCor contains 157 Italian plays, mostly retrieved from Biblioteca Italiana, under the source's personal/scientific-use notice.",
        ("contains 157 original Italian plays", "mostly", "Biblioteca Italiana", "uso personale o scientifico"),
        "Establishes measured scope and direct canonical lineage overlap.",
        "No independent corpus-text license or unique-value claim is established.",
    ),
    EvidenceSpec(
        "postdata_bibit_repo", "postdata_biblioteca_italiana", "derivative_repository_lineage",
        "https://api.github.com/repos/linhd-postdata/biblioteca_italiana",
        "The POSTDATA repository is explicitly a poetry corpus from Biblioteca Italiana.",
        ("linhd-postdata/biblioteca_italiana", "Poetry corpus from the Biblioteca Italiana"),
        "Identifies a derivative mirror of an already-complete archive.",
        "Repository/code licensing cannot replace the upstream edition terms.",
    ),
    EvidenceSpec(
        "zenodo_eneide", "zenodo_eneide", "dataset_scope_terms_and_lineage",
        "https://zenodo.org/api/records/17407356",
        "ENEIDE is a CC BY-NC-SA 4.0 NER dataset derived from Zibaldone and Aldo Moro digital editions.",
        ("ENEIDE", "2,111 documents", "Digital Zibaldone", "Aldo Moro Digitale", "cc-by-nc-sa-4.0"),
        "Establishes annotation scope and mixed historical/modern source lineage.",
        "It is not an independent historical primary-text archive and includes post-1900 material.",
    ),
    EvidenceSpec(
        "zenodo_htromance", "zenodo_htromance", "dataset_scope_terms_and_language",
        "https://zenodo.org/api/records/14718897",
        "HTRomance is a CC BY 4.0 medieval Italian/Venetian HTR and layout-segmentation ground-truth package.",
        ("HTRomance", "Venitian", "cc-by-4.0", "Handwritten Text Recognition"),
        "Identifies a permitted but conditioned HTR dataset.",
        "Primary-text character scale and standard-Italian compatibility are not established.",
    ),
    EvidenceSpec(
        "github_embedding_repo", "github_word_embedding_literature", "repository_scope_and_access",
        "https://api.github.com/repos/giocoal/word-embedding-italian-literature",
        "The repository publishes analysis/code under MIT but no primary corpus payload.",
        ("word-embedding-italian-literature", "MIT License", "corpuses"),
        "Documents a corpus-discovery lead found by the GitHub query.",
        "A code license does not license uncommitted source texts.",
    ),
)


def _decision_rows() -> list[dict[str, str]]:
    rows = [
        ("ilc_cnr_rosmini", "Corpus Antonio Rosmini - Serbati", "re3data_01;clarin_it_06", "ILC-CNR for CLARIN-IT", "https://hdl.handle.net/20.500.11752/ILC-57", "auxiliary_capped_ottocento_bridge", "standard Italian", "nineteenth century", "philosophical and religious prose", "4,311,182 words in one downloadable corpus archive", "", "1", "0", "pass: 4,311,182 words necessarily exceed the 1,000,000-character floor", "CC BY-NC-SA 4.0", "Attribute ILC-CNR and contributors; non-commercial use; ShareAlike", "DSpace item metadata plus one downloadable corpus archive", "very high single-author concentration; cross-archive overlap unmeasured", "re3data_ilc_cnr;ilc_rosmini", "auxiliary", "eligible_bounded_source_audit_inactive", "ilc_cnr_historical_corpora", "Run a bounded format, primary-text, overlap, and concentration audit; no full exposure without a capped bridge experiment.", "inactive_metadata_only"),
        ("ilc_cnr_opera_libretti", "Digital edition of opera libretti", "re3data_01;clarin_it_06", "ILC-CNR for CLARIN-IT", "https://hdl.handle.net/20.500.11752/OPEN-979", "core_training_candidate", "standard Italian", "1636-1705", "opera libretti; historical verse and drama", "56 texts in one 2,737,739-byte XML resource", "", "56", "0", "pass: 56 underrepresented seventeenth-century works exceed the 10-work scarcity threshold", "CC BY 4.0", "Attribute the digital edition and named contributors", "DSpace item metadata plus one downloadable XML resource", "moderate canonical/embedded-poem overlap; exact text unmeasured", "re3data_ilc_cnr;ilc_libretti;ilc_libretti_bitstream", "core_training", "eligible_bounded_source_audit_inactive", "ilc_cnr_historical_corpora", "Run a bounded XML extraction, quality, overlap, embedded-sonnet, and concentration audit.", "inactive_metadata_only"),
        ("ilc_cnr_bellini", "Bellini Digital Correspondence", "re3data_01;clarin_it_06", "ILC-CNR for CLARIN-IT", "https://hdl.handle.net/20.500.11752/OPEN-1000", "auxiliary_capped_ottocento_bridge", "standard Italian", "nineteenth century", "private correspondence", "40 textual units in one 407,671-byte TEI ZIP", "", "40", "0", "pass: 40 underrepresented epistolary units exceed the 10-work scarcity threshold", "CC BY-NC 4.0", "Attribute the edition; non-commercial use", "DSpace item metadata plus one downloadable TEI ZIP", "single-author concentration; likely low canonical overlap", "re3data_ilc_cnr;ilc_bellini;ilc_bellini_bitstream", "auxiliary", "eligible_bounded_source_audit_inactive", "ilc_cnr_historical_corpora", "Run a bounded TEI extraction, primary-text, overlap, and concentration audit; activate nothing.", "inactive_metadata_only"),
        ("ilc_cnr_codice_pelavicino", "Codice Pelavicino", "re3data_01;clarin_it_04;clarin_it_05", "ILC-CNR for CLARIN-IT", "https://hdl.handle.net/20.500.11752/OPEN-1012", "auxiliary_conditioned_experiment", "mixed Italian and Latin", "tenth-thirteenth centuries", "ecclesiastical and legal charters", "536 XML files; repository reports 66,000,000 bytes", "", "536", "0", "conditioned pass: more than 100 documents, but not a standard-Italian source", "CC BY 4.0", "Attribute the digital edition and named contributors", "DSpace item metadata and downloadable XML files", "high Latin dominance and source-internal repetition risk", "re3data_ilc_cnr;ilc_pelavicino", "auxiliary", "conditioned_auxiliary_experiment_required_inactive", "ilc_cnr_historical_corpora", "Keep outside the standard queue; require a separately approved mixed-language experiment before extraction.", "inactive_conditioned_only"),
        ("ilc_cnr_alfieri", "Vittorio Alfieri's tragedies", "clarin_it_02;clarin_it_04", "ILC-CNR for CLARIN-IT", "https://hdl.handle.net/20.500.11752/ILC-63", "excluded", "standard Italian", "eighteenth century", "tragedy", "19 named tragedies; zero downloadable files", "", "19", "0", "fail: no bulk text and fewer than 100 works; BibIt lineage overlap", "not established for corpus text", "Retain ILC-CNR and source-edition provenance", "service page only", "very high BibIt overlap", "ilc_alfieri", "excluded", "closed_access_and_overlap_exclusion", "", "No further audit; canonical source archives already cover the works.", "inactive_excluded"),
        ("ilc_cnr_bruno", "Dialoghi Italiani di Giordano Bruno", "clarin_it_06", "ILC-CNR for CLARIN-IT", "https://hdl.handle.net/20.500.11752/ILC-67", "excluded", "standard Italian", "sixteenth century", "philosophical dialogues", "334,854 advertised tokens; zero downloadable files", "", "", "0", "fail: no licensed bulk text route and high canonical overlap", "not established for corpus text", "Retain ILC-CNR provenance", "service page only", "very high BibIt/Gutenberg overlap", "ilc_bruno", "excluded", "closed_access_and_overlap_exclusion", "", "No further audit unless a licensed downloadable source with unique text appears.", "inactive_excluded"),
        ("oxford_text_archive_italian", "Oxford Text Archive Italian-language collection", "re3data_01;re3data_02;ota_01", "Oxford Text Archive", "https://llds.ling-phil.ox.ac.uk/llds/xmlui/", "core_training_candidate", "standard and historical Italian; item review required", "fourteenth-nineteenth centuries plus later exclusions", "prose, poetry, letters, reference, and early printed works", "43 complete Italian-language catalog records, including at least 10 early-modern primary works", "", "43", "", "pass: at least 10 works from an underrepresented early-modern register", "item-level CC0 or CC BY-NC-SA; verify every record", "Preserve item creators, OTA/TCP credit, license URI, and any non-commercial/ShareAlike terms", "DSpace catalog plus item XML/TXT/EPUB bitstreams", "high for canonical works; lower for EEBO-TCP Italian works", "ota_cc0_early_modern;ota_by_nc_sa_item", "core_training", "eligible_bounded_source_audit_inactive", "oxford_text_archive", "Inventory all 43 records, verify item terms/dates/languages, then probe only compatible unique candidates against existing corpora.", "inactive_metadata_only"),
        ("itadracor", "ItaDraCor", "github_01;github_03", "DraCor / GitHub", "https://github.com/dracor-org/itadracor", "excluded", "standard Italian", "historical through nineteenth century", "drama", "157 Italian plays, mostly retrieved from Biblioteca Italiana", "", "157", "0", "fail unique-value gate: direct derivative of an already-complete source", "upstream personal/scientific-use notice; no independent corpus license established", "Retain DraCor contributor and BibIt lineage notices", "Git repository TEI files", "near-total direct BibIt lineage", "itadracor_readme", "excluded", "closed_canonical_derivative_exclusion", "", "Use only as structural metadata reference; do not reacquire duplicate text.", "inactive_excluded"),
        ("postdata_biblioteca_italiana", "POSTDATA Biblioteca Italiana poetry corpus", "github_04", "GitHub / POSTDATA", "https://github.com/linhd-postdata/biblioteca_italiana", "excluded", "standard and conditioned Italian inherited from BibIt", "historical", "poetry", "Derivative poetry corpus explicitly sourced from Biblioteca Italiana", "", "", "", "fail unique-value gate: mirror/derivative of an already-complete source", "repository terms do not replace upstream edition terms", "Retain POSTDATA and BibIt provenance", "Git repository", "direct BibIt derivative", "postdata_bibit_repo", "excluded", "closed_canonical_derivative_exclusion", "", "No text audit; existing BibIt extraction is canonical.", "inactive_excluded"),
        ("zenodo_eneide", "ENEIDE historical Italian NER dataset", "zenodo_phrase_01", "Zenodo", "https://zenodo.org/records/17407356", "excluded", "Italian", "1817-1978 source span", "entity annotations over diary and political works", "2,111 documents in a 972,266-byte annotation package", "", "2", "0", "fail: annotation payload below threshold and mixed with post-1900 source material", "CC BY-NC-SA 4.0", "Attribute creators; non-commercial use; ShareAlike", "single Zenodo ZIP", "Digital Zibaldone overlap plus modern Aldo Moro content", "zenodo_eneide", "excluded", "closed_annotation_and_period_exclusion", "", "No primary-text acquisition; retain as an evaluation-resource reference only.", "inactive_excluded"),
        ("zenodo_htromance", "HTRomance medieval Italian HTR ground truth", "zenodo_phrase_04", "Zenodo / HTRomance", "https://zenodo.org/records/14718897", "excluded", "medieval Italian and Venetian", "medieval", "manuscript HTR/layout ground truth", "104,649,449-byte package dominated by images/layout data; primary characters unmeasured", "", "", "0", "fail standard queue: conditioned language and no measured 250,000-character primary-text contribution", "CC BY 4.0", "Attribute HTRomance creators", "single Zenodo ZIP", "unknown text overlap; high HTR/image overhead", "zenodo_htromance", "excluded", "closed_below_materiality_conditioned_htr", "", "Keep outside the standard queue; reconsider only under an approved HTR/dialect experiment.", "inactive_excluded"),
        ("github_word_embedding_literature", "Italian-literature word-embedding project", "github_02", "GitHub", "https://github.com/giocoal/word-embedding-italian-literature", "excluded", "Italian", "mixed literary periods", "analysis code and derived embeddings", "MIT-licensed code repository with no committed primary corpus payload", "0", "0", "0", "fail: no primary-text archive", "MIT applies to code only", "Credit code authors if reused", "code repository", "source texts originate elsewhere and are not published", "github_embedding_repo", "excluded", "closed_no_primary_text_exclusion", "", "No corpus action.", "inactive_excluded"),
        ("phaidra_unipd", "PHAIDRA University of Padua", "re3data_03;phaidra_01;phaidra_02;phaidra_03;phaidra_04;phaidra_05;phaidra_06", "PHAIDRA", "https://phaidra.unipd.it/", "excluded", "mixed", "mixed", "institutional repository", "Frozen queries returned no material historical Italian primary-text corpus", "0", "0", "0", "fail: no material primary-text candidate in the frozen query boundary", "item-level", "Item-specific", "public Solr metadata API and OAI-PMH", "not applicable", "", "excluded", "closed_discovery_surface_no_material_candidate", "", "Retain query evidence; no further audit.", "inactive_excluded"),
        ("dalia", "DALIA CNR open-data catalog", "re3data_03;dalia_01;dalia_02;dalia_03;dalia_04;dalia_05", "CNR DALIA", "https://dalia-bo.cnr.it/", "excluded", "mixed", "mostly modern", "CNR library open datasets", "All five frozen literary/historical queries returned zero records", "0", "0", "0", "fail: no candidate", "catalog metadata and item-level Creative Commons", "Item-specific", "public CKAN API", "not applicable", "", "excluded", "closed_discovery_surface_no_material_candidate", "", "Retain query evidence; no further audit.", "inactive_excluded"),
        ("eurac_clarin", "Eurac Research CLARIN Centre", "re3data_01;eurac_01", "Eurac CLARIN", "https://clarin.eurac.edu/", "excluded", "multilingual with South-Tyrol varieties", "contemporary", "learner language, CMC, terminology, and regional corpora", "Complete OAI set list exposes no historical Italian literary collection", "0", "0", "0", "fail: no target-period literary collection", "repository metadata CC0; item-level terms", "Item-specific", "OAI-PMH and DSpace repository", "not applicable", "", "excluded", "closed_discovery_surface_no_material_candidate", "", "Retain query evidence; no further audit.", "inactive_excluded"),
        ("clarin_resource_family", "CLARIN curated literary-corpora family", "clarin_family_01", "CLARIN ERIC", "https://www.clarin.eu/resource-families/literary-corpora", "excluded", "multilingual", "mixed", "curated literary corpus index", "Complete overview names no Italian-language literary corpus", "0", "0", "0", "fail: no Italian candidate", "metadata discovery only", "Cite CLARIN if referenced", "curated overview", "not applicable", "", "excluded", "closed_discovery_surface_no_material_candidate", "", "Retain query evidence; no further audit.", "inactive_excluded"),
    ]
    return [dict(zip(DECISION_FIELDS, row, strict=True)) for row in rows]


REGISTRY_ADDITIONS = (
    {
        "archive_id": "ilc_cnr_historical_corpora",
        "archive_name": "ILC-CNR for CLARIN-IT historical corpora",
        "landing_page": "https://dspace-clarin-it.ilc.cnr.it/",
        "corpus_roles": "general historical | non-sonnet poetry | Ottocento bridge | conditioned medieval documents",
        "coverage": "49 bounded Italian-corpus metadata matches; four historical deposits retained for bounded audit",
        "license_or_reuse_status": "item-level CC BY 4.0, CC BY-NC 4.0, or CC BY-NC-SA 4.0; repository metadata CC0",
        "bulk_access": "DSpace 7 metadata API plus item bitstreams",
        "status": "discovered_material_bounded_audit_pending_inactive",
        "next_action": "Checkpoint 6D: audit Rosmini, opera libretti, and Bellini formats/overlap; keep Codice Pelavicino conditioned and separately gated.",
        "notes": "Rosmini reports 4,311,182 words; libretti contains 56 texts from 1636-1705; Bellini contains 40 letters; Codice Pelavicino contains 536 mixed Italian/Latin XML files. No text acquired or activated.",
    },
    {
        "archive_id": "oxford_text_archive",
        "archive_name": "Oxford Text Archive Italian-language collection",
        "landing_page": "https://llds.ling-phil.ox.ac.uk/llds/xmlui/",
        "corpus_roles": "general historical | non-sonnet poetry | sonnets | Ottocento bridge",
        "coverage": "43 complete Italian-language catalog records; at least 10 underrepresented early-modern works",
        "license_or_reuse_status": "item-level CC0 or CC BY-NC-SA; verify every record and preserve TCP/OTA attribution",
        "bulk_access": "DSpace catalog plus item XML/TXT/EPUB bitstreams",
        "status": "discovered_material_bounded_audit_pending_inactive",
        "next_action": "Checkpoint 6D: inventory all 43 items, verify terms/dates/languages, and probe only compatible unique candidates.",
        "notes": "Canonical works have high BibIt/Gutenberg overlap; EEBO-TCP Italian works may add rare early-modern registers. No text acquired or activated.",
    },
)


def build_archive_discovery(
    config: ArchiveDiscoveryConfig,
    *,
    session: requests.Session | None = None,
    sleep: Sleep = default_sleep,
    progress: Progress | None = None,
    query_rows: list[dict[str, str]] | None = None,
    evidence_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build checkpoint 6C artifacts without downloading or activating corpus text."""

    _validate_config(config)
    queries = query_rows or fetch_query_evidence(
        config, session=session, sleep=sleep, progress=progress,
    )
    evidence = evidence_rows or fetch_candidate_evidence(
        config, session=session, sleep=sleep, progress=progress,
    )
    decisions = _decision_rows()
    _validate_accounting(queries, evidence, decisions)

    _write_csv(config.query_path, QUERY_FIELDS, sorted(queries, key=lambda row: row["query_id"]))
    _write_csv(config.evidence_path, EVIDENCE_FIELDS, sorted(evidence, key=lambda row: row["evidence_id"]))
    _write_csv(config.decision_path, DECISION_FIELDS, decisions)
    _update_registry(config.registry_path)

    audit_date = max(
        [row["retrieval_date"] for row in queries]
        + [row["retrieval_date"] for row in evidence],
    )
    status_counts = Counter(row["final_status"] for row in decisions)
    role_counts = Counter(row["composition_decision"] for row in decisions)
    registry_ids = [row["archive_id"] for row in REGISTRY_ADDITIONS]
    report: dict[str, Any] = {
        "report_version": AUDIT_VERSION,
        "audit_date": audit_date,
        "query_count": len(queries),
        "surface_count": len({row["surface_id"] for row in queries}),
        "surface_ids": sorted({row["surface_id"] for row in queries}),
        "candidate_decision_count": len(decisions),
        "evidence_count": len(evidence),
        "eligible_standard_audit_count": sum(
            row["final_status"] == "eligible_bounded_source_audit_inactive"
            for row in decisions
        ),
        "eligible_standard_candidate_ids": [
            row["candidate_id"] for row in decisions
            if row["final_status"] == "eligible_bounded_source_audit_inactive"
        ],
        "conditioned_auxiliary_count": sum(
            row["final_status"] == "conditioned_auxiliary_experiment_required_inactive"
            for row in decisions
        ),
        "materiality_hold_count": sum(
            row["final_status"] == "hold_materiality_unverified_inactive"
            for row in decisions
        ),
        "closed_or_excluded_count": sum(
            row["final_status"].startswith("closed_") for row in decisions
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "composition_role_counts": dict(sorted(role_counts.items())),
        "registry_addition_count": len(REGISTRY_ADDITIONS),
        "registry_addition_ids": registry_ids,
        "base_corpus_characters": BASE_CORPUS_CHARACTERS,
        "activated_corpus_characters": 0,
        "corpus_text_acquired": False,
        "text_activated": False,
        "v7_created": False,
        "mixture_weights_assigned": False,
        "gpu_work_started": False,
        "cache_deleted": False,
        "stop_rule_result": "material_archives_found_schedule_checkpoint_6D_before_checkpoint_7",
        "next_checkpoint": "6D bounded ILC-CNR and Oxford Text Archive source audit",
        "artifact_sha256": {
            "query_csv": _sha256_file(config.query_path),
            "evidence_csv": _sha256_file(config.evidence_path),
            "decision_csv": _sha256_file(config.decision_path),
            "registry_csv": _sha256_file(config.registry_path),
        },
    }
    _write_text(
        config.json_report_path,
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    _write_text(config.markdown_report_path, _render_markdown(report, decisions))
    return report


def fetch_query_evidence(
    config: ArchiveDiscoveryConfig,
    *,
    session: requests.Session | None = None,
    sleep: Sleep = default_sleep,
    progress: Progress | None = None,
) -> list[dict[str, str]]:
    client = session or requests.Session()
    client.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    rows: list[dict[str, str]] = []
    for index, spec in enumerate(QUERY_SPECS, 1):
        _report(progress, f"query={index}/{len(QUERY_SPECS)} id={spec.query_id} surface={spec.surface_id} start")
        content, metadata = _fetch_cached(
            config, client, spec.query_id, spec.endpoint_url,
            category="queries", sleep=sleep,
        )
        result_count = parse_query_result_count(spec.parser, content)
        rows.append({
            "query_id": spec.query_id,
            "surface_id": spec.surface_id,
            "authority": spec.authority,
            "query_text": spec.query_text,
            "endpoint_url": spec.endpoint_url,
            "result_boundary": spec.result_boundary,
            "response_format": spec.response_format,
            "purpose": spec.purpose,
            "retrieval_date": metadata["retrieval_date"],
            "http_status": str(metadata["http_status"]),
            "content_type": metadata["content_type"],
            "content_sha256": metadata["content_sha256"],
            "result_count": str(result_count),
            "verification_status": "official_or_curated_metadata_response_verified",
        })
        _report(progress, f"query={index}/{len(QUERY_SPECS)} id={spec.query_id} results={result_count} complete")
    return rows


def fetch_candidate_evidence(
    config: ArchiveDiscoveryConfig,
    *,
    session: requests.Session | None = None,
    sleep: Sleep = default_sleep,
    progress: Progress | None = None,
) -> list[dict[str, str]]:
    client = session or requests.Session()
    client.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    rows: list[dict[str, str]] = []
    for index, spec in enumerate(EVIDENCE_SPECS, 1):
        _report(progress, f"evidence={index}/{len(EVIDENCE_SPECS)} id={spec.evidence_id} start")
        content, metadata = _fetch_cached(
            config, client, spec.evidence_id, spec.url,
            category="evidence", sleep=sleep,
        )
        plain = _plain(content)
        missing = [needle for needle in spec.needles if _clean(needle).casefold() not in plain.casefold()]
        if missing:
            raise ValueError(f"official evidence drift for {spec.evidence_id}: missing {missing}")
        rows.append({
            "evidence_id": spec.evidence_id,
            "candidate_id": spec.candidate_id,
            "evidence_type": spec.evidence_type,
            "authority": "official_first_party" if "re3data.org" not in spec.url else "curated_repository_registry",
            "source_url": spec.url,
            "resolved_url": metadata["resolved_url"],
            "retrieval_date": metadata["retrieval_date"],
            "http_status": str(metadata["http_status"]),
            "content_type": metadata["content_type"],
            "content_sha256": metadata["content_sha256"],
            "evidence_quote": spec.quote,
            "supports_decision": spec.supports,
            "limitation": spec.limitation,
            "verification_status": "content_needles_verified",
        })
        _report(progress, f"evidence={index}/{len(EVIDENCE_SPECS)} id={spec.evidence_id} complete")
    return rows


def parse_query_result_count(parser: str, content: bytes) -> int:
    """Return the advertised result count for one frozen discovery surface."""

    if parser == "re3data":
        return len(ET.fromstring(content).findall("repository"))
    if parser == "oai_sets":
        root = ET.fromstring(content)
        return len(root.findall(".//{http://www.openarchives.org/OAI/2.0/}set"))
    if parser == "ota_handles":
        handles = set(re.findall(
            rb"/llds/xmlui/handle/20\.500\.14106/([^\"'?&<]+)", content,
        ))
        return len(handles)
    if parser == "clarin_family":
        match = re.search(rb'<meta name="description" content="([^"]*)"', content, re.I)
        description = html.unescape(match.group(1).decode("utf-8", "replace")) if match else ""
        return int(bool(re.search(r"\bItalian\b", description, re.I)))

    payload = json.loads(content)
    if parser == "zenodo":
        total = payload["hits"]["total"]
        return int(total["value"] if isinstance(total, dict) else total)
    if parser == "github":
        return int(payload["total_count"])
    if parser == "dspace7":
        return int(payload["_embedded"]["searchResult"]["page"]["totalElements"])
    if parser == "solr":
        return int(payload["response"]["numFound"])
    if parser == "ckan":
        return int(payload["result"]["count"])
    raise ValueError(f"unknown query parser: {parser}")


def _validate_accounting(
    queries: list[dict[str, str]],
    evidence: list[dict[str, str]],
    decisions: list[dict[str, str]],
) -> None:
    expected_queries = {spec.query_id for spec in QUERY_SPECS}
    actual_queries = {row["query_id"] for row in queries}
    if expected_queries != actual_queries or len(queries) != len(actual_queries):
        raise ValueError("query accounting does not match the frozen discovery matrix")
    expected_evidence = {spec.evidence_id for spec in EVIDENCE_SPECS}
    actual_evidence = {row["evidence_id"] for row in evidence}
    if expected_evidence != actual_evidence or len(evidence) != len(actual_evidence):
        raise ValueError("candidate evidence accounting does not match the frozen specification")
    candidate_ids = {row["candidate_id"] for row in decisions}
    if len(candidate_ids) != len(decisions):
        raise ValueError("candidate decisions must be unique")
    allowed_roles = {"core_training", "auxiliary", "excluded"}
    for row in decisions:
        query_refs = set(filter(None, row["discovery_query_ids"].split(";")))
        evidence_refs = set(filter(None, row["official_evidence_ids"].split(";")))
        if not query_refs <= actual_queries:
            raise ValueError(f"missing discovery query for {row['candidate_id']}")
        if not evidence_refs <= actual_evidence:
            raise ValueError(f"missing official evidence for {row['candidate_id']}")
        if row["composition_decision"] not in allowed_roles:
            raise ValueError(f"candidate lacks exactly one composition role: {row['candidate_id']}")
        if not row["activation_status"].startswith("inactive_"):
            raise ValueError("checkpoint 6C cannot activate corpus text")
        if row["final_status"] == "eligible_bounded_source_audit_inactive" and not row["materiality_basis"].startswith("pass:"):
            raise ValueError(f"eligible candidate does not pass materiality: {row['candidate_id']}")
    registry_ids = {row["archive_id"] for row in REGISTRY_ADDITIONS}
    referenced_registry_ids = {row["registry_archive_id"] for row in decisions if row["registry_archive_id"]}
    if registry_ids != referenced_registry_ids:
        raise ValueError("registry additions do not reconcile with candidate decisions")


def _validate_config(config: ArchiveDiscoveryConfig) -> None:
    if config.request_timeout_seconds <= 0:
        raise ValueError("request timeout must be positive")
    if config.request_delay_seconds < 0:
        raise ValueError("request delay cannot be negative")
    if config.max_attempts <= 0:
        raise ValueError("max attempts must be positive")


def _update_registry(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if fields != REGISTRY_FIELDS:
        raise ValueError("archive registry schema drift")
    additions = {row["archive_id"]: dict(row) for row in REGISTRY_ADDITIONS}
    existing_ids = {row["archive_id"] for row in rows}
    unknown_existing = (existing_ids & additions.keys()) - additions.keys()
    if unknown_existing:
        raise ValueError(f"unexpected registry collision: {sorted(unknown_existing)}")
    updated = [additions.get(row["archive_id"], row) for row in rows]
    for archive_id in (row["archive_id"] for row in REGISTRY_ADDITIONS):
        if archive_id not in existing_ids:
            updated.append(additions[archive_id])
    if len({row["archive_id"] for row in updated}) != len(updated):
        raise ValueError("archive registry contains duplicate IDs")
    _write_csv(path, REGISTRY_FIELDS, updated)


def _fetch_cached(
    config: ArchiveDiscoveryConfig,
    session: requests.Session,
    cache_id: str,
    url: str,
    *,
    category: str,
    sleep: Sleep,
) -> tuple[bytes, dict[str, Any]]:
    directory = config.cache_dir / category
    body_path = directory / f"{cache_id}.bin"
    metadata_path = directory / f"{cache_id}.json"
    if body_path.is_file() and metadata_path.is_file():
        content = body_path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_url") != url:
            raise ValueError(f"cached URL drift for {cache_id}")
        if hashlib.sha256(content).hexdigest() != metadata.get("content_sha256"):
            raise ValueError(f"cached content hash mismatch for {cache_id}")
        return content, metadata

    response: requests.Response | None = None
    error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            response = session.get(url, timeout=config.request_timeout_seconds)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            error = exc
            if attempt < config.max_attempts:
                sleep(config.request_delay_seconds * attempt)
    if response is None or not response.ok:
        raise RuntimeError(f"failed to fetch discovery evidence {cache_id}: {error}")
    content = response.content
    retrieved = response.headers.get("Date", "")
    try:
        retrieval_date = datetime.strptime(retrieved, "%a, %d %b %Y %H:%M:%S %Z").date().isoformat()
    except ValueError:
        retrieval_date = datetime.now(UTC).date().isoformat()
    metadata = {
        "source_url": url,
        "resolved_url": response.url,
        "retrieval_date": retrieval_date,
        "http_status": response.status_code,
        "content_type": response.headers.get("Content-Type", "").split(";", 1)[0],
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }
    directory.mkdir(parents=True, exist_ok=True)
    body_path.write_bytes(content)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if config.request_delay_seconds:
        sleep(config.request_delay_seconds)
    return content, metadata


def _render_markdown(report: dict[str, Any], decisions: list[dict[str, str]]) -> str:
    eligible = [row for row in decisions if row["final_status"] == "eligible_bounded_source_audit_inactive"]
    conditioned = [row for row in decisions if row["final_status"] == "conditioned_auxiliary_experiment_required_inactive"]
    holds = [row for row in decisions if row["final_status"] == "hold_materiality_unverified_inactive"]
    lines = [
        "# Checkpoint 6C: Final Archive Discovery Pass", "",
        f"Audit date: `{report['audit_date']}`", "",
        "## Outcome", "",
        f"The frozen {report['query_count']}-query matrix covered {report['surface_count']} independent discovery surfaces. "
        f"It resolved {report['candidate_decision_count']} candidate or surface decisions from {report['evidence_count']} pinned evidence records.",
        "",
        "The evidence-based stop rule did **not** close directly into checkpoint 7: material source boundaries were found. "
        "They remain metadata-only and require checkpoint 6D before cross-archive canonicalization.", "",
        "## Standard-queue bounded audits", "",
        "| Candidate | Role | Materiality | Next action |", "|---|---|---|---|",
    ]
    for row in eligible:
        lines.append(f"| {row['candidate_name']} | `{row['assigned_corpus_role']}` | {row['materiality_basis']} | {row['next_action']} |")
    lines.extend(["", "## Conditioned and held discoveries", ""])
    for row in conditioned + holds:
        lines.append(f"- **{row['candidate_name']}** — `{row['final_status']}`. {row['materiality_basis']} {row['next_action']}")
    lines.extend([
        "", "## Registry closure", "",
        f"- New inactive registry boundaries: {', '.join(f'`{value}`' for value in report['registry_addition_ids'])}",
        f"- Closed or excluded discoveries: {report['closed_or_excluded_count']}",
        f"- Existing broader-pool subtotal: {report['base_corpus_characters']:,} characters",
        "- New corpus characters acquired or activated: 0", "",
        "## Frozen constraints", "",
        "- Discovery results and official evidence are metadata only.",
        "- Item-level terms override repository-level metadata licenses.",
        "- Canonical derivatives of BibIt remain excluded even when technically downloadable.",
        "- Mixed Italian/Latin or Venetian HTR resources remain outside the standard-Italian queue.",
        "- No corpus text, V7 split, mixture weight, cache deletion, or GPU work is authorized.",
        "- Next checkpoint: 6D bounded ILC-CNR and Oxford Text Archive audit.", "",
    ])
    return "\n".join(lines)


def _plain(content: bytes) -> str:
    text = html.unescape(content.decode("utf-8", "replace"))
    text = re.sub(r"<[^>]+>", " ", text)
    return _clean(text)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
