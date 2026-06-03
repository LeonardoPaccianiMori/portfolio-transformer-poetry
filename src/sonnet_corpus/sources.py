"""Source registry and source-specific discovery rules."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote

from .wikisource import PageLink, iter_index_links, url_from_title


@dataclass(frozen=True)
class SourceSpec:
    key: str
    author: str
    index_url: str
    source_collection: str
    source_edition: str
    period: str
    license_notes: str
    source_archive: str = "Italian Wikisource"
    include_in_core_pre_petrarch: bool = True
    include_in_expanded_with_petrarch: bool = True


@dataclass(frozen=True)
class CandidatePoem:
    poem_id: str
    title_or_first_line: str
    author: str
    displayed_author: str
    source_archive: str
    source_collection: str
    source_subcollection: str
    source_url: str
    source_edition: str
    license_notes: str
    period: str
    form_evidence: str
    count_method: str
    attribution_status: str
    include_in_core_pre_petrarch: bool
    include_in_expanded_with_petrarch: bool
    audit_notes: str = ""


COMMON_LICENSE = "CC BY-SA / GFDL metadata on Italian Wikisource"


SOURCES: dict[str, SourceSpec] = {
    "giacomo": SourceSpec(
        key="giacomo",
        author="Giacomo da Lentini",
        index_url="https://it.wikisource.org/wiki/Poesie_(Giacomo_da_Lentini)",
        source_collection="Poesie (Giacomo da Lentini)",
        source_edition="Roberto Antonelli, Bulzoni Editore, Roma, 1979",
        period="XIII secolo",
        license_notes=COMMON_LICENSE,
    ),
    "dante": SourceSpec(
        key="dante",
        author="Dante Alighieri",
        index_url="https://it.wikisource.org/wiki/Rime_(Dante)",
        source_collection="Rime (Dante)",
        source_edition="",
        period="XIII secolo",
        license_notes=COMMON_LICENSE,
    ),
    "cavalcanti": SourceSpec(
        key="cavalcanti",
        author="Guido Cavalcanti",
        index_url="https://it.wikisource.org/wiki/Rime_(Cavalcanti)",
        source_collection="Rime (Cavalcanti)",
        source_edition="Ercole Rivalta, Zanichelli, Bologna, 1902",
        period="XIII secolo",
        license_notes=COMMON_LICENSE,
    ),
    "cino": SourceSpec(
        key="cino",
        author="Cino da Pistoia",
        index_url="https://it.wikisource.org/wiki/Autore:Cino_da_Pistoia",
        source_collection="Rime (Cino da Pistoia)",
        source_edition="",
        period="XIV secolo",
        license_notes=COMMON_LICENSE,
    ),
    "cecco": SourceSpec(
        key="cecco",
        author="Cecco Angiolieri",
        index_url="https://it.wikisource.org/wiki/Rime_(Angiolieri)",
        source_collection="Rime (Angiolieri)",
        source_edition="",
        period="XIII secolo",
        license_notes=COMMON_LICENSE,
    ),
    "folgore": SourceSpec(
        key="folgore",
        author="Folgore da San Gimignano",
        index_url="https://it.wikisource.org/wiki/Autore:Folgore_da_San_Gimignano",
        source_collection="Folgore da San Gimignano sonnet cycles",
        source_edition="",
        period="XIV secolo",
        license_notes=COMMON_LICENSE,
    ),
    "guittone": SourceSpec(
        key="guittone",
        author="Guittone d'Arezzo",
        index_url="https://it.wikisource.org/wiki/Rime_(Guittone_d%27Arezzo)",
        source_collection="Rime (Guittone d'Arezzo)",
        source_edition="Francesco Egidi, Laterza, Bari, 1940",
        period="XIII secolo",
        license_notes=COMMON_LICENSE,
    ),
    "petrarca": SourceSpec(
        key="petrarca",
        author="Francesco Petrarca",
        index_url="https://it.wikisource.org/wiki/Canzoniere_(Rerum_vulgarium_fragmenta)",
        source_collection="Canzoniere (Rerum vulgarium fragmenta)",
        source_edition="",
        period="XIV secolo",
        license_notes=COMMON_LICENSE,
        include_in_core_pre_petrarch=False,
    ),
}


OTHER_DISPLAYED_AUTHORS = {
    "Oi deo d'amore": "Abate di Tivoli",
    "Qual om riprende altrù'": "Abate di Tivoli",
    "Con vostro onore facciovi uno 'nvito": "Abate di Tivoli",
    "Solicitando un poco meo savere": "Iacopo Mostacci",
    "Però c'Amore non si pò vedere": "Pier della Vigna",
}


FOLGORE_CYCLES = {
    "Sonetti dei mesi",
    "Sonetti per l'armamento di un cavaliere",
    'Sonetti della "Semana"',
    "Sonetti politici e moraleggianti",
    "Sonetti di dubbia attribuzione",
}

FOLGORE_CYCLE_URLS = {
    "Sonetti dei mesi": "https://it.wikisource.org/wiki/Sonetti_dei_mesi",
    "Sonetti per l'armamento di un cavaliere": (
        "https://it.wikisource.org/wiki/Sonetti_per_l%27armamento_di_un_cavaliere"
    ),
    'Sonetti della "Semana"': "https://it.wikisource.org/wiki/Sonetti_della_%22Semana%22",
    "Sonetti politici e moraleggianti": (
        "https://it.wikisource.org/wiki/Sonetti_politici_e_moraleggianti"
    ),
    "Sonetti di dubbia attribuzione": (
        "https://it.wikisource.org/wiki/Sonetti_di_dubbia_attribuzione"
    ),
}


def source_keys(selection: str) -> list[str]:
    if selection == "all":
        return list(SOURCES)
    keys = [part.strip() for part in selection.split(",") if part.strip()]
    unknown = sorted(set(keys) - set(SOURCES))
    if unknown:
        raise ValueError(f"unknown sources: {', '.join(unknown)}")
    return keys


def discover_candidates(spec: SourceSpec, html: str) -> list[CandidatePoem]:
    links = iter_index_links(html, spec.index_url)
    if spec.key == "giacomo":
        return _discover_giacomo(spec, links)
    if spec.key == "cecco":
        return _discover_cecco(spec, links)
    if spec.key == "folgore":
        return _discover_folgore(spec, links)
    if spec.key == "guittone":
        return _discover_guittone(spec, links)
    if spec.key == "petrarca":
        return _discover_petrarca(spec, links)
    return _discover_mixed_index(spec, links)


def discover_cino_from_titles(spec: SourceSpec, titles: list[str]) -> list[CandidatePoem]:
    candidates: list[CandidatePoem] = []
    for title in titles:
        link = PageLink(url=url_from_title(title), title=title, section="Categoria")
        candidates.append(
            _base_candidate(
                spec,
                link,
                source_subcollection="Categoria:Testi di Cino da Pistoia",
                form_evidence="Wikisource author-text category; include after cleaned 14-line validation",
                count_method="line_count_14",
                attribution_status="secure",
            )
        )
    return candidates


def _base_candidate(
    spec: SourceSpec,
    link: PageLink,
    *,
    source_subcollection: str,
    form_evidence: str,
    count_method: str,
    attribution_status: str = "secure",
    displayed_author: str | None = None,
    audit_notes: str = "",
) -> CandidatePoem:
    title = _title_from_link(link)
    author = displayed_author or spec.author
    return CandidatePoem(
        poem_id=_poem_id(spec.key, title),
        title_or_first_line=title,
        author=author,
        displayed_author=author,
        source_archive=spec.source_archive,
        source_collection=spec.source_collection,
        source_subcollection=source_subcollection,
        source_url=link.url,
        source_edition=spec.source_edition,
        license_notes=spec.license_notes,
        period=spec.period,
        form_evidence=form_evidence,
        count_method=count_method,
        attribution_status=attribution_status,
        include_in_core_pre_petrarch=spec.include_in_core_pre_petrarch,
        include_in_expanded_with_petrarch=spec.include_in_expanded_with_petrarch,
        audit_notes=audit_notes,
    )


def _discover_giacomo(spec: SourceSpec, links: list[PageLink]) -> list[CandidatePoem]:
    candidates: list[CandidatePoem] = []
    for link in links:
        if "/Sonetti/" in link.url:
            title = _title_from_link(link)
            displayed_author = OTHER_DISPLAYED_AUTHORS.get(title)
            status = "correspondence" if displayed_author else "secure"
            candidates.append(
                _base_candidate(
                    spec,
                    link,
                    source_subcollection="Sonetti",
                    form_evidence="explicit Sonetti section",
                    count_method="explicit_index_section",
                    attribution_status=status,
                    displayed_author=displayed_author,
                )
            )
        elif "/Dubbie_attribuzioni/" in link.url and any(
            name in link.url
            for name in [
                "Lo_badalisco_a_lo_specchio_lucente",
                "Guardando_basalisco_velenoso",
            ]
        ):
            candidates.append(
                _base_candidate(
                    spec,
                    link,
                    source_subcollection="Dubbie attribuzioni",
                    form_evidence="manual audit found 14-line doubtful sonnet",
                    count_method="line_count_14",
                    attribution_status="doubtful",
                )
            )
    return candidates


def _discover_cecco(spec: SourceSpec, links: list[PageLink]) -> list[CandidatePoem]:
    candidates = []
    for link in links:
        if "/Rime_(Angiolieri)/" not in link.url:
            continue
        status = "doubtful" if "dubbia" in link.section.lower() else "secure"
        candidates.append(
            _base_candidate(
                spec,
                link,
                source_subcollection=link.section or "Rime",
                form_evidence="explicit sonnet section in source index",
                count_method="explicit_index_section",
                attribution_status=status,
            )
        )
    return candidates


def _discover_folgore(spec: SourceSpec, links: list[PageLink]) -> list[CandidatePoem]:
    candidates: list[CandidatePoem] = []
    for title, url in FOLGORE_CYCLE_URLS.items():
        cycle = PageLink(url=url, title=title, section="Sonetti")
        # Cycle pages are expanded later by the orchestration layer.
        candidates.append(
            _base_candidate(
                spec,
                cycle,
                source_subcollection=_title_from_link(cycle),
                form_evidence="explicit sonnet-cycle page",
                count_method="explicit_index_section",
                attribution_status=(
                    "doubtful" if "dubbia" in _title_from_link(cycle).lower() else "secure"
                ),
                audit_notes="cycle_page_expand",
            )
        )
    return candidates


def _discover_guittone(spec: SourceSpec, links: list[PageLink]) -> list[CandidatePoem]:
    candidates = []
    for link in links:
        section = link.section.lower()
        if "sonetti d" not in section and "sonetti ascetici" not in section:
            continue
        candidates.append(
            _base_candidate(
                spec,
                link,
                source_subcollection=link.section,
                form_evidence="explicit sonnet section in source index",
                count_method="explicit_index_section",
            )
        )
    return candidates


def _discover_petrarca(spec: SourceSpec, links: list[PageLink]) -> list[CandidatePoem]:
    candidates = []
    for link in links:
        if "/Canzoniere_(Rerum_vulgarium_fragmenta)/" not in link.url:
            continue
        candidates.append(
            _base_candidate(
                spec,
                link,
                source_subcollection=link.section,
                form_evidence="canonical external sonnet count; verify by cleaned 14-line count",
                count_method="canonical_external_count",
            )
        )
    return candidates


def _discover_mixed_index(spec: SourceSpec, links: list[PageLink]) -> list[CandidatePoem]:
    candidates = []
    for link in links:
        if spec.key != "cino" and spec.index_url.rstrip("/") + "/" not in link.url:
            continue
        if spec.key == "cino" and link.url == spec.index_url:
            continue
        candidates.append(
            _base_candidate(
                spec,
                link,
                source_subcollection=link.section,
                form_evidence="mixed source index; include after cleaned 14-line validation",
                count_method="line_count_14",
                attribution_status="unknown",
            )
        )
    return candidates


def _title_from_link(link: PageLink) -> str:
    path = unquote(link.url.rsplit("/", 1)[-1])
    return path.replace("_", " ")


def _poem_id(source_key: str, title: str) -> str:
    slug = []
    for char in title.lower():
        if char.isalnum():
            slug.append(char)
        elif char in {" ", "_", "-", "'"}:
            slug.append("_")
    compact = "_".join(part for part in "".join(slug).split("_") if part)
    return f"{source_key}_{compact[:80]}"
