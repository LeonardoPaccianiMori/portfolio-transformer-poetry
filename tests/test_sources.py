from sonnet_corpus.sources import SOURCES, discover_candidates, discover_cino_from_titles


def test_giacomo_discovery_keeps_correspondence_author():
    html = """
    <div class="mw-parser-output">
      <h2>Sonetti</h2>
      <ul>
        <li><a href="/wiki/Poesie_(Giacomo_da_Lentini)/Sonetti/Oi_deo_d%27amore">XVIIIa</a></li>
        <li><a href="/wiki/Poesie_(Giacomo_da_Lentini)/Sonetti/Lo_giglio_quand%27%C3%A8_colto">XX</a></li>
      </ul>
      <h2>Dubbie attribuzioni</h2>
      <ul>
        <li><a href="/wiki/Poesie_(Giacomo_da_Lentini)/Dubbie_attribuzioni/Lo_badalisco_a_lo_specchio_lucente">Lo badalisco</a></li>
      </ul>
    </div>
    """

    rows = discover_candidates(SOURCES["giacomo"], html)

    assert len(rows) == 3
    correspondence = next(row for row in rows if row.title_or_first_line == "Oi deo d'amore")
    assert correspondence.displayed_author == "Abate di Tivoli"
    assert correspondence.attribution_status == "correspondence"
    doubtful = next(row for row in rows if row.source_subcollection == "Dubbie attribuzioni")
    assert doubtful.attribution_status == "doubtful"


def test_cecco_discovery_marks_doubtful_section():
    html = """
    <div class="mw-parser-output">
      <h2>I sonetti di Cecco Angiolieri</h2>
      <ul><li><a href="/wiki/Rime_(Angiolieri)/I">I</a></li></ul>
      <h2>Sonetti di dubbia attribuzione</h2>
      <ul><li><a href="/wiki/Rime_(Angiolieri)/CIX">CIX</a></li></ul>
    </div>
    """

    rows = discover_candidates(SOURCES["cecco"], html)

    assert [row.attribution_status for row in rows] == ["secure", "doubtful"]


def test_cino_category_titles_become_candidates():
    rows = discover_cino_from_titles(
        SOURCES["cino"],
        ["A che, Roma superba, tante leggi", "Ahi lasso!, ch'io credea trovar pietate"],
    )

    assert len(rows) == 2
    assert rows[0].source_url.endswith("A_che,_Roma_superba,_tante_leggi")
    assert rows[0].count_method == "line_count_14"


def test_folgore_uses_static_cycle_urls():
    rows = discover_candidates(SOURCES["folgore"], "<div class='mw-parser-output'></div>")

    assert len(rows) == 5
    assert {row.source_subcollection for row in rows} >= {
        "Sonetti dei mesi",
        'Sonetti della "Semana"',
    }
