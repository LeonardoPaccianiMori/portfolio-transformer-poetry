from sonnet_corpus.wikisource import extract_poem_text


def test_extract_poem_text_keeps_inline_markup_on_its_verse_line():
    html = """
    <div class="mw-parser-output">
      <div class="poem">
        <p><span style="font-size: xx-large">F</span>orse della fatal quïete<br />
        <span class="numeroriga">4</span>Tu sei l'immago <a title="Autore:Example">cara</a><br />
        <span class="numeroriga">8</span>L'inclito verso di <a title="Autore:Omero">Colui</a> che l'acque<br />
        Baciò Itaca <span class="errata" title="Ulisse?">Ulisse.</span></p>
      </div>
    </div>
    """

    assert extract_poem_text(html) == (
        "Forse della fatal quïete\n"
        "Tu sei l'immago cara\n"
        "L'inclito verso di Colui che l'acque\n"
        "Baciò Itaca Ulisse."
    )
