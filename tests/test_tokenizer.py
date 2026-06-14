import pytest

from sonnet_corpus.tokenizer import CharTokenizer


def test_encode_decode_round_trip_preserves_text():
    text = "Amor, ch'a nullo amato amar perdona\npiù non dirò. ï"

    tokenizer = CharTokenizer.from_texts([text])

    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)

    assert decoded_text == text
    assert len(token_ids) == len(text)
    assert tokenizer.vocab_size == len(set(text))


def test_from_texts_builds_sorted_vocabulary_from_all_texts():
    texts = ["amor", "più\n"]

    tokenizer = CharTokenizer.from_texts(texts)

    all_characters = set("".join(texts))
    sorted_characters = sorted(all_characters)

    expected_mapping = {
        char: token_id
        for token_id, char in enumerate(sorted_characters)
    }

    assert tokenizer.char_to_id == expected_mapping
    assert tokenizer.vocab_size == len(all_characters)


def test_encode_rejects_unknown_character():
    tokenizer = CharTokenizer.from_texts(["amor"])

    with pytest.raises(KeyError):
        tokenizer.encode("amore")
