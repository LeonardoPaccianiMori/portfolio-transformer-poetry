class CharTokenizer:
    """Convert individual characters to integer token IDs and back."""

    def __init__(self, char_to_id: dict[str, int]):
        self.char_to_id = char_to_id
        self.id_to_char = {
            token_id: char
            for char, token_id in char_to_id.items()
        }

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        unique_chars = {
            char
            for text in texts
            for char in text
        }
        sorted_chars = sorted(unique_chars)

        char_to_id = {
            char: token_id
            for token_id, char in enumerate(sorted_chars)
        }

        return cls(char_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def encode(self, text: str) -> list[int]:
        return [
            self.char_to_id[char]
            for char in text
        ]

    def decode(self, token_ids: list[int]) -> str:
        characters = [
            self.id_to_char[token_id]
            for token_id in token_ids
        ]
        return "".join(characters)
