from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


TokenPair = tuple[str, str]


def validate_special_tokens(special_tokens: list[str]) -> None:
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens must not contain duplicates")

    for token in special_tokens:
        if token == "":
            raise ValueError("special_tokens must not contain empty strings")


def split_text_into_initial_tokens(
    text: str,
    special_tokens: list[str],
) -> list[str]:
    tokens = []
    index = 0
    ordered_special_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

    while index < len(text):
        matched_special_token = None

        for special_token in ordered_special_tokens:
            if text.startswith(special_token, index):
                matched_special_token = special_token
                break

        if matched_special_token is not None:
            tokens.append(matched_special_token)
            index += len(matched_special_token)
        else:
            tokens.append(text[index])
            index += 1

    return tokens


def get_pair_counts(
    token_sequences: list[list[str]],
    special_tokens: list[str] | None = None,
) -> Counter[TokenPair]:
    protected_tokens = set(special_tokens or [])
    pair_counts: Counter[TokenPair] = Counter()

    for token_sequence in token_sequences:
        for left_token, right_token in zip(token_sequence, token_sequence[1:]):
            if left_token in protected_tokens or right_token in protected_tokens:
                continue

            pair_counts[(left_token, right_token)] += 1

    return pair_counts


def merge_token_pair(
    token_sequence: list[str],
    pair: TokenPair,
    merged_token: str,
) -> list[str]:
    merged_sequence = []
    index = 0

    while index < len(token_sequence):
        has_pair_at_index = (
            index < len(token_sequence) - 1
            and token_sequence[index] == pair[0]
            and token_sequence[index + 1] == pair[1]
        )

        if has_pair_at_index:
            merged_sequence.append(merged_token)
            index += 2
        else:
            merged_sequence.append(token_sequence[index])
            index += 1

    return merged_sequence


def apply_merges(
    token_sequence: list[str],
    merges: list[TokenPair],
) -> list[str]:
    for pair in merges:
        token_sequence = merge_token_pair(
            token_sequence=token_sequence,
            pair=pair,
            merged_token="".join(pair),
        )

    return token_sequence


@dataclass
class BytePairEncodingTokenizer:
    token_to_id: dict[str, int]
    merges: list[TokenPair]
    special_tokens: list[str]

    def __post_init__(self) -> None:
        validate_special_tokens(self.special_tokens)
        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }

        if len(self.id_to_token) != len(self.token_to_id):
            raise ValueError("token_to_id must not contain duplicate IDs")

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "unicode_bpe",
            "token_to_id": self.token_to_id,
            "merges": [
                list(pair)
                for pair in self.merges
            ],
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BytePairEncodingTokenizer":
        if payload.get("type") != "unicode_bpe":
            raise ValueError("tokenizer payload must have type unicode_bpe")

        token_to_id = payload["token_to_id"]
        merges = [
            (pair[0], pair[1])
            for pair in payload["merges"]
        ]
        special_tokens = payload["special_tokens"]

        return cls(
            token_to_id=token_to_id,
            merges=merges,
            special_tokens=special_tokens,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BytePairEncodingTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(payload, dict):
            raise ValueError("tokenizer file must contain a JSON object")

        return cls.from_dict(payload)

    def encode(self, text: str) -> list[int]:
        token_sequence = split_text_into_initial_tokens(
            text=text,
            special_tokens=self.special_tokens,
        )
        token_sequence = apply_merges(
            token_sequence=token_sequence,
            merges=self.merges,
        )

        return [
            self.token_to_id[token]
            for token in token_sequence
        ]

    def decode(self, token_ids: list[int]) -> str:
        tokens = [
            self.id_to_token[token_id]
            for token_id in token_ids
        ]

        return "".join(tokens)

    def tokenize_texts(self, texts: list[str]) -> list[list[int]]:
        return [
            self.encode(text)
            for text in texts
        ]


def build_base_vocabulary(
    texts: list[str],
    special_tokens: list[str],
) -> list[str]:
    token_set = set(special_tokens)

    for text in texts:
        token_set.update(
            split_text_into_initial_tokens(
                text=text,
                special_tokens=special_tokens,
            )
        )

    return [
        *special_tokens,
        *sorted(token for token in token_set if token not in special_tokens),
    ]


def choose_best_pair(pair_counts: Counter[TokenPair]) -> TokenPair | None:
    if not pair_counts:
        return None

    highest_count = max(pair_counts.values())
    best_pairs = [
        pair
        for pair, count in pair_counts.items()
        if count == highest_count
    ]

    return sorted(best_pairs)[0]


def train_bpe_tokenizer(
    texts: list[str],
    vocab_size: int,
    special_tokens: list[str] | None = None,
    base_texts: list[str] | None = None,
) -> BytePairEncodingTokenizer:
    if not texts:
        raise ValueError("texts must contain at least one training text")

    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than 0")

    special_tokens = special_tokens or []
    validate_special_tokens(special_tokens)

    vocabulary_tokens = build_base_vocabulary(
        texts=base_texts or texts,
        special_tokens=special_tokens,
    )

    if vocab_size < len(vocabulary_tokens):
        raise ValueError("vocab_size must be at least the base vocabulary size")

    token_sequences = [
        split_text_into_initial_tokens(
            text=text,
            special_tokens=special_tokens,
        )
        for text in texts
    ]
    merges: list[TokenPair] = []
    vocabulary_set = set(vocabulary_tokens)

    while len(vocabulary_tokens) < vocab_size:
        pair_counts = get_pair_counts(
            token_sequences=token_sequences,
            special_tokens=special_tokens,
        )
        best_pair = choose_best_pair(pair_counts)

        if best_pair is None:
            break

        merged_token = "".join(best_pair)

        if merged_token in special_tokens:
            break

        token_sequences = [
            merge_token_pair(
                token_sequence=token_sequence,
                pair=best_pair,
                merged_token=merged_token,
            )
            for token_sequence in token_sequences
        ]
        merges.append(best_pair)

        if merged_token not in vocabulary_set:
            vocabulary_tokens.append(merged_token)
            vocabulary_set.add(merged_token)

    token_to_id = {
        token: token_id
        for token_id, token in enumerate(vocabulary_tokens)
    }

    return BytePairEncodingTokenizer(
        token_to_id=token_to_id,
        merges=merges,
        special_tokens=special_tokens,
    )
