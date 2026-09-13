"""Build and query an Italian rhyme lexicon from the V8 train sonnets.

The lexicon records the checker rhyme key of every line-final word, its
frequency, and the most frequent words per key. It supports plan construction
for the plan-then-poem pilot.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from sonnet_evaluation.sonnet_prosody import rhyme_key, soft_rhyme_key
from sonnet_training.form_targeted_data import load_train_rows, read_sonnet_text

LEXICON_VERSION = "sonnet_rhyme_lexicon_v1"
WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÿ']+")
Progress = Callable[[str], None]


def line_final_word(line: str) -> str | None:
    words = WORD_PATTERN.findall(line)
    return words[-1].lower() if words else None


def sonnet_endings(text: str) -> list[str]:
    endings = []
    for line in text.splitlines():
        if not line.strip():
            continue
        word = line_final_word(line)
        if word:
            endings.append(word)
    return endings


def build_lexicon(
    rows: Iterable[Mapping[str, str]],
    root: Path,
    *,
    manifest_path: Path,
    progress: Progress | None = None,
) -> dict[str, Any]:
    exact: Counter[str] = Counter()
    words: dict[str, Counter[str]] = defaultdict(Counter)
    sonnets = 0
    lines = 0
    row_list = list(rows)
    for index, row in enumerate(row_list, start=1):
        text = read_sonnet_text(root, row)
        endings = sonnet_endings(text)
        if not endings:
            continue
        sonnets += 1
        lines += len(endings)
        for word in endings:
            key = rhyme_key(word)
            if not key:
                continue
            exact[key] += 1
            words[key][word] += 1
        if progress and index % 4000 == 0:
            progress(f"scanned {index}/{len(row_list)} sonnets")
    entries = []
    for key, count in sorted(exact.items(), key=lambda item: (-item[1], item[0])):
        entries.append(
            {
                "key": key,
                "soft_key": soft_rhyme_key(key),
                "count": count,
                "words": [word for word, _ in words[key].most_common(10)],
            }
        )
    return {
        "lexicon_version": LEXICON_VERSION,
        "source_manifest": str(manifest_path),
        "source_split": "train",
        "sonnet_count": sonnets,
        "line_count": lines,
        "unique_keys": len(exact),
        "entries": entries,
    }


def write_lexicon(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_lexicon(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("lexicon_version") != LEXICON_VERSION:
        raise ValueError("rhyme lexicon version mismatch")
    if not payload.get("entries"):
        raise ValueError("rhyme lexicon is empty")
    return payload


def choose_endings(
    scheme: str,
    lexicon: Mapping[str, Any],
    *,
    seed: int,
    min_candidates: int = 5,
    max_keys_per_class: int = 20,
) -> list[dict[str, Any]]:
    if not scheme or len(scheme) != 14:
        raise ValueError("rhyme scheme must contain 14 letters")
    entries = [
        entry
        for entry in lexicon["entries"]
        if entry["count"] >= min_candidates and len(entry["words"]) >= 5
    ][: max_keys_per_class * 8]
    if len(entries) < min_candidates:
        raise ValueError("rhyme lexicon has too few candidate keys")
    rng = random.Random(seed)
    order = list(entries)
    rng.shuffle(order)
    classes: dict[str, dict[str, Any]] = {}
    used_words: set[str] = set()
    used_keys: set[str] = set()
    for letter in sorted(set(scheme)):
        chosen = None
        for entry in order:
            if entry["key"] in used_keys:
                continue
            word_pool = [word for word in entry["words"] if word not in used_words]
            if len(word_pool) < 2:
                continue
            word = rng.choice(word_pool[1:])
            chosen = {"key": entry["key"], "word": word}
            break
        if chosen is None:
            raise ValueError(f"no lexicon candidates for class {letter}")
        used_keys.add(chosen["key"])
        used_words.add(chosen["word"])
        classes[letter] = chosen
    return [
        {
            "line": index + 1,
            "class": letter,
            "key": classes[letter]["key"],
            "word": classes[letter]["word"],
        }
        for index, letter in enumerate(scheme)
    ]
