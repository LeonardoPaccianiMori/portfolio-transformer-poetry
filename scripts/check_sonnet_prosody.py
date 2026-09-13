#!/usr/bin/env python3
"""Check Italian sonnet prosody with the rule-based sonnet_prosody checker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.sonnet_prosody import analyse_sonnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--poem", type=Path, help="file with one sonnet")
    source.add_argument("--text", type=str, help="sonnet text on the command line")
    parser.add_argument(
        "--scheme",
        type=str,
        default=None,
        help="expected rhyme scheme, for example ABBAABBACDECDE",
    )
    parser.add_argument(
        "--stress-lexicon",
        type=Path,
        default=None,
        help="JSON object mapping a word to its stress from the end (1, 2, 3)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the complete result as JSON to this path",
    )
    return parser.parse_args()


def load_stress_lexicon(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stress lexicon must be a JSON object")
    return {str(word).lower(): int(value) for word, value in payload.items()}


def main() -> None:
    args = parse_args()
    text = args.poem.read_text(encoding="utf-8") if args.poem else args.text
    result = analyse_sonnet(
        text,
        expected_scheme=args.scheme,
        stress_overrides=load_stress_lexicon(args.stress_lexicon),
    )

    print(f"lines: {result['line_count']}")
    print(f"structure_ok: {result['structure_ok']}")
    print(f"stanza_pattern: {result['stanza_pattern']}")
    print(f"metre_valid_lines: {result['metre_valid_lines']}")
    print(f"metre_uncertain_lines: {result['metre_uncertain_lines']}")
    print(f"metre_failed_lines: {result['metre_failed_lines']}")
    print(f"metre_ok: {result['metre_ok']}")
    print(f"rhyme_scheme: {result['rhyme_scheme']}")
    if result["expected_scheme"] is not None:
        print(f"expected_scheme: {result['expected_scheme']}")
        print(f"scheme_ok: {result['scheme_ok']}")
    print(f"perfect_rhyme_pairs: {result['perfect_rhyme_pairs']}")
    print(f"soft_rhyme_pairs: {result['soft_rhyme_pairs']}")
    print(f"rhyme_score: {result['rhyme_score']:.4f}")
    print(f"hard_gate_pass: {result['hard_gate_pass']}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote json: {args.json}")


if __name__ == "__main__":
    main()
