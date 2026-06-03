"""Deterministic poem-level split assignment."""

from __future__ import annotations

import hashlib


def assign_split(poem_id: str, *, seed: int = 1337) -> str:
    """Assign a deterministic 80/10/10 split by poem ID."""

    digest = hashlib.sha256(f"{seed}:{poem_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"
