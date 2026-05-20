"""Content-hashing primitives shared across ``ef``'s layers.

Content addressing is the backbone of ``ef``'s identity and change-detection
model: a segment's id (L2, :func:`ef.segments.segment_id`), a source's content
hash (L0, :func:`ef.corpus.content_hash`) and — from the ``ArtifactGraph`` on —
every artifact id are all ``SHA-256`` over *normalized* content.

Normalizing *before* hashing is the point. Cosmetically-different encodings of
the same content — NFD vs NFC, a stray BOM, CRLF vs LF line endings, JSON with
keys in a different order — must hash *identically*, so that re-ingesting
unchanged data is a no-op rather than a spurious "change" that re-runs the
expensive downstream pipeline. This module is the **single source of truth**
for that normalization; every other ``ef`` module hashes through it.

>>> sha256_hex(normalize_text('a\\r\\nb')) == sha256_hex(normalize_text('a\\nb'))
True
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any

__all__ = ["normalize_text", "canonical_json", "sha256_hex"]

#: The Unicode byte-order-mark codepoint, stripped by :func:`normalize_text`.
_BOM = "﻿"


def normalize_text(text: str) -> str:
    """Normalize ``text`` for hashing: Unicode NFC, no BOM, ``\\n`` line endings.

    The normalization makes hashing robust to differences that carry no
    meaning: a composed vs decomposed accent, a leading byte-order mark, or
    Windows vs Unix line endings. Only the *hash input* is normalized — callers
    keep the original ``text`` verbatim.

    >>> normalize_text('a\\r\\nb\\rc') == 'a\\nb\\nc'
    True
    >>> normalize_text('\\ufeffhi') == 'hi'
    True
    >>> normalize_text('café') == normalize_text('cafe\\u0301')  # NFC vs NFD
    True
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_BOM, "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to deterministic JSON: sorted keys, compact separators.

    Two mappings that are equal but were built in a different key order produce
    the *same* string — so they hash identically. Non-JSON values fall back to
    their ``str`` form, so the function never raises on exotic input (at the
    cost of those values not round-tripping).

    >>> canonical_json({'b': 1, 'a': 2})
    '{"a":2,"b":1}'
    >>> canonical_json({'a': 2, 'b': 1}) == canonical_json({'b': 1, 'a': 2})
    True
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def sha256_hex(payload: str | bytes) -> str:
    """Return the ``SHA-256`` hex digest of ``payload`` (``str`` is UTF-8 encoded).

    >>> sha256_hex('')
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    >>> sha256_hex('abc') == sha256_hex(b'abc')
    True
    """
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
