"""Deterministic transcript fingerprints for full-program deduplication.

The exact digest catches byte-independent equality after Unicode/token
normalization. MinHash catches lightly edited or re-cut copies without storing
whole transcripts in the shared delivery log. A near match additionally
requires comparable token counts, so excerpts remain distinct deliveries.
"""
from __future__ import annotations

import base64
import hashlib
import struct
import unicodedata
from dataclasses import dataclass
from typing import Literal

MINHASH_SIZE = 128
SHINGLE_SIZE = 4
CONTENT_WINDOW_DAYS = 30
MIN_EXACT_TOKENS = 50
MIN_NEAR_TOKENS = 500
NEAR_SIMILARITY = 0.60
MIN_LENGTH_RATIO = 0.90

_SERIALIZATION_PREFIX = "mh1:"
_HASH_PERSON = b"pd-mh-v1"
_PRIME = (1 << 61) - 1


def _permutation_constants() -> tuple[tuple[int, int], ...]:
    constants: list[tuple[int, int]] = []
    for index in range(MINHASH_SIZE):
        seed = hashlib.sha256(f"podcast-digest-minhash-v1:{index}".encode()).digest()
        multiplier = int.from_bytes(seed[:8], "big") % (_PRIME - 1) + 1
        offset = int.from_bytes(seed[8:16], "big") % _PRIME
        constants.append((multiplier, offset))
    return tuple(constants)


_PERMUTATIONS = _permutation_constants()


def _is_cjk_token(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= codepoint <= 0x9FFF  # CJK Unified Ideographs
        or 0xF900 <= codepoint <= 0xFAFF  # CJK Compatibility Ideographs
        or 0x20000 <= codepoint <= 0x2FA1F  # Supplementary CJK extensions
        or 0x3040 <= codepoint <= 0x30FF  # Hiragana + Katakana
        or 0xAC00 <= codepoint <= 0xD7AF  # Hangul syllables
    )


def transcript_tokens(text: str) -> list[str]:
    """Normalize transcript text into language-independent comparison tokens."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    word: list[str] = []

    def flush_word() -> None:
        if word:
            tokens.append("".join(word))
            word.clear()

    for character in normalized:
        if _is_cjk_token(character):
            flush_word()
            tokens.append(character)
        elif character.isalnum() or (
            unicodedata.category(character).startswith("M") and word
        ):
            word.append(character)
        else:
            flush_word()
    flush_word()
    return tokens


@dataclass(frozen=True)
class TranscriptFingerprint:
    sha256: str
    minhash: tuple[int, ...]
    token_count: int

    def serialize_minhash(self) -> str:
        if len(self.minhash) != MINHASH_SIZE:
            raise ValueError(f"expected {MINHASH_SIZE} MinHash values")
        payload = struct.pack(f">{MINHASH_SIZE}Q", *self.minhash)
        encoded = base64.urlsafe_b64encode(payload).decode("ascii")
        return _SERIALIZATION_PREFIX + encoded

    @classmethod
    def from_storage(
        cls, sha256: str, serialized_minhash: str, token_count: int,
    ) -> TranscriptFingerprint:
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError("invalid transcript SHA-256")
        if token_count < 0:
            raise ValueError("negative transcript token count")
        if not serialized_minhash.startswith(_SERIALIZATION_PREFIX):
            raise ValueError("unsupported transcript MinHash version")
        try:
            payload = base64.b64decode(
                serialized_minhash[len(_SERIALIZATION_PREFIX):],
                altchars=b"-_",
                validate=True,
            )
            values = struct.unpack(f">{MINHASH_SIZE}Q", payload)
        except (ValueError, struct.error) as exc:
            raise ValueError("invalid transcript MinHash") from exc
        return cls(sha256=sha256, minhash=values, token_count=token_count)


@dataclass(frozen=True)
class ContentCandidate:
    feed_url: str
    guid: str
    title: str
    emitted_at: str
    fingerprint: TranscriptFingerprint


@dataclass(frozen=True)
class ContentMatch:
    kind: Literal["exact", "near"]
    score: float
    candidate: ContentCandidate


def fingerprint_transcript(text: str) -> TranscriptFingerprint:
    tokens = transcript_tokens(text)
    normalized = "\x1f".join(tokens).encode("utf-8")
    exact_hash = hashlib.sha256(normalized).hexdigest()

    shingle_hashes: set[int] = set()
    if len(tokens) < SHINGLE_SIZE:
        if tokens:
            shingle_hashes.add(_stable_hash(normalized))
    else:
        for start in range(len(tokens) - SHINGLE_SIZE + 1):
            shingle = "\x1f".join(tokens[start:start + SHINGLE_SIZE]).encode("utf-8")
            shingle_hashes.add(_stable_hash(shingle))

    minima = [_PRIME] * MINHASH_SIZE
    for shingle_hash in shingle_hashes:
        for index, (multiplier, offset) in enumerate(_PERMUTATIONS):
            candidate = (multiplier * shingle_hash + offset) % _PRIME
            if candidate < minima[index]:
                minima[index] = candidate

    return TranscriptFingerprint(
        sha256=exact_hash,
        minhash=tuple(minima),
        token_count=len(tokens),
    )


def find_content_match(
    fingerprint: TranscriptFingerprint,
    candidates: list[ContentCandidate],
) -> ContentMatch | None:
    """Find the strongest safe duplicate among previously delivered content."""
    exact_match: ContentMatch | None = None
    near_match: ContentMatch | None = None

    for candidate in candidates:
        other = candidate.fingerprint
        if (
            fingerprint.token_count >= MIN_EXACT_TOKENS
            and other.token_count >= MIN_EXACT_TOKENS
            and fingerprint.sha256 == other.sha256
        ):
            exact_match = ContentMatch(
                kind="exact", score=1.0, candidate=candidate,
            )
            break

        if min(fingerprint.token_count, other.token_count) < MIN_NEAR_TOKENS:
            continue
        length_ratio = min(fingerprint.token_count, other.token_count) / max(
            fingerprint.token_count, other.token_count,
        )
        if length_ratio < MIN_LENGTH_RATIO:
            continue
        score = minhash_similarity(fingerprint, other)
        if score >= NEAR_SIMILARITY and (
            near_match is None or score > near_match.score
        ):
            near_match = ContentMatch(
                kind="near", score=score, candidate=candidate,
            )

    return exact_match or near_match


def minhash_similarity(
    left: TranscriptFingerprint, right: TranscriptFingerprint,
) -> float:
    if len(left.minhash) != MINHASH_SIZE or len(right.minhash) != MINHASH_SIZE:
        return 0.0
    equal = sum(a == b for a, b in zip(left.minhash, right.minhash, strict=True))
    return equal / MINHASH_SIZE


def _stable_hash(value: bytes) -> int:
    return int.from_bytes(
        hashlib.blake2b(value, digest_size=8, person=_HASH_PERSON).digest(),
        "big",
    ) % _PRIME
