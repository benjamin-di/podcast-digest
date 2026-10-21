"""Transcript fingerprint and full-program matching policy."""
from __future__ import annotations

import pytest

from podcast_digest.dedupe import (
    ContentCandidate,
    TranscriptFingerprint,
    find_content_match,
    fingerprint_transcript,
    transcript_tokens,
)


def _candidate(text: str) -> ContentCandidate:
    return ContentCandidate(
        feed_url="https://feeds.example/original",
        guid="original-guid",
        title="Original",
        emitted_at="2026-07-29T00:00:00+00:00",
        fingerprint=fingerprint_transcript(text),
    )


def test_tokenization_is_unicode_normalized_and_cjk_aware():
    assert transcript_tokens("ＡＩ Café") == ["ai", "café"]
    assert transcript_tokens("人工智能の未来") == ["人", "工", "智", "能", "の", "未", "来"]
    assert transcript_tokens("𠀀𠀁") == ["𠀀", "𠀁"]


def test_fingerprint_storage_round_trip():
    fingerprint = fingerprint_transcript(" ".join(f"token{i}" for i in range(700)))

    restored = TranscriptFingerprint.from_storage(
        fingerprint.sha256,
        fingerprint.serialize_minhash(),
        fingerprint.token_count,
    )

    assert restored == fingerprint


@pytest.mark.parametrize(
    ("sha256", "minhash", "token_count"),
    [
        ("bad", "mh1:bad", 10),
        ("0" * 64, "future:value", 10),
        ("0" * 64, "mh1:bad", -1),
    ],
)
def test_fingerprint_storage_rejects_malformed_values(sha256, minhash, token_count):
    with pytest.raises(ValueError):
        TranscriptFingerprint.from_storage(sha256, minhash, token_count)


def test_exact_full_transcript_matches():
    text = " ".join(f"token{i}" for i in range(700))

    match = find_content_match(fingerprint_transcript(text), [_candidate(text)])

    assert match is not None
    assert match.kind == "exact"
    assert match.score == 1.0


def test_lightly_edited_full_transcript_matches():
    original_tokens = [f"token{i}" for i in range(1000)]
    edited_tokens = original_tokens.copy()
    edited_tokens[400:450] = [f"replacement{i}" for i in range(50)]

    match = find_content_match(
        fingerprint_transcript(" ".join(edited_tokens)),
        [_candidate(" ".join(original_tokens))],
    )

    assert match is not None
    assert match.kind == "near"
    assert match.score >= 0.60


def test_excerpt_is_not_suppressed_even_with_high_containment():
    original_tokens = [f"token{i}" for i in range(1000)]
    excerpt = original_tokens[200:700]

    match = find_content_match(
        fingerprint_transcript(" ".join(excerpt)),
        [_candidate(" ".join(original_tokens))],
    )

    assert match is None


def test_unrelated_equal_length_transcript_is_not_suppressed():
    first = " ".join(f"alpha{i}" for i in range(1000))
    second = " ".join(f"beta{i}" for i in range(1000))

    assert find_content_match(
        fingerprint_transcript(second), [_candidate(first)]
    ) is None
