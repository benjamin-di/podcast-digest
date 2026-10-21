"""Unit tests for the shared delivery log (emitted.py)."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from podcast_digest.dedupe import fingerprint_transcript
from podcast_digest.emitted import EmittedLog


def test_disabled_when_path_empty(tmp_path):
    for empty in ("", None):
        log = EmittedLog.load(empty)
        assert log.enabled is False
        assert log.contains("feed", "guid") is False
        # record is a no-op and writes nothing
        log.record("feed", "guid", "2026-06-28T00:00:00+00:00")
        assert log.contains("feed", "guid") is False
    # nothing was created in the tmp dir
    assert list(tmp_path.iterdir()) == []


def test_missing_file_is_enabled_but_empty(tmp_path):
    p = tmp_path / "sub" / "emitted.tsv"
    log = EmittedLog.load(p)
    assert log.enabled is True
    assert log.contains("feed", "guid") is False


def test_record_then_contains_and_persists(tmp_path):
    p = tmp_path / "emitted.tsv"
    log = EmittedLog.load(p)
    log.record("https://f/feed.xml", "ep-1", "2026-06-28T01:00:00+00:00")
    assert log.contains("https://f/feed.xml", "ep-1") is True
    # written to disk and re-readable by a fresh loader (another instance's view)
    reloaded = EmittedLog.load(p)
    assert reloaded.contains("https://f/feed.xml", "ep-1") is True
    assert p.read_text().count("\n") == 1


def test_record_is_idempotent(tmp_path):
    p = tmp_path / "emitted.tsv"
    log = EmittedLog.load(p)
    log.record("f", "g", "t1")
    log.record("f", "g", "t2")  # same key — no second line
    assert p.read_text().count("\n") == 1


def test_load_tolerates_garbage_and_dedups(tmp_path):
    p = tmp_path / "emitted.tsv"
    # blank line, a short (single-column) line, a dup, and two real records —
    # mirrors a git-merged concatenation of two instances' appends.
    p.write_text(
        "\n"
        "incomplete-line-no-tab\n"
        "f1\tg1\t2026-01-01T00:00:00+00:00\n"
        "f1\tg1\t2026-01-02T00:00:00+00:00\n"   # duplicate key, later ts
        "f2\tg2\t2026-01-03T00:00:00+00:00\n",
        encoding="utf-8",
    )
    log = EmittedLog.load(p)
    assert log.contains("f1", "g1") is True
    assert log.contains("f2", "g2") is True
    assert log.contains("f9", "g9") is False
    # union-on-read: the duplicate folds into a single key
    assert log._guid_keys == {("f1", "g1"), ("f2", "g2")}


def test_append_accumulates(tmp_path):
    p = tmp_path / "emitted.tsv"
    log = EmittedLog.load(p)
    log.record("f", "a", "t")
    log.record("f", "b", "t")
    # 5-column rows now (show/title empty when not supplied)
    assert p.read_text().splitlines() == [
        "f\ta\tt\t\t",
        "f\tb\tt\t\t",
    ]


def test_content_fingerprint_persists_and_respects_cutoff(tmp_path):
    path = tmp_path / "emitted.tsv"
    fingerprint = fingerprint_transcript(" ".join(f"token{i}" for i in range(700)))
    EmittedLog.load(path).record(
        "feed",
        "guid",
        "2026-07-29T01:00:00+00:00",
        "Show",
        "Episode",
        fingerprint=fingerprint,
    )

    reloaded = EmittedLog.load(path)
    candidates = reloaded.content_candidates(
        datetime(2026, 7, 1, tzinfo=UTC)
    )
    assert len(candidates) == 1
    assert candidates[0].fingerprint == fingerprint
    assert len(reloaded.content_candidates(datetime(2026, 7, 1))) == 1
    assert reloaded.content_candidates(datetime(2026, 8, 1, tzinfo=UTC)) == []
    assert len(path.read_text().split("\t")) == 8


def test_existing_identity_can_be_enriched_without_duplicate_content(tmp_path):
    path = tmp_path / "emitted.tsv"
    log = EmittedLog.load(path)
    log.record("feed", "guid", "2026-07-29T01:00:00+00:00", "Show", "Episode")
    fingerprint = fingerprint_transcript(" ".join(f"token{i}" for i in range(700)))

    log.record(
        "feed",
        "guid",
        "2026-07-29T01:00:00+00:00",
        "Show",
        "Episode",
        fingerprint=fingerprint,
    )
    log.record(
        "feed",
        "guid",
        "2026-07-29T01:00:00+00:00",
        "Show",
        "Episode",
        fingerprint=fingerprint,
    )

    assert len(path.read_text().splitlines()) == 2
    assert len(log.content_candidates(datetime(2026, 7, 1, tzinfo=UTC))) == 1


# --- secondary (show, title) key + mailbox backfill -------------------------


def test_title_key_dedups_without_guid(tmp_path):
    p = tmp_path / "emitted.tsv"
    log = EmittedLog.load(p)
    # backfill a title-only entry (as a mailbox backfill would)
    log.record_title("Market Hours", "A Strategist on Bubbles (59m)")
    # a later sighting with a DIFFERENT feed_url/guid but same show+title is skipped
    assert log.contains("https://feeds/market-hours-mirror.xml", "guid-xyz",
                        "Market Hours", "A Strategist on Bubbles") is True


def test_title_norm_tolerates_case_punct_and_duration(tmp_path):
    p = tmp_path / "emitted.tsv"
    log = EmittedLog.load(p)
    log.record_title("Tech & Markets Pod", "The Chip IPO, Model Launches & Market Check")
    # casing/punctuation differences + a duration suffix still match
    assert log.contains(
        "f", "g", "tech  markets pod", "the chip ipo model launches  market check (1h20m)",
    ) is True


def test_title_norm_keeps_unicode_scripts_distinct(tmp_path):
    """A Chinese mailbox record must not collapse unrelated CJK titles."""
    log = EmittedLog.load(tmp_path / "emitted.tsv")
    log.record_title("左右之间", "李明：城市往事")

    assert log.contains("f", "g1", "左右之间", "李明 城市往事") is True
    assert log.contains("f", "g2", "声东击西", "人工智能的未来") is False


def test_year_and_number_suffixes_do_not_collide(tmp_path):
    """F1 regression: the duration-stripper must NOT treat a trailing year or
    sequence number as a duration. "Year in Review (2024)" and "(2025)" are
    DISTINCT episodes and must not dedup each other; likewise "(1)"/"(2)"."""
    log = EmittedLog.load(tmp_path / "emitted.tsv")
    log.record_title("The Show", "Year in Review (2024)")
    # The 2025 recap has a distinct normalized title → NOT already emitted.
    assert log.contains("f", "g-2025", "The Show", "Year in Review (2025)") is False
    log.record_title("The Show", "Deep Dive (1)")
    assert log.contains("f", "g-dd2", "The Show", "Deep Dive (2)") is False
    # But a real duration suffix is still stripped (it's the same episode).
    log.record_title("The Show", "The Big One")
    assert log.contains("f", "g", "The Show", "The Big One (1h20m)") is True
    assert log.contains("f", "g", "The Show", "The Big One (44m)") is True


def test_record_writes_both_keys_and_reloads(tmp_path):
    p = tmp_path / "emitted.tsv"
    EmittedLog.load(p).record("https://f/x.xml", "ep-9", "2026-06-28T00:00:00+00:00",
                              "Company Histories", "Some Episode")
    reloaded = EmittedLog.load(p)
    assert reloaded.contains("https://f/x.xml", "ep-9") is True          # exact key
    # Show-scoped guid match: the OTHER instance resolved the show to a different
    # feed URL, but (show, guid) both come from the feed itself → recognized.
    assert reloaded.contains("https://other-instance.example/feed.xml", "ep-9", "Company Histories") is True
    # Scoped by show on purpose: guids are only unique per feed, so another
    # show reusing the same guid STRING must not be suppressed.
    assert reloaded.contains("https://other-show/feed.xml", "ep-9", "Market Hours") is False
    # show/title are persisted in the record (human-readable, backfill dedup)…
    assert "Company Histories\tSome Episode" in p.read_text()
    # …but a GUID'd record must NOT gate future episodes by title (F1): a show
    # reusing a title for a NEW episode with a distinct guid is not a duplicate.
    assert reloaded.contains("https://f/x.xml", "ep-10",
                             "Company Histories", "Some Episode") is False
    # has_title (backfill idempotency) DOES see GUID'd records' titles — the
    # mailbox backfill must skip these rather than re-append them as GUID-less
    # title suppressors (which would re-arm the F1 bug it exists to avoid).
    assert reloaded.has_title("Company Histories", "Some Episode") is True
    assert reloaded.has_title("Company Histories", "Different Episode") is False


@pytest.mark.parametrize(
    "guid",
    [
        "5f0c1e2d3b4a596877665544",
        "550e8400-e29b-41d4-a716-446655440000",
        "0198c9f4-d72d-7d73-b0f8-cd3a17c146ea",
        "0198c9f4-d72d-8d73-b0f8-cd3a17c146ea",
        "https://publisher.example/episodes/42#player",
        "yt:video:AbCdEf12345",
    ],
)
def test_strong_guid_dedups_across_feed_and_show(tmp_path, guid):
    """Syndicated content IDs stay global across differently named feeds."""
    path = tmp_path / "emitted.tsv"
    EmittedLog.load(path).record(
        "https://publisher.example/umbrella.xml",
        guid,
        "2026-07-29T00:00:00Z",
        "World News Daily",
        "Episode",
    )

    assert EmittedLog.load(path).contains(
        "https://publisher.example/show.xml",
        guid,
        "World News Daily Briefing",
        "Episode",
    ) is True


@pytest.mark.parametrize(
    "guid",
    ["1", "episode-7", "123456789012345678901234", "https://["],
)
def test_weak_guid_remains_show_scoped(tmp_path, guid):
    path = tmp_path / "emitted.tsv"
    EmittedLog.load(path).record("feed-a", guid, "t", "Show A", "Episode")

    assert EmittedLog.load(path).contains(
        "feed-b", guid, "Show B", "Different episode"
    ) is False


def test_record_title_disabled_is_noop(tmp_path):
    log = EmittedLog.load("")
    log.record_title("Show", "Title")
    assert log.contains("f", "g", "Show", "Title") is False
