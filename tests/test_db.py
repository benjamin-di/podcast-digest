import sqlite3
from pathlib import Path

from podcast_digest.db import (
    MAX_ATTEMPTS,
    SUMMARIZE_TRANSPORT_MAX_ATTEMPTS,
    WIKI_PUSH_MAX_ATTEMPTS,
    Database,
    stage_lt,
)
from podcast_digest.dedupe import fingerprint_transcript

# The episodes schema as it was *before* the clean_attempts column was added —
# used to exercise the idempotent ADD COLUMN migration against an existing DB.
_OLD_EPISODES_SCHEMA = """
CREATE TABLE episodes (
    feed_url TEXT NOT NULL, guid TEXT NOT NULL, title TEXT NOT NULL,
    published_at TEXT NOT NULL, duration_seconds INTEGER, audio_url TEXT NOT NULL,
    audio_path TEXT, transcript_path TEXT, segments_path TEXT, summary_html TEXT,
    status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT, failed_at TEXT, emailed_at TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (feed_url, guid)
);
"""


def test_stage_lt_orders_pipeline_stages():
    assert stage_lt("discovered", "downloaded")
    assert stage_lt("downloaded", "summarized")
    assert not stage_lt("emailed", "downloaded")
    assert not stage_lt("summarized", "summarized")


def test_stage_lt_treats_none_as_pre_pipeline():
    # `None` is only emitted before any stage has run (newly-inserted row
    # would normally have 'discovered' but be defensive).
    assert stage_lt(None, "downloaded")
    assert stage_lt(None, "emailed")


def test_feed_and_episode_roundtrip(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    try:
        # First upsert is "first run" (last_polled_at is NULL).
        assert db.upsert_feed(url="https://x/rss", name="X") is True
        # Still first-run-flagged until we mark it polled.
        assert db.upsert_feed(url="https://x/rss", name="X") is True
        db.mark_feed_polled("https://x/rss")
        # Now it's a known feed.
        assert db.upsert_feed(url="https://x/rss", name="X") is False

        assert not db.has_episode("https://x/rss", "guid-1")

        db.insert_episode(
            feed_url="https://x/rss",
            guid="guid-1",
            title="Episode One",
            published_at="2024-04-02T10:00:00+00:00",
            duration_seconds=5400,
            audio_url="https://cdn/ep.mp3",
        )

        assert db.has_episode("https://x/rss", "guid-1")
        ep = db.get_episode("https://x/rss", "guid-1")
        assert ep is not None
        assert ep.status == "discovered"
        assert ep.title == "Episode One"
        assert ep.content_sha256 is None

        db.set_status(
            feed_url="https://x/rss", guid="guid-1",
            status="downloaded", audio_path="/tmp/ep.mp3",
        )
        ep2 = db.get_episode("https://x/rss", "guid-1")
        assert ep2 is not None
        assert ep2.status == "downloaded"
        assert ep2.audio_path == "/tmp/ep.mp3"

        # mark_failed preserves the high-water-mark status so the next tick
        # resumes from the right place; it only updates failure metadata.
        db.mark_failed("https://x/rss", "guid-1", "boom")
        ep3 = db.get_episode("https://x/rss", "guid-1")
        assert ep3 is not None
        assert ep3.status == "downloaded"        # NOT clobbered to 'failed'
        assert ep3.last_error == "boom"
        assert ep3.failed_at is not None
        assert ep3.attempts == 1

        # Successful stage completion clears the error.
        db.set_status(
            feed_url="https://x/rss", guid="guid-1",
            status="transcribed",
            transcript_path="/tmp/t.txt", segments_path="/tmp/t.json",
        )
        ep4 = db.get_episode("https://x/rss", "guid-1")
        assert ep4 is not None
        assert ep4.status == "transcribed"
        assert ep4.last_error is None
    finally:
        db.close()


def test_episodes_in_progress_caps_at_max_attempts(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://x/rss", name="X")
        db.insert_episode(
            feed_url="https://x/rss", guid="a", title="A",
            published_at="2024-04-02T10:00:00+00:00",
            duration_seconds=60, audio_url="https://x/a.mp3",
        )
        # Burn through the retry budget.
        for _ in range(MAX_ATTEMPTS):
            db.mark_failed("https://x/rss", "a", "fail")
        # No longer pending — would otherwise spam retries forever.
        assert db.episodes_in_progress("https://x/rss") == []
    finally:
        db.close()


def test_set_status_resets_attempts_on_success(tmp_path: Path):
    """F2: `attempts` is a *consecutive*-failure budget — a successful stage
    transition must clear it, so blips spread across an episode's life don't
    accumulate to the drop cap."""
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://x/rss", name="X")
        db.insert_episode(
            feed_url="https://x/rss", guid="a", title="A",
            published_at="2024-04-02T10:00:00+00:00",
            duration_seconds=60, audio_url="https://x/a.mp3",
        )
        for _ in range(MAX_ATTEMPTS - 1):
            db.mark_failed("https://x/rss", "a", "blip")
        assert db.get_episode("https://x/rss", "a").attempts == MAX_ATTEMPTS - 1
        # A stage succeeds → the counter resets.
        db.set_status(feed_url="https://x/rss", guid="a", status="downloaded")
        assert db.get_episode("https://x/rss", "a").attempts == 0
        # Still pending (would have been dropped had attempts not reset).
        assert [p.guid for p in db.episodes_in_progress("https://x/rss")] == ["a"]
    finally:
        db.close()


def test_mark_failed_transport_uses_separate_generous_budget(tmp_path: Path):
    """F2/R1: a summarizer-transport outage (VpsUnavailable) bumps the separate,
    generous `summarize_attempts` counter — NOT the small `attempts` one — so a
    multi-day outage doesn't dark-hole episodes via MAX_ATTEMPTS. But it's still
    bounded, so a deterministically-failing episode eventually drops."""
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://x/rss", name="X")
        db.insert_episode(
            feed_url="https://x/rss", guid="a", title="A",
            published_at="2024-04-02T10:00:00+00:00",
            duration_seconds=60, audio_url="https://x/a.mp3",
        )
        # A multi-day outage: far more transport failures than MAX_ATTEMPTS, but
        # under the transport cap → still pending (the small budget is untouched).
        for _ in range(SUMMARIZE_TRANSPORT_MAX_ATTEMPTS - 1):
            db.mark_failed("https://x/rss", "a", "summarizer down", transport=True)
        ep = db.get_episode("https://x/rss", "a")
        assert ep.attempts == 0                                   # general budget untouched
        assert ep.summarize_attempts == SUMMARIZE_TRANSPORT_MAX_ATTEMPTS - 1
        assert ep.last_error == "summarizer down"
        assert [p.guid for p in db.episodes_in_progress("https://x/rss")] == ["a"]
        # One more transport failure crosses the transport threshold → NOT
        # dropped (that would dark-hole the whole backlog during a long outage
        # and re-arm the heartbeat): the episode stays pending as the canary.
        db.mark_failed("https://x/rss", "a", "summarizer down", transport=True)
        assert [p.guid for p in db.episodes_in_progress("https://x/rss")] == ["a"]
        # The transport budget is independent of the general one.
        assert db.get_episode("https://x/rss", "a").attempts == 0
    finally:
        db.close()


def test_transport_threshold_throttles_to_single_canary(tmp_path: Path):
    """Past the transport threshold, capped episodes are throttled — exactly ONE
    rides along per tick as the canary — and a recovered transport drains the
    backlog one canary-success at a time: each success clears its OWN counters
    via set_status, which re-admits it. Deliberately no bulk reset (it would
    zero a stuck episode's hard-cap progress whenever its feed publishes)."""
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://x/rss", name="X")
        for guid, ts in (("old", "2024-04-01T10:00:00+00:00"),
                         ("mid", "2024-04-02T10:00:00+00:00"),
                         ("new", "2024-04-03T10:00:00+00:00")):
            db.insert_episode(
                feed_url="https://x/rss", guid=guid, title=guid.upper(),
                published_at=ts, duration_seconds=60,
                audio_url=f"https://x/{guid}.mp3",
            )
        # Outage: every episode crosses the threshold.
        for guid in ("old", "mid", "new"):
            for _ in range(SUMMARIZE_TRANSPORT_MAX_ATTEMPTS):
                db.mark_failed("https://x/rss", guid, "summarizer down", transport=True)
        # Only the newest capped episode rides as the canary — the expensive
        # LLM work is not re-run for the whole backlog every tick.
        assert [p.guid for p in db.episodes_in_progress("https://x/rss")] == ["new"]
        # An uncapped episode takes precedence in the normal sweep; the canary
        # still tags along (full list = uncapped ++ one canary).
        db.insert_episode(
            feed_url="https://x/rss", guid="fresh",
            title="FRESH", published_at="2024-04-04T10:00:00+00:00",
            duration_seconds=60, audio_url="https://x/fresh.mp3",
        )
        assert [p.guid for p in db.episodes_in_progress("https://x/rss")] == ["fresh", "new"]
        # Recovery: the canary succeeds → set_status clears ITS counters and
        # it rejoins normal flow; the canary slot rotates to the remaining
        # throttled episodes — the backlog drains one per tick, and a stuck
        # episode's hard-cap progress is never zeroed by siblings' successes.
        db.set_status(feed_url="https://x/rss", guid="new", status="summarized")
        pending = [p.guid for p in db.episodes_in_progress("https://x/rss")]
        assert "new" in pending and "fresh" in pending
        assert pending.count("mid") + pending.count("old") == 1  # next canary
    finally:
        db.close()


def test_set_status_resets_both_failure_budgets(tmp_path: Path):
    """R1: a successful stage clears BOTH the general and transport counters, so
    an episode recovers the moment the summarizer comes back."""
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://x/rss", name="X")
        db.insert_episode(
            feed_url="https://x/rss", guid="a", title="A",
            published_at="2024-04-02T10:00:00+00:00",
            duration_seconds=60, audio_url="https://x/a.mp3",
        )
        db.mark_failed("https://x/rss", "a", "blip")                 # attempts=1
        db.mark_failed("https://x/rss", "a", "summarizer down", transport=True)  # summarize=1
        db.set_status(feed_url="https://x/rss", guid="a", status="transcribed")
        ep = db.get_episode("https://x/rss", "a")
        assert ep.attempts == 0 and ep.summarize_attempts == 0
    finally:
        db.close()


def test_episodes_in_progress_excludes_emailed(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://x/rss", name="X")
        db.insert_episode(
            feed_url="https://x/rss", guid="a", title="A",
            published_at="2024-04-02T10:00:00+00:00",
            duration_seconds=60, audio_url="https://x/a.mp3",
        )
        db.insert_episode(
            feed_url="https://x/rss", guid="b", title="B",
            published_at="2024-04-09T10:00:00+00:00",
            duration_seconds=60, audio_url="https://x/b.mp3",
        )
        db.set_status(
            feed_url="https://x/rss", guid="b",
            status="emailed", emailed_at="2024-04-09T11:00:00+00:00",
        )
        pending = db.episodes_in_progress("https://x/rss")
        assert [p.guid for p in pending] == ["a"]
    finally:
        db.close()


def test_migration_adds_clean_attempts_to_existing_db(tmp_path: Path):
    path = tmp_path / "state.db"
    # Seed a DB with the pre-migration schema + one row.
    con = sqlite3.connect(path)
    con.executescript(_OLD_EPISODES_SCHEMA)
    con.execute(
        "INSERT INTO episodes (feed_url, guid, title, published_at, audio_url, "
        "status, attempts, created_at, updated_at) "
        "VALUES ('f','g','T','2024','u','transcribed',0,'2024','2024')"
    )
    con.commit()
    con.close()

    # Opening it runs _migrate(), which must add clean_attempts (default 0).
    db = Database(path)
    try:
        ep = db.get_episode("f", "g")
        assert ep is not None
        assert ep.clean_attempts == 0
        db.bump_clean_attempt("f", "g")
        db.bump_clean_attempt("f", "g")
        assert db.get_episode("f", "g").clean_attempts == 2
        # attempts is untouched by clean bumps (separate budgets).
        assert db.get_episode("f", "g").attempts == 0
        assert db.get_episode("f", "g").content_sha256 is None
    finally:
        db.close()


def test_content_fingerprint_roundtrip_and_delivery_filter(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://x/rss", name="X")
        fingerprint = fingerprint_transcript(
            " ".join(f"token{i}" for i in range(700))
        )
        for guid, summary in (("sent", "<p>summary</p>"), ("skipped", None)):
            db.insert_episode(
                feed_url="https://x/rss",
                guid=guid,
                title=guid.title(),
                published_at="2026-07-29T00:00:00+00:00",
                duration_seconds=3600,
                audio_url=f"https://x/{guid}.mp3",
            )
            db.set_content_fingerprint("https://x/rss", guid, fingerprint)
            db.set_status(
                feed_url="https://x/rss",
                guid=guid,
                status="emailed",
                summary_html=summary,
                emailed_at="2026-07-29T01:00:00+00:00",
            )

        stored = db.get_episode("https://x/rss", "sent")
        assert stored is not None
        assert stored.content_sha256 == fingerprint.sha256
        assert stored.content_minhash == fingerprint.serialize_minhash()
        assert stored.content_token_count == fingerprint.token_count

        candidates = db.delivered_content_since("2026-07-01T00:00:00+00:00")
        assert [(candidate.guid, candidate.fingerprint) for candidate in candidates] == [
            ("sent", fingerprint)
        ]
        assert db.delivered_content_since("2026-08-01T00:00:00+00:00") == []
    finally:
        db.close()


def test_migration_adds_wiki_columns_and_is_forward_only(tmp_path: Path):
    """The pre-wiki schema must gain both columns on open. Forward-only guard:
    rows ALREADY emailed are backfilled as pushed (so enabling the feature does
    not flood the wiki with history); rows not yet emailed stay eligible."""
    path = tmp_path / "state.db"
    con = sqlite3.connect(path)
    con.executescript(_OLD_EPISODES_SCHEMA)  # lacks the wiki columns entirely
    # An already-emailed (historical) row + a not-yet-emailed row.
    con.execute(
        "INSERT INTO episodes (feed_url, guid, title, published_at, audio_url, "
        "status, attempts, emailed_at, created_at, updated_at) "
        "VALUES ('f','old','T','2024','u','emailed',0,'2024','2024','2024')"
    )
    con.execute(
        "INSERT INTO episodes (feed_url, guid, title, published_at, audio_url, "
        "status, attempts, created_at, updated_at) "
        "VALUES ('f','cur','T','2024','u','transcribed',0,'2024','2024')"
    )
    con.commit()
    con.close()

    db = Database(path)
    try:
        # Historical emailed row: backfilled as pushed → never retro-flooded.
        old = db.get_episode("f", "old")
        assert old is not None
        assert old.wiki_pushed_at is not None
        assert old.wiki_push_attempts == 0
        assert old.terminal_reason == "legacy_suppressed"
        assert old.emailed_at is None
        # Not-yet-emailed row: untouched, will push once it reaches 'emailed'.
        cur = db.get_episode("f", "cur")
        assert cur is not None and cur.wiki_pushed_at is None

        db.bump_wiki_attempt("f", "cur")
        assert db.get_episode("f", "cur").wiki_push_attempts == 1
        assert db.get_episode("f", "cur").attempts == 0  # separate budget
        db.set_wiki_pushed("f", "cur")
        assert db.get_episode("f", "cur").wiki_pushed_at is not None
    finally:
        db.close()


def _insert_emailed(db: Database, guid: str, *, summary: str | None) -> None:
    """Insert an episode and drive it to 'emailed', optionally with a summary
    (None models a suppressed back-catalog episode that was never processed)."""
    db.insert_episode(
        feed_url="https://x/rss", guid=guid, title=guid.upper(),
        published_at="2026-06-10T00:00:00+00:00",
        duration_seconds=3600, audio_url=f"https://x/{guid}.mp3",
    )
    db.set_status(
        feed_url="https://x/rss", guid=guid, status="emailed",
        summary_html=summary,
        emailed_at="2026-06-10T11:00:00+00:00" if summary else None,
        terminal_reason="sent" if summary else "backlog_suppressed",
    )


def test_episodes_pending_wiki_gates_correctly(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://x/rss", name="X")

        # Emailed with a summary → eligible.
        _insert_emailed(db, "a", summary="<p>summary</p>")
        # Emailed but no summary (suppressed back-catalog) → NOT eligible.
        _insert_emailed(db, "b", summary=None)
        # Discovered-only (not emailed) → NOT eligible.
        db.insert_episode(
            feed_url="https://x/rss", guid="c", title="C",
            published_at="2026-06-10T00:00:00+00:00",
            duration_seconds=3600, audio_url="https://x/c.mp3",
        )

        pending = db.episodes_pending_wiki("https://x/rss")
        assert [p.guid for p in pending] == ["a"]

        # Once pushed, it drops out (idempotency watermark).
        db.set_wiki_pushed("https://x/rss", "a")
        assert db.episodes_pending_wiki("https://x/rss") == []
    finally:
        db.close()


def test_episodes_pending_wiki_caps_at_max_attempts(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://x/rss", name="X")
        _insert_emailed(db, "a", summary="<p>s</p>")
        assert len(db.episodes_pending_wiki("https://x/rss")) == 1
        for _ in range(WIKI_PUSH_MAX_ATTEMPTS):
            db.bump_wiki_attempt("https://x/rss", "a")
        # Budget spent → no longer retried, but never pushed (digest is the backstop).
        assert db.episodes_pending_wiki("https://x/rss") == []
        assert db.get_episode("https://x/rss", "a").wiki_pushed_at is None
    finally:
        db.close()


def test_resolved_url_cache_is_keyed_by_name_and_spec(tmp_path: Path):
    """F5 hardening: the identity cache only hits for the exact
    (name, source_spec) pair that produced it — an edited feeds.toml source or
    a never-cached feed must MISS so the pipeline re-resolves; and a re-key
    REPLACES the same row, so the newest resolution wins by construction (no
    row-ordering heuristics — the old rowid scheme broke on A→B→A re-keys)."""
    db = Database(tmp_path / "state.db")
    try:
        db.set_resolved_url("Market Hours", "Market Hours", "https://feeds/a.xml")
        assert db.get_resolved_url("Market Hours", "Market Hours") == "https://feeds/a.xml"
        # Edited spec (operator changed `search`/`youtube` in feeds.toml) → miss.
        assert db.get_resolved_url("Market Hours", "Market Hours Rebooted") is None
        # Unknown feed → miss (URLs cached before this table existed simply
        # aren't in it — one-time re-resolve with the current resolver).
        assert db.get_resolved_url("Example Host", "@examplehost") is None
        # Re-key A→B, then back B→A: each REPLACE wins outright.
        db.set_resolved_url("Market Hours", "Market Hours", "https://feeds/b.xml")
        assert db.get_resolved_url("Market Hours", "Market Hours") == "https://feeds/b.xml"
        db.set_resolved_url("Market Hours", "Market Hours", "https://feeds/a.xml")
        assert db.get_resolved_url("Market Hours", "Market Hours") == "https://feeds/a.xml"
    finally:
        db.close()


def test_resolved_url_cache_supports_two_names_for_one_url(tmp_path: Path):
    """Two feeds.toml entries resolving to the same URL keep independent cache
    rows — the old feeds-table scheme (PK url) flip-flopped a single row's
    name/spec each tick, so the cache never engaged for either feed."""
    db = Database(tmp_path / "state.db")
    try:
        db.set_resolved_url("Name A", "spec-a", "https://feeds/shared.xml")
        db.set_resolved_url("Name B", "spec-b", "https://feeds/shared.xml")
        assert db.get_resolved_url("Name A", "spec-a") == "https://feeds/shared.xml"
        assert db.get_resolved_url("Name B", "spec-b") == "https://feeds/shared.xml"
    finally:
        db.close()


def test_old_db_gains_resolved_urls_table(tmp_path: Path):
    """A pre-existing DB (created before resolved_urls existed) gains the table
    on open; legacy feeds rows never populate it (unknown provenance ⇒ the
    cache misses until the current resolver writes a fresh entry)."""
    raw = sqlite3.connect(tmp_path / "state.db")
    raw.executescript(
        """
        CREATE TABLE feeds (
            url TEXT PRIMARY KEY, name TEXT NOT NULL, itunes_collection_id INTEGER,
            first_seen_at TEXT NOT NULL, last_polled_at TEXT, last_error TEXT
        );
        """
    )
    raw.execute("INSERT INTO feeds (url, name, first_seen_at) VALUES ('u', 'N', 't')")
    raw.commit()
    raw.close()
    db = Database(tmp_path / "state.db")
    try:
        assert db.get_resolved_url("N", "anything") is None  # not cached: miss
        assert db.get_feed_rss_url("N") == "u"               # diagnostics still see it
        db.set_resolved_url("N", "anything", "u2")
        assert db.get_resolved_url("N", "anything") == "u2"
    finally:
        db.close()


def test_transport_hard_cap_finally_drops_the_canary(tmp_path: Path):
    """A deterministically transport-failing episode rotates through canary
    duty only until SUMMARIZE_TRANSPORT_HARD_CAP, then drops for real — it must
    not re-run its expensive LLM work every tick forever, nor keep the
    heartbeat muted permanently on an otherwise-quiet feed."""
    from podcast_digest.db import SUMMARIZE_TRANSPORT_HARD_CAP

    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://x/rss", name="X")
        db.insert_episode(
            feed_url="https://x/rss", guid="stuck", title="S",
            published_at="2024-04-02T10:00:00+00:00",
            duration_seconds=60, audio_url="https://x/s.mp3",
        )
        for _ in range(SUMMARIZE_TRANSPORT_HARD_CAP - 1):
            db.mark_failed("https://x/rss", "stuck", "summarizer down", transport=True)
        # One shy of the cap: still riding as the canary.
        assert [p.guid for p in db.episodes_in_progress("https://x/rss")] == ["stuck"]
        db.mark_failed("https://x/rss", "stuck", "summarizer down", transport=True)
        assert db.episodes_in_progress("https://x/rss") == []
    finally:
        db.close()


def test_canary_rotation_prefers_least_tried(tmp_path: Path):
    """The canary slot rotates least-tried-first, so a deterministically
    failing episode can't monopolize it and starve the rest of a throttled
    backlog of their recovery probe."""
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://x/rss", name="X")
        for guid, ts in (("older", "2024-04-01T10:00:00+00:00"),
                         ("newer", "2024-04-02T10:00:00+00:00")):
            db.insert_episode(
                feed_url="https://x/rss", guid=guid, title=guid,
                published_at=ts, duration_seconds=60,
                audio_url=f"https://x/{guid}.mp3",
            )
        for _ in range(SUMMARIZE_TRANSPORT_MAX_ATTEMPTS):
            db.mark_failed("https://x/rss", "older", "summarizer down", transport=True)
        for _ in range(SUMMARIZE_TRANSPORT_MAX_ATTEMPTS + 2):
            db.mark_failed("https://x/rss", "newer", "summarizer down", transport=True)
        # Both throttled; 'newer' has MORE failures → 'older' gets the slot.
        assert [p.guid for p in db.episodes_in_progress("https://x/rss")] == ["older"]
        # Ties go to the newest (stable min over a newest-first scan).
        db.mark_failed("https://x/rss", "older", "summarizer down", transport=True)
        db.mark_failed("https://x/rss", "older", "summarizer down", transport=True)
        assert [p.guid for p in db.episodes_in_progress("https://x/rss")] == ["newer"]
    finally:
        db.close()


def test_summary_html_sanitized_by_one_time_migration(tmp_path: Path):
    """Rows stored before production-time sanitizing get cleaned exactly once
    at open (PRAGMA user_version gate), making 'stored summaries are always
    sanitized' a real invariant every consumer can trust."""
    db = Database(tmp_path / "state.db")
    db.upsert_feed(url="https://x/rss", name="X")
    db.insert_episode(
        feed_url="https://x/rss", guid="a", title="A",
        published_at="2024-04-02T10:00:00+00:00",
        duration_seconds=60, audio_url="https://x/a.mp3",
    )
    # Simulate a pre-sanitizer row: raw model output straight into the column,
    # and rewind user_version as it would be on an old DB.
    db._conn.execute(
        "UPDATE episodes SET summary_html=? WHERE guid='a'",
        ("<h2>ok</h2><script>alert(1)</script>",),
    )
    db._conn.execute("PRAGMA user_version = 0")
    db.close()

    db = Database(tmp_path / "state.db")
    try:
        stored = db.get_episode("https://x/rss", "a").summary_html
        assert "<script" not in stored and "<h2>ok</h2>" in stored
        assert db._conn.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        db.close()


def test_mark_failed_returns_exhausted_from_post_bump_value(tmp_path: Path):
    """mark_failed owns the exhaustion verdict, computed from the SAME UPDATE's
    post-bump counter — a caller's tick-start snapshot is stale whenever an
    intermediate stage success reset the counters mid-tick, and snapshot+1
    arithmetic would spuriously GC a still-pending episode's audio."""
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://x/rss", name="X")
        db.insert_episode(
            feed_url="https://x/rss", guid="a", title="A",
            published_at="2024-04-02T10:00:00+00:00",
            duration_seconds=60, audio_url="https://x/a.mp3",
        )
        for _ in range(MAX_ATTEMPTS - 1):
            assert db.mark_failed("https://x/rss", "a", "blip") is False
        # The stale-snapshot scenario: episode enters a tick at attempts=4, a
        # stage succeeds (reset to 0), then a later stage fails — NOT exhausted.
        db.set_status(feed_url="https://x/rss", guid="a", status="downloaded")
        assert db.mark_failed("https://x/rss", "a", "late failure") is False
        assert [p.guid for p in db.episodes_in_progress("https://x/rss")] == ["a"]
        # Genuine exhaustion returns True exactly on the capping failure.
        for _ in range(MAX_ATTEMPTS - 2):
            assert db.mark_failed("https://x/rss", "a", "blip") is False
        assert db.mark_failed("https://x/rss", "a", "final") is True
        assert db.episodes_in_progress("https://x/rss") == []
        # Unknown episode: no row updated, never "exhausted".
        assert db.mark_failed("https://x/rss", "ghost", "?") is False
    finally:
        db.close()


def test_rekey_feed_migrates_episand_feed_rows(tmp_path: Path):
    """rekey_feed carries the whole state machine to the new PK: episode rows
    (in-flight ones keep their stage), the feeds row (last_polled_at survives,
    so first_run can't re-fire and re-email/pre-suppress), and guid collisions
    prefer the row already under the new URL."""
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://old/rss", name="X")
        db.mark_feed_polled("https://old/rss")
        for guid, status in (("sent", "emailed"), ("inflight", "transcribed")):
            db.insert_episode(
                feed_url="https://old/rss", guid=guid, title=guid,
                published_at="2024-04-02T10:00:00+00:00",
                duration_seconds=60, audio_url=f"https://x/{guid}.mp3",
            )
            db.set_status(feed_url="https://old/rss", guid=guid, status=status)
        # A colliding row already tracked under the new URL survives as-is.
        db.upsert_feed(url="https://new/rss", name="X")
        db.insert_episode(
            feed_url="https://new/rss", guid="sent", title="sent",
            published_at="2024-04-02T10:00:00+00:00",
            duration_seconds=60, audio_url="https://x/sent.mp3",
        )

        migrated = db.rekey_feed("https://old/rss", "https://new/rss")
        assert migrated == 1  # 'inflight' moved; 'sent' collided (kept new row)
        assert db.get_episode("https://old/rss", "inflight") is None
        moved = db.get_episode("https://new/rss", "inflight")
        assert moved is not None and moved.status == "transcribed"
        assert db.get_episode("https://new/rss", "sent") is not None
        # first_run must NOT re-fire under the new PK (polled watermark kept).
        assert db.upsert_feed(url="https://new/rss", name="X") is False
    finally:
        db.close()


def test_migrate_legacy_synthetic_guid_preserves_episode_state(tmp_path: Path):
    """A pre-upgrade enclosure-URL GUID is adopted without reprocessing."""
    db = Database(tmp_path / "state.db")
    feed_url = "https://publisher.example/rss"
    old_guid = "https://cdn.example/episode.mp3?signature=old"
    new_guid = "synthetic:v1:" + "a" * 64
    try:
        db.insert_episode(
            feed_url=feed_url,
            guid=old_guid,
            title="Unsigned episode",
            published_at="2026-07-22T10:00:00+00:00",
            duration_seconds=3600,
            audio_url=old_guid,
        )
        db.set_status(feed_url=feed_url, guid=old_guid, status="emailed")

        assert db.migrate_legacy_synthetic_guid(
            feed_url=feed_url,
            guid=new_guid,
            title="Unsigned episode",
            published_at="2026-07-22T10:00:00+00:00",
            duration_seconds=3601,
            audio_url="https://cdn.example/episode.mp3?signature=rotated",
        )

        assert db.get_episode(feed_url, old_guid) is None
        migrated = db.get_episode(feed_url, new_guid)
        assert migrated is not None
        assert migrated.status == "emailed"
        assert migrated.audio_url.endswith("signature=rotated")
        assert migrated.duration_seconds == 3601

        assert not db.migrate_legacy_synthetic_guid(
            feed_url=feed_url,
            guid=new_guid,
            title="Unsigned episode",
            published_at="2026-07-22T10:00:00+00:00",
            duration_seconds=3602,
            audio_url="https://cdn.example/episode.mp3?signature=newest",
        )
        refreshed = db.get_episode(feed_url, new_guid)
        assert refreshed is not None
        assert refreshed.audio_url.endswith("signature=newest")
        assert refreshed.duration_seconds == 3602
    finally:
        db.close()


def test_unique_feed_url_refuses_ambiguous_display_name(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    try:
        db.upsert_feed(url="https://one.example/rss", name="Same name")
        assert db.get_unique_feed_url("Same name") == "https://one.example/rss"
        db.upsert_feed(url="https://two.example/rss", name="Same name")
        assert db.get_unique_feed_url("Same name") is None
    finally:
        db.close()
