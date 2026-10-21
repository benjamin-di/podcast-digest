"""SQLite state.

Two tables: `feeds` tracks per-podcast metadata + first-seen watermark;
`episodes` tracks every discovered episode plus its forward-only status.

Status machine: discovered -> downloaded -> transcribed -> summarized -> emailed.
Failure is orthogonal to status: `mark_failed` leaves the status alone (so the
next tick resumes from the last successful stage), and bumps `attempts` plus
`last_error` + `failed_at`. Episodes whose `attempts` exceed MAX_ATTEMPTS are
excluded from the pending queue until manually cleared.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .dedupe import ContentCandidate, TranscriptFingerprint
from .sanitize import sanitize_summary_html

logger = logging.getLogger(__name__)

# Forward order of status values. Lower index = earlier stage.
STAGES = ("discovered", "downloaded", "transcribed", "summarized", "emailed")
_STAGE_IDX = {s: i for i, s in enumerate(STAGES)}

# After this many consecutive failures we stop retrying and leave the episode
# in the DB for inspection. The operator can clear `attempts` to retry.
MAX_ATTEMPTS = 5

# Bound on wiki-push retries. The push is a non-fatal post-email step; if it
# keeps failing we stop retrying (the digest email is the durable backstop) but
# leave the row for inspection. Separate budget from `attempts` so a flaky wiki
# never blocks or drops the actual digest pipeline.
WIKI_PUSH_MAX_ATTEMPTS = 5

# Backoff threshold for *summarizer-transport* failures (SummarizerUnavailable),
# tracked separately from `attempts`. Transport failures are a property of the
# summarizer backend, not the episode, so crossing this threshold must not
# immediately DROP an episode — during a long outage every pending episode
# would cross it and be dark-holed right as the last one stopped attempting
# (which would also zero `summarize_failed` and falsely re-arm the
# dead-man's-switch heartbeat mid-outage). Instead, episodes past the
# threshold are throttled to a CANARY: one attempt per tick, least-tried
# first (see episodes_in_progress), and the pipeline additionally limits
# canary attempts to one per TICK across feeds once an outage is evident.
# During an outage the canary keeps the heartbeat honestly muted; after
# recovery the backlog drains one canary per tick, each success resetting its
# OWN counters via set_status. Deliberately NO bulk reset on another
# episode's success: that would zero a deterministically-failing episode's
# hard-cap progress every time its feed publishes normally, making the drop
# below unreachable on any active feed. At the default 6h poll, 10 ticks ≈
# 2.5 days of full-rate retries before throttling kicks in.
SUMMARIZE_TRANSPORT_MAX_ATTEMPTS = 10

# The genuine drop cap for transport failures. A single episode that
# deterministically fails transport (an oversized transcript that always times
# out) rotates through the canary slot until it alone accumulates this many
# failures, then drops for real — otherwise it would re-run its expensive LLM
# work forever and, on an otherwise-quiet feed, keep the heartbeat muted
# permanently (a standing false alert). 3× the threshold ≈ 20 canary slices
# past throttling; a genuinely global outage advances counters only one
# canary per tick, so even a multi-week outage doesn't exhaust a backlog.
SUMMARIZE_TRANSPORT_HARD_CAP = 3 * SUMMARIZE_TRANSPORT_MAX_ATTEMPTS


def stage_lt(current: str | None, target: str) -> bool:
    """True if `current` precedes `target` in the pipeline.

    `current` is the *highest stage successfully completed*; failures don't
    move it, so resumption naturally picks up at the right place.
    """
    if current is None:
        return True
    return _STAGE_IDX.get(current, -1) < _STAGE_IDX[target]


SCHEMA = """
CREATE TABLE IF NOT EXISTS feeds (
    url TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    itunes_collection_id INTEGER,
    first_seen_at TEXT NOT NULL,
    last_polled_at TEXT,
    last_error TEXT
);

-- Identity-resolution cache: the URL a (feed name, configured source value)
-- pair last resolved to. Deliberately its OWN table rather than columns on
-- `feeds` (whose PK is url): a re-key is then an INSERT OR REPLACE of the
-- same row — no "newest row wins" ordering heuristics — and two feeds that
-- resolve to one URL keep independent cache entries. Rows are never trusted
-- across a config edit: the spec is part of the key, so editing feeds.toml
-- misses the cache and re-resolves naturally.
CREATE TABLE IF NOT EXISTS resolved_urls (
    name TEXT NOT NULL,
    source_spec TEXT NOT NULL,
    url TEXT NOT NULL,
    resolved_at TEXT NOT NULL,
    PRIMARY KEY (name, source_spec)
);

CREATE TABLE IF NOT EXISTS episodes (
    feed_url TEXT NOT NULL,
    guid TEXT NOT NULL,
    title TEXT NOT NULL,
    published_at TEXT NOT NULL,
    duration_seconds INTEGER,
    audio_url TEXT NOT NULL,
    audio_path TEXT,
    transcript_path TEXT,
    segments_path TEXT,
    summary_html TEXT,
    content_sha256 TEXT,
    content_minhash TEXT,
    content_token_count INTEGER,
    status TEXT NOT NULL,
    terminal_reason TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    clean_attempts INTEGER NOT NULL DEFAULT 0,
    summarize_attempts INTEGER NOT NULL DEFAULT 0,
    wiki_push_attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    failed_at TEXT,
    emailed_at TEXT,
    wiki_pushed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (feed_url, guid)
);

-- Small key/value store for instance-level state (e.g. when the delivery
-- webhook was switched on).
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_status
    ON episodes(status);
CREATE INDEX IF NOT EXISTS idx_episodes_published
    ON episodes(feed_url, published_at DESC);
"""


@dataclass
class EpisodeRow:
    feed_url: str
    guid: str
    title: str
    published_at: str
    duration_seconds: int | None
    audio_url: str
    audio_path: str | None
    transcript_path: str | None
    segments_path: str | None
    summary_html: str | None
    content_sha256: str | None
    content_minhash: str | None
    content_token_count: int | None
    status: str
    terminal_reason: str | None
    attempts: int
    clean_attempts: int
    summarize_attempts: int
    wiki_push_attempts: int
    last_error: str | None
    failed_at: str | None
    emailed_at: str | None
    wiki_pushed_at: str | None

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> EpisodeRow:
        return cls(
            feed_url=r["feed_url"],
            guid=r["guid"],
            title=r["title"],
            published_at=r["published_at"],
            duration_seconds=r["duration_seconds"],
            audio_url=r["audio_url"],
            audio_path=r["audio_path"],
            transcript_path=r["transcript_path"],
            segments_path=r["segments_path"],
            summary_html=r["summary_html"],
            content_sha256=r["content_sha256"],
            content_minhash=r["content_minhash"],
            content_token_count=r["content_token_count"],
            status=r["status"],
            terminal_reason=r["terminal_reason"],
            attempts=r["attempts"],
            clean_attempts=r["clean_attempts"],
            summarize_attempts=r["summarize_attempts"],
            wiki_push_attempts=r["wiki_push_attempts"],
            last_error=r["last_error"],
            failed_at=r["failed_at"],
            emailed_at=r["emailed_at"],
            wiki_pushed_at=r["wiki_pushed_at"],
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    """Thin SQLite wrapper. One connection per process is fine — pipeline is serial."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, isolation_level=None)  # autocommit
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Idempotently add columns missing from a pre-existing DB.

        ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so new
        columns need an explicit ALTER. Cheap to run on every boot; a no-op once
        the column exists.
        """
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(episodes)")}
        if "clean_attempts" not in have:
            self._conn.execute(
                "ALTER TABLE episodes ADD COLUMN clean_attempts INTEGER NOT NULL DEFAULT 0"
            )
        if "summarize_attempts" not in have:
            self._conn.execute(
                "ALTER TABLE episodes ADD COLUMN summarize_attempts INTEGER NOT NULL DEFAULT 0"
            )
        if "wiki_push_attempts" not in have:
            self._conn.execute(
                "ALTER TABLE episodes ADD COLUMN wiki_push_attempts INTEGER NOT NULL DEFAULT 0"
            )
        if "content_sha256" not in have:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN content_sha256 TEXT")
        if "content_minhash" not in have:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN content_minhash TEXT")
        if "content_token_count" not in have:
            self._conn.execute(
                "ALTER TABLE episodes ADD COLUMN content_token_count INTEGER"
            )
        if "terminal_reason" not in have:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN terminal_reason TEXT")
            # Old rows overloaded `emailed` for delivered and deliberately
            # suppressed work. Preserve compatibility while making the
            # distinction explicit for every historical terminal row.
            self._conn.execute(
                """
                UPDATE episodes
                SET terminal_reason = CASE
                    WHEN summary_html IS NOT NULL AND summary_html != '' THEN 'sent'
                    ELSE 'legacy_suppressed'
                END
                WHERE status='emailed'
                """
            )
            self._conn.execute(
                """
                UPDATE episodes
                SET emailed_at=NULL
                WHERE status='emailed' AND terminal_reason='legacy_suppressed'
                """
            )
        # One-time cleanup of rows stored before production-time sanitizing
        # existed: make "episodes.summary_html is always sanitized" actually
        # true, so consumers (email, wiki push, the MCP preview, anything
        # added later) can trust the DB instead of each re-sanitizing forever.
        ver = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if ver < 1:
            # One transaction: a mature DB can have hundreds of changed rows
            # (nh3 adds rel= to every <a>), and per-row autocommit would mean
            # one fsync each on a slow disk — plus a crash mid-loop redoing
            # the whole pass. Atomic + single fsync + version stamp together.
            with self.tx():
                rows = self._conn.execute(
                    "SELECT feed_url, guid, summary_html FROM episodes "
                    "WHERE summary_html IS NOT NULL AND summary_html != ''"
                ).fetchall()
                for r in rows:
                    clean = sanitize_summary_html(r["summary_html"])
                    if clean != r["summary_html"]:
                        self._conn.execute(
                            "UPDATE episodes SET summary_html=? WHERE feed_url=? AND guid=?",
                            (clean, r["feed_url"], r["guid"]),
                        )
                self._conn.execute("PRAGMA user_version = 1")
        if "wiki_pushed_at" not in have:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN wiki_pushed_at TEXT")
            # Forward-only: treat everything already emailed as 'pushed' so that
            # enabling wiki ingest on an existing DB does NOT retroactively flood
            # the wiki with the entire back catalogue. Only episodes emailed
            # AFTER this migration are eligible for a push. (A fresh DB created
            # from SCHEMA skips this block and has no history to guard.)
            self._conn.execute(
                "UPDATE episodes SET wiki_pushed_at=? WHERE status='emailed'",
                (_now(),),
            )

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        # autocommit + BEGIN/COMMIT keeps multi-statement updates atomic.
        self._conn.execute("BEGIN")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ---- feeds ---------------------------------------------------------

    def upsert_feed(
        self,
        *,
        url: str,
        name: str,
        itunes_collection_id: int | None = None,
    ) -> bool:
        """Upsert a feed row. Returns True if this was the first time we saw it."""
        now = _now()
        cur = self._conn.execute(
            """
            INSERT INTO feeds (url, name, itunes_collection_id, first_seen_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                name = excluded.name,
                itunes_collection_id = COALESCE(excluded.itunes_collection_id, feeds.itunes_collection_id)
            RETURNING (last_polled_at IS NULL) AS is_first
            """,
            (url, name, itunes_collection_id, now),
        )
        row = cur.fetchone()
        return bool(row["is_first"]) if row else False

    def mark_feed_polled(self, url: str, *, error: str | None = None) -> None:
        self._conn.execute(
            "UPDATE feeds SET last_polled_at=?, last_error=? WHERE url=?",
            (_now(), error, url),
        )

    def get_feed_rss_url(self, name: str) -> str | None:
        """Best-effort URL lookup by display name — diagnostics only (the
        feed-failed error path). NOT a resolution cache: `name` is not unique
        (PK is url; re-keys leave same-name siblings), so identity decisions go
        through :meth:`get_resolved_url`, which demands provenance."""
        cur = self._conn.execute(
            "SELECT url FROM feeds WHERE name=? ORDER BY rowid DESC", (name,)
        )
        row = cur.fetchone()
        return row["url"] if row else None

    def get_unique_feed_url(self, name: str) -> str | None:
        """Return the feed URL for ``name`` only when it is unambiguous.

        This is a compatibility fallback for state created before the
        provenance-keyed resolution cache existed. Identity-sensitive paths
        must never use :meth:`get_feed_rss_url`, whose newest-row behavior is
        intentionally suitable only for diagnostics.
        """
        rows = self._conn.execute(
            "SELECT url FROM feeds WHERE name=? LIMIT 2", (name,)
        ).fetchall()
        return rows[0]["url"] if len(rows) == 1 else None

    def get_resolved_url(self, name: str, source_spec: str) -> str | None:
        """The cached identity URL for exactly this (name, source_spec) pair.

        Keying on the configured source value — not just the name — makes the
        cache safe to trust forever-until-changed: editing the feed's
        `search`/`youtube` value in feeds.toml changes the key, misses, and
        re-resolves naturally; and URLs cached before this table existed (or by
        a since-fixed resolver bug) simply aren't in it.
        """
        cur = self._conn.execute(
            "SELECT url FROM resolved_urls WHERE name=? AND source_spec=?",
            (name, source_spec),
        )
        row = cur.fetchone()
        return row["url"] if row else None

    def set_resolved_url(self, name: str, source_spec: str, url: str) -> None:
        """Record what this (name, spec) resolved to. A re-key REPLACES the
        same row, so the newest resolution always wins — by construction, not
        by row-ordering heuristics."""
        self._conn.execute(
            "INSERT OR REPLACE INTO resolved_urls (name, source_spec, url, resolved_at) "
            "VALUES (?, ?, ?, ?)",
            (name, source_spec, url, _now()),
        )

    # ---- episodes ------------------------------------------------------

    def has_episode(self, feed_url: str, guid: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM episodes WHERE feed_url=? AND guid=?",
            (feed_url, guid),
        )
        return cur.fetchone() is not None

    def insert_episode(
        self,
        *,
        feed_url: str,
        guid: str,
        title: str,
        published_at: str,
        duration_seconds: int | None,
        audio_url: str,
    ) -> None:
        now = _now()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO episodes (
                feed_url, guid, title, published_at, duration_seconds, audio_url,
                status, attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'discovered', 0, ?, ?)
            """,
            (feed_url, guid, title, published_at, duration_seconds, audio_url, now, now),
        )

    def get_episode(self, feed_url: str, guid: str) -> EpisodeRow | None:
        cur = self._conn.execute(
            "SELECT * FROM episodes WHERE feed_url=? AND guid=?",
            (feed_url, guid),
        )
        row = cur.fetchone()
        return EpisodeRow.from_row(row) if row else None

    def migrate_legacy_synthetic_guid(
        self,
        *,
        feed_url: str,
        guid: str,
        title: str,
        published_at: str,
        duration_seconds: int | None,
        audio_url: str,
    ) -> bool:
        """Refresh a stable row or adopt its GUID for one legacy row.

        Older releases used the complete enclosure URL as the fallback GUID.
        Only rows whose GUID exactly equals their stored audio URL can be from
        that fallback. Matching the title, timestamp, and query-free enclosure
        URL prevents a one-time duplicate when upgrading while refusing to
        guess if multiple legacy rows could represent the new entry. Existing
        stable rows receive the latest signed enclosure URL for crash recovery.
        """
        if self.has_episode(feed_url, guid):
            self._conn.execute(
                """
                UPDATE episodes
                SET audio_url=?, duration_seconds=COALESCE(?, duration_seconds),
                    updated_at=?
                WHERE feed_url=? AND guid=?
                """,
                (audio_url, duration_seconds, _now(), feed_url, guid),
            )
            return False
        candidates = self._conn.execute(
            """
            SELECT guid, audio_url FROM episodes
            WHERE feed_url=? AND title=? AND published_at=? AND guid=audio_url
            """,
            (feed_url, title, published_at),
        ).fetchall()
        stable_audio = _url_without_query(audio_url)
        matches = [
            row for row in candidates
            if _url_without_query(row["audio_url"]) == stable_audio
        ]
        if len(matches) != 1:
            return False
        old_guid = matches[0]["guid"]
        cur = self._conn.execute(
            """
            UPDATE episodes
            SET guid=?, audio_url=?, duration_seconds=COALESCE(?, duration_seconds),
                updated_at=?
            WHERE feed_url=? AND guid=?
            """,
            (guid, audio_url, duration_seconds, _now(), feed_url, old_guid),
        )
        return cur.rowcount == 1

    def set_status(
        self,
        *,
        feed_url: str,
        guid: str,
        status: str,
        audio_path: str | None = None,
        transcript_path: str | None = None,
        segments_path: str | None = None,
        summary_html: str | None = None,
        emailed_at: str | None = None,
        terminal_reason: str | None = None,
    ) -> None:
        # Reset both failure budgets to 0 on every successful stage transition.
        # They are *consecutive*-failure counters (a cap drops the episode from
        # the pending queue), so they must clear when a stage actually succeeds —
        # otherwise blips accumulate across an episode's whole life (2 download +
        # 2 transcribe + 1 summarize = dropped) and, worse, a multi-tick
        # summarizer outage bumps them once per tick until the episode is
        # silently dark-holed. `summarize_attempts` is the transport budget (see
        # mark_failed's `transport` flag); resetting it here is what lets an
        # episode recover the moment the summarizer comes back.
        cols = ["status=?", "updated_at=?", "last_error=NULL",
                "attempts=0", "summarize_attempts=0"]
        vals: list[object] = [status, _now()]
        if audio_path is not None:
            cols.append("audio_path=?"); vals.append(audio_path)
        if transcript_path is not None:
            cols.append("transcript_path=?"); vals.append(transcript_path)
        if segments_path is not None:
            cols.append("segments_path=?"); vals.append(segments_path)
        if summary_html is not None:
            cols.append("summary_html=?"); vals.append(summary_html)
        if emailed_at is not None:
            cols.append("emailed_at=?"); vals.append(emailed_at)
        if terminal_reason is not None:
            if status != "emailed":
                raise ValueError("terminal_reason requires status='emailed'")
            cols.append("terminal_reason=?"); vals.append(terminal_reason)
        vals.extend([feed_url, guid])
        self._conn.execute(
            f"UPDATE episodes SET {', '.join(cols)} WHERE feed_url=? AND guid=?",
            vals,
        )

    def clear_audio_path(self, feed_url: str, guid: str) -> None:
        self._conn.execute(
            "UPDATE episodes SET audio_path=NULL, updated_at=? WHERE feed_url=? AND guid=?",
            (_now(), feed_url, guid),
        )

    def set_content_fingerprint(
        self,
        feed_url: str,
        guid: str,
        fingerprint: TranscriptFingerprint,
    ) -> None:
        self._conn.execute(
            """
            UPDATE episodes
            SET content_sha256=?, content_minhash=?, content_token_count=?, updated_at=?
            WHERE feed_url=? AND guid=?
            """,
            (
                fingerprint.sha256,
                fingerprint.serialize_minhash(),
                fingerprint.token_count,
                _now(),
                feed_url,
                guid,
            ),
        )

    def delivered_content_since(self, since: str) -> list[ContentCandidate]:
        """Return fingerprints for genuine prior deliveries, never terminal skips."""
        rows = self._conn.execute(
            """
            SELECT feed_url, guid, title, emailed_at,
                   content_sha256, content_minhash, content_token_count
            FROM episodes
            WHERE status='emailed'
              AND summary_html IS NOT NULL
              AND emailed_at >= ?
              AND content_sha256 IS NOT NULL
              AND content_minhash IS NOT NULL
              AND content_token_count IS NOT NULL
            """,
            (since,),
        ).fetchall()
        candidates: list[ContentCandidate] = []
        for row in rows:
            try:
                fingerprint = TranscriptFingerprint.from_storage(
                    row["content_sha256"],
                    row["content_minhash"],
                    row["content_token_count"],
                )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "episodes.invalid_content_fingerprint feed=%s guid=%s err=%s",
                    row["feed_url"],
                    row["guid"],
                    exc,
                )
                continue
            candidates.append(
                ContentCandidate(
                    feed_url=row["feed_url"],
                    guid=row["guid"],
                    title=row["title"],
                    emitted_at=row["emailed_at"],
                    fingerprint=fingerprint,
                )
            )
        return candidates

    def mark_failed(
        self, feed_url: str, guid: str, error: str, *, transport: bool = False
    ) -> bool:
        """Record a failure WITHOUT clobbering `status`.

        Status is the high-water-mark of completed stages; failures leave it
        alone so resumption picks up where it left off. We stash the error for
        diagnostics and bump a failure counter toward a drop cap.

        ``transport=False`` (default): a content/email/agent-output failure — bump
        the general ``attempts`` counter (dropped at ``MAX_ATTEMPTS``).

        ``transport=True``: a *summarizer-transport* failure (``SummarizerUnavailable``) —
        bump the separate ``summarize_attempts`` counter instead. A whole-
        summarizer outage isn't the episode's fault and can span many ticks, so
        it must not exhaust the small ``attempts`` budget. Past
        ``SUMMARIZE_TRANSPORT_MAX_ATTEMPTS`` the episode is throttled to canary
        duty (see ``episodes_in_progress``); at ``SUMMARIZE_TRANSPORT_HARD_CAP``
        it genuinely drops, so a deterministically-failing episode cannot re-run
        its expensive LLM work forever. Both counters reset to 0 on the next
        successful stage (see ``set_status``).

        Returns True when THIS failure exhausted the episode's budget (it will
        no longer appear in ``episodes_in_progress``). Computed here, from the
        post-bump value the same UPDATE produced — the caller's tick-start
        snapshot is stale whenever an intermediate stage succeeded this tick
        (``set_status`` resets the counters), so callers must not re-derive
        this with snapshot+1 arithmetic.
        """
        now = _now()
        bump = "summarize_attempts=summarize_attempts+1, " if transport \
            else "attempts=attempts+1, "
        row = self._conn.execute(
            f"""
            UPDATE episodes
            SET last_error=?, failed_at=?, {bump}updated_at=?
            WHERE feed_url=? AND guid=?
            RETURNING attempts, summarize_attempts
            """,
            (error[:1000], now, now, feed_url, guid),
        ).fetchone()
        if row is None:
            return False
        if transport:
            return row["summarize_attempts"] >= SUMMARIZE_TRANSPORT_HARD_CAP
        return row["attempts"] >= MAX_ATTEMPTS

    def bump_clean_attempt(self, feed_url: str, guid: str) -> None:
        """Count one transcript-cleanup attempt for this episode.

        Separate from `attempts` (the main per-stage retry budget) so that
        cleanup retries never trip MAX_ATTEMPTS and drop the episode — cleanup
        degrades gracefully to a raw-transcript digest instead of failing it.
        """
        self._conn.execute(
            "UPDATE episodes SET clean_attempts=clean_attempts+1, updated_at=? "
            "WHERE feed_url=? AND guid=?",
            (_now(), feed_url, guid),
        )

    # ---- wiki push (post-email, non-fatal) -----------------------------

    def episodes_pending_wiki(
        self, feed_url: str, *, emailed_since: str | None = None,
    ) -> list[EpisodeRow]:
        """Emailed episodes not yet pushed to the webhook, within the retry
        budget (the wiki_* columns hold the webhook's delivery state).

        Gated on `summary_html IS NOT NULL` so that *suppressed* back-catalog
        episodes (pre-marked 'emailed' but never processed — no summary, no
        transcript) are never pushed, and on `emailed_since` (see
        `webhook_enabled_since`) so switching the webhook on never posts the
        history emailed before it. A freshly-emailed episode becomes eligible
        in the same tick; a failed push retries on later ticks until the budget
        is spent.
        """
        since_clause = "AND emailed_at >= ?" if emailed_since else ""
        params: tuple = (feed_url, WIKI_PUSH_MAX_ATTEMPTS)
        if emailed_since:
            params += (emailed_since,)
        cur = self._conn.execute(
            f"""
            SELECT * FROM episodes
            WHERE feed_url=?
              AND status = 'emailed'
              AND wiki_pushed_at IS NULL
              AND summary_html IS NOT NULL
              AND wiki_push_attempts < ?
              {since_clause}
            ORDER BY published_at DESC
            """,
            params,
        )
        return [EpisodeRow.from_row(r) for r in cur.fetchall()]

    def webhook_enabled_since(self, enabled: bool) -> str | None:
        """When the delivery webhook was switched on (ISO UTC), recorded on the
        first tick it is configured; None while it is off.

        The marker is cleared on every tick the webhook is off, so a later
        re-enable starts from that moment instead of posting everything emailed
        in between (which, on a mature database, would be the whole history).
        """
        if not enabled:
            self._conn.execute("DELETE FROM meta WHERE key='webhook_enabled_since'")
            return None
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('webhook_enabled_since', ?)",
            (_now(),),
        )
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='webhook_enabled_since'"
        ).fetchone()
        return str(row[0])

    def set_wiki_pushed(self, feed_url: str, guid: str) -> None:
        """Mark an episode as pushed to the wiki (idempotency watermark)."""
        self._conn.execute(
            "UPDATE episodes SET wiki_pushed_at=?, updated_at=? "
            "WHERE feed_url=? AND guid=?",
            (_now(), _now(), feed_url, guid),
        )

    def bump_wiki_attempt(self, feed_url: str, guid: str) -> None:
        """Count one failed wiki-push attempt (bounds retries; non-fatal path)."""
        self._conn.execute(
            "UPDATE episodes SET wiki_push_attempts=wiki_push_attempts+1, updated_at=? "
            "WHERE feed_url=? AND guid=?",
            (_now(), feed_url, guid),
        )

    def episodes_in_progress(self, feed_url: str) -> list[EpisodeRow]:
        """Episodes that started but didn't reach `emailed` and haven't blown
        either budget — plus at most ONE transport-throttled canary.

        Episodes past the transport threshold are excluded from the full sweep
        (their expensive LLM clean+summarize isn't re-run en masse while the
        summarizer is down) but not dropped until the hard cap: one rides along
        as a canary. During an outage the canary's failure keeps the
        dead-man's-switch heartbeat honestly muted; after recovery each tick's
        canary succeeds and rejoins normal flow (set_status resets its
        counters), draining the throttled backlog one episode per tick.
        Without the canary, a >threshold outage would silence every pending
        episode at once — zero attempts, zero failures, heartbeat falsely
        healthy, backlog dark-holed forever.

        Canary selection is LEAST-TRIED first (tie: newest): a deterministically
        failing episode accumulates attempts and falls to the back of the
        rotation instead of monopolizing the slot, so the rest of a recovered
        backlog still gets its canary turn; at SUMMARIZE_TRANSPORT_HARD_CAP the
        stuck episode drops for real.
        """
        cur = self._conn.execute(
            """
            SELECT * FROM episodes
            WHERE feed_url=?
              AND status != 'emailed'
              AND attempts < ?
              AND summarize_attempts < ?
            ORDER BY published_at DESC
            """,
            (feed_url, MAX_ATTEMPTS, SUMMARIZE_TRANSPORT_HARD_CAP),
        )
        rows = [EpisodeRow.from_row(r) for r in cur.fetchall()]
        active = [r for r in rows if r.summarize_attempts < SUMMARIZE_TRANSPORT_MAX_ATTEMPTS]
        throttled = [r for r in rows if r.summarize_attempts >= SUMMARIZE_TRANSPORT_MAX_ATTEMPTS]
        if throttled:
            # rows are newest-first; min() is stable, so ties pick the newest.
            active.append(min(throttled, key=lambda r: r.summarize_attempts))
        return active

    def rekey_feed(self, old_url: str, new_url: str) -> int:
        """Migrate a feed's identity from one URL PK to another, atomically.

        A genuine re-key (host migration, channel moved) must carry the feed's
        state with it: without this, the new PK starts empty, so first_run
        re-fires (the already-sent newest episode is re-emailed on single-instance
        deploys) and every in-flight episode is re-inserted under the new URL
        and pre-suppressed by the first-run cap — silently lost — while its old
        row strands forever. Migrating the rows makes a re-key invisible to the
        episode state machine.

        Guid collisions (an episode already tracked under the new URL) keep the
        new row and drop the old one. Returns the number of episodes migrated.
        """
        with self.tx():
            cur = self._conn.execute(
                "UPDATE OR IGNORE episodes SET feed_url=?, updated_at=? "
                "WHERE feed_url=?",
                (new_url, _now(), old_url),
            )
            migrated = cur.rowcount
            self._conn.execute("DELETE FROM episodes WHERE feed_url=?", (old_url,))
            # If a row already exists under the new URL (e.g. the target was
            # tracked before), the OR IGNORE below keeps it — so carry the old
            # row's polled watermark over first, or first_run would re-fire on
            # a never-polled survivor.
            self._conn.execute(
                "UPDATE feeds SET last_polled_at = COALESCE(last_polled_at, "
                "  (SELECT last_polled_at FROM feeds WHERE url=?)) WHERE url=?",
                (old_url, new_url),
            )
            self._conn.execute(
                "UPDATE OR IGNORE feeds SET url=? WHERE url=?", (new_url, old_url)
            )
            self._conn.execute("DELETE FROM feeds WHERE url=?", (old_url,))
        return migrated


def _url_without_query(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, "", "")
    )
