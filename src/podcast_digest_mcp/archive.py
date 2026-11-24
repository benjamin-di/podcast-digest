"""Read-only access to state.db and saved transcripts for the MCP server.

The MCP server never mutates SQLite or transcript files. When `/data` is
bind-mounted read-only in Docker and `state.db` uses WAL mode, SQLite cannot
create/map `-shm` in the read-only directory; in that case we snapshot
`state.db` (plus `-wal` if present) into a private temporary directory so
SQLite replays committed WAL frames cleanly without touching `/data`.
"""
from __future__ import annotations

import atexit
import contextlib
import hashlib
import logging
import os
import re
import shlex
import shutil
import sqlite3
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from podcast_digest.db import MAX_ATTEMPTS, SUMMARIZE_TRANSPORT_HARD_CAP
from podcast_digest.mail import _html_to_text
from podcast_digest.youtube import canonical_watch_url, extract_video_id

logger = logging.getLogger(__name__)

_SNAPSHOT_LOCK = threading.Lock()
_SNAPSHOT_DIR: Path | None = None
_SNAPSHOT_KEY: tuple[object, ...] | None = None
_SNAPSHOT_DB: Path | None = None
_SNAPSHOT_GEN = 0
_SNAPSHOT_REFS: dict[Path, int] = {}
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def _cleanup_snapshot_dir() -> None:
    global _SNAPSHOT_DIR
    if _SNAPSHOT_DIR is not None:
        shutil.rmtree(_SNAPSHOT_DIR, ignore_errors=True)
        _SNAPSHOT_DIR = None


atexit.register(_cleanup_snapshot_dir)


def _stat_id(st: os.stat_result) -> tuple[int, int, int, int, int]:
    return (st.st_dev, st.st_ino, st.st_ctime_ns, st.st_mtime_ns, st.st_size)


def _db_fingerprint(db_path: Path) -> tuple[str, tuple[int, int, int, int, int], tuple[int, int, int, int, int] | None]:
    db_id = _stat_id(db_path.stat())
    wal = db_path.with_name(db_path.name + "-wal")
    try:
        wal_id: tuple[int, int, int, int, int] | None = _stat_id(wal.stat())
    except FileNotFoundError:
        wal_id = None
    return (str(db_path.resolve()), db_id, wal_id)


def _prune_unreferenced_snapshots_locked(keep: Path | None) -> None:
    if _SNAPSHOT_DIR is None or not _SNAPSHOT_DIR.exists():
        return
    for child in list(_SNAPSHOT_DIR.iterdir()):
        if child == keep:
            continue
        if _SNAPSHOT_REFS.get(child, 0) == 0:
            shutil.rmtree(child, ignore_errors=True)
            _SNAPSHOT_REFS.pop(child, None)


def _acquire_snapshot(db_path: Path, *, force: bool = False) -> tuple[Path, Path]:
    """Return ``(gen_dir, snapshot_db_path)`` with ``gen_dir`` pinned by refcount.

    Copies ``state.db`` + optional ``-wal`` (never ``-shm``) and verifies the
    stat fingerprint did not change during copy; raises ``sqlite3.OperationalError``
    if the writer keeps mutating across all retries rather than caching stale bytes.
    """
    global _SNAPSHOT_DIR, _SNAPSHOT_KEY, _SNAPSHOT_DB, _SNAPSHOT_GEN
    with _SNAPSHOT_LOCK:
        fp = _db_fingerprint(db_path)
        if not force and fp == _SNAPSHOT_KEY and _SNAPSHOT_DB is not None and _SNAPSHOT_DB.is_file():
            gen_dir = _SNAPSHOT_DB.parent
            _SNAPSHOT_REFS[gen_dir] = _SNAPSHOT_REFS.get(gen_dir, 0) + 1
            return gen_dir, _SNAPSHOT_DB
        if _SNAPSHOT_DIR is None or not _SNAPSHOT_DIR.exists():
            _SNAPSHOT_DIR = Path(tempfile.mkdtemp(prefix="podcast-mcp-db-"))
        for _attempt in range(3):
            before = _db_fingerprint(db_path)
            _SNAPSHOT_GEN += 1
            gen_dir = _SNAPSHOT_DIR / f"snap-{_SNAPSHOT_GEN}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            dst = gen_dir / db_path.name
            try:
                shutil.copy2(db_path, dst)
                wal = db_path.with_name(db_path.name + "-wal")
                if before[2] is not None:
                    shutil.copy2(wal, gen_dir / wal.name)
            except OSError:
                shutil.rmtree(gen_dir, ignore_errors=True)
                continue
            after = _db_fingerprint(db_path)
            if after == before:
                _SNAPSHOT_KEY = after
                _SNAPSHOT_DB = dst
                _SNAPSHOT_REFS[gen_dir] = _SNAPSHOT_REFS.get(gen_dir, 0) + 1
                _prune_unreferenced_snapshots_locked(gen_dir)
                return gen_dir, dst
            shutil.rmtree(gen_dir, ignore_errors=True)
        raise sqlite3.OperationalError("read-only WAL snapshot unstable while writer active")


def _release_snapshot(gen_dir: Path) -> None:
    with _SNAPSHOT_LOCK:
        remaining = _SNAPSHOT_REFS.get(gen_dir, 1) - 1
        if remaining <= 0:
            _SNAPSHOT_REFS.pop(gen_dir, None)
        else:
            _SNAPSHOT_REFS[gen_dir] = remaining
        keep = _SNAPSHOT_DB.parent if _SNAPSHOT_DB is not None else None
        _prune_unreferenced_snapshots_locked(keep)


@contextlib.contextmanager
def _open_readonly(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Yield a read-only SQLite connection, even on a read-only WAL mount."""
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        if conn is not None:
            conn.close()
        snap_conn: sqlite3.Connection | None = None
        gen_dir: Path | None = None
        for attempt in range(2):
            gen_dir, snap = _acquire_snapshot(db_path, force=(attempt > 0))
            try:
                snap_conn = sqlite3.connect(snap)
                snap_conn.row_factory = sqlite3.Row
                snap_conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
                break
            except sqlite3.DatabaseError:
                if snap_conn is not None:
                    snap_conn.close()
                    snap_conn = None
                _release_snapshot(gen_dir)
                gen_dir = None
                if attempt == 1:
                    raise
        assert snap_conn is not None and gen_dir is not None
        try:
            yield snap_conn
        finally:
            snap_conn.close()
            _release_snapshot(gen_dir)
        return
    try:
        yield conn
    finally:
        conn.close()


def _resolve_transcript_paths(
    data_dir: Path, recorded_path: str | None,
) -> tuple[Path | None, Path | None]:
    """Resolve `.clean.md` and `.txt` safely within ``data_dir / 'transcripts'``."""
    if not recorded_path:
        return None, None
    transcripts_root = (data_dir / "transcripts").resolve()
    raw_candidate = Path(recorded_path)
    candidates: list[Path] = []
    if raw_candidate.is_absolute():
        candidates.append(raw_candidate)
        parts = raw_candidate.parts
        if "transcripts" in parts:
            idx = len(parts) - 1 - parts[::-1].index("transcripts")
            rel = Path(*parts[idx + 1:])
            candidates.append(transcripts_root / rel)
    else:
        candidates.append(transcripts_root / raw_candidate)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_relative_to(transcripts_root):
            continue
        clean = resolved.with_suffix(".clean.md")
        raw = resolved if resolved.is_file() else None
        clean_file = clean if clean.is_file() else None
        if clean_file or raw:
            return clean_file, raw
    return None, None


def _read_transcript(data_dir: Path, recorded_path: str | None) -> tuple[str | None, str | None]:
    """Return ``(kind, text)`` where kind is ``'cleaned'``, ``'raw'``, or ``None``."""
    clean_file, raw_file = _resolve_transcript_paths(data_dir, recorded_path)
    for kind, path in (("cleaned", clean_file), ("raw", raw_file)):
        if path is None:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return kind, text
        except PermissionError as exc:
            logger.warning("archive.transcript_unreadable path=%s err=%s", path, exc)
        except OSError:
            continue
    return None, None


def _has_transcript(data_dir: Path, recorded_path: str | None) -> bool:
    clean_file, raw_file = _resolve_transcript_paths(data_dir, recorded_path)
    return any(p is not None and os.access(p, os.R_OK) for p in (clean_file, raw_file))


def _db_path(data_dir: Path | None) -> Path | None:
    if data_dir is None:
        return None
    path = data_dir / "state.db"
    return path if path.is_file() else None


def _snippet(text: str, terms: list[str], *, radius: int = 160) -> str:
    lower = text.casefold()
    pos = -1
    for term in terms:
        idx = lower.find(term)
        if idx >= 0 and (pos == -1 or idx < pos):
            pos = idx
    if pos < 0:
        return " ".join(text[: radius * 2].split())
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    excerpt = " ".join(text[start:end].split())
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{excerpt}{suffix}"


def completed_oneoff_keys(data_dir: Path | None) -> set[tuple[str, str]]:
    """Return ``(feed_url, guid)`` pairs for emailed episodes so completed one-offs can be hidden."""
    db_path = _db_path(data_dir)
    if db_path is None:
        return set()
    with _open_readonly(db_path) as conn:
        rows = conn.execute(
            "SELECT feed_url, guid FROM episodes WHERE status='emailed'"
        ).fetchall()
    return {(str(r["feed_url"]), str(r["guid"])) for r in rows}


def oneoff_is_completed(
    source: str | None, value: str | None, completed: set[tuple[str, str]],
) -> bool:
    if not value or not completed:
        return False
    if source == "audio_url":
        guid = f"audio:{hashlib.sha1(value.encode()).hexdigest()[:16]}"
        return (value, guid) in completed
    if source == "youtube_video":
        try:
            video_id = extract_video_id(value)
        except Exception:
            return False
        return (canonical_watch_url(video_id), f"yt:video:{video_id}") in completed
    return False


def digest_status(data_dir: Path | None, *, limit: int = 10) -> dict[str, Any]:
    """Read-only pipeline status: counts, recent deliveries, in-progress, held, feed errors."""
    db_path = _db_path(data_dir)
    if db_path is None or data_dir is None:
        return {
            "available": False,
            "reason": "state.db is not available (set PODCAST_DIGEST_DATA_DIR)",
        }
    limit = max(1, min(limit, 100))
    with _open_readonly(db_path) as conn:
        stage_rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM episodes GROUP BY status"
        ).fetchall()
        counts: dict[str, int] = {str(r["status"]): int(r["n"]) for r in stage_rows}
        delivered_row = conn.execute(
            "SELECT COUNT(*) AS n FROM episodes WHERE status = 'emailed' AND emailed_at IS NOT NULL"
        ).fetchone()
        counts["delivered"] = int(delivered_row["n"]) if delivered_row else 0
        counts["suppressed"] = max(0, counts.get("emailed", 0) - counts["delivered"])
        failed_retired = conn.execute(
            "SELECT COUNT(*) AS n FROM episodes "
            "WHERE status != 'emailed' AND (attempts >= ? OR summarize_attempts >= ?)",
            (MAX_ATTEMPTS, SUMMARIZE_TRANSPORT_HARD_CAP),
        ).fetchone()
        counts["failed_retired"] = int(failed_retired["n"]) if failed_retired else 0

        held_rows = conn.execute(
            "SELECT operation_id, show, feed_url, guid, updated_at, last_error "
            "FROM digest_deliveries WHERE state='unknown' ORDER BY updated_at DESC"
        ).fetchall()
        counts["held_deliveries"] = len(held_rows)

        recent_rows = conn.execute(
            """
            SELECT e.feed_url, e.guid, e.title, e.published_at, e.duration_seconds,
                   e.emailed_at, e.transcript_path, COALESCE(f.name, e.feed_url) AS show
            FROM episodes e
            LEFT JOIN feeds f ON f.url = e.feed_url
            WHERE e.status = 'emailed' AND e.emailed_at IS NOT NULL
            ORDER BY e.emailed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        in_progress_rows = conn.execute(
            """
            SELECT e.feed_url, e.guid, e.title, e.status, e.attempts, e.summarize_attempts,
                   e.last_error, e.updated_at, COALESCE(f.name, e.feed_url) AS show,
                   (e.attempts >= ? OR e.summarize_attempts >= ?) AS is_retired
            FROM episodes e
            LEFT JOIN feeds f ON f.url = e.feed_url
            WHERE e.status != 'emailed'
            ORDER BY is_retired ASC, e.updated_at DESC
            LIMIT ?
            """,
            (MAX_ATTEMPTS, SUMMARIZE_TRANSPORT_HARD_CAP, limit),
        ).fetchall()

        feed_error_rows = conn.execute(
            """
            SELECT name, url, last_polled_at, last_error
            FROM feeds
            WHERE last_error IS NOT NULL AND last_error != ''
            ORDER BY last_polled_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return {
        "available": True,
        "counts": counts,
        "recent_delivered": [
            {
                "show": r["show"],
                "title": r["title"],
                "guid": r["guid"],
                "published_at": r["published_at"],
                "emailed_at": r["emailed_at"],
                "duration_seconds": r["duration_seconds"],
                "has_transcript": _has_transcript(data_dir, r["transcript_path"]),
            }
            for r in recent_rows
        ],
        "in_progress": [
            {
                "show": r["show"],
                "title": r["title"],
                "guid": r["guid"],
                "status": r["status"],
                "attempts": r["attempts"],
                "summarize_attempts": r["summarize_attempts"],
                "retired": bool(r["is_retired"]),
                "last_error": r["last_error"],
                "updated_at": r["updated_at"],
            }
            for r in in_progress_rows
        ],
        "held_deliveries": [dict(r) for r in held_rows],
        "feed_errors": [dict(r) for r in feed_error_rows],
    }


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _parse_query_terms(query: str) -> list[str]:
    """Parse search query into normalized terms/phrases, stripping outer quotes."""
    try:
        raw_terms = shlex.split(query)
    except ValueError:
        raw_terms = [t.strip('"').strip("'") for t in query.split()]
    return [t.casefold().strip() for t in raw_terms if t.strip()]


def _fts5_query(terms: list[str]) -> str:
    """Build a safe FTS5 AND query with each term/phrase double-quoted."""
    return " ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _search_via_fts5(
    conn: sqlite3.Connection,
    *,
    data_dir: Path,
    terms: list[str],
    feed_filter: str | None,
    limit: int,
) -> tuple[int, list[dict[str, Any]]] | None:
    """Query the FTS5 index when present and populated; return None to fall back."""
    # unicode61 does not segment CJK characters; fall back to substring scan.
    if any(_CJK_RE.search(t) for t in terms):
        return None
    try:
        has_rows = conn.execute("SELECT 1 FROM episode_search LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return None
    if has_rows is None:
        return None

    fts_query = _fts5_query(terms)
    where = "WHERE episode_search MATCH ?"
    params: list[Any] = [fts_query]
    if feed_filter:
        where += " AND lower(s.show) LIKE ? ESCAPE '\\'"
        params.append(f"%{_escape_like(feed_filter)}%")

    count_sql = f"SELECT COUNT(*) AS n FROM episode_search s {where}"
    sql = f"""
        SELECT s.feed_url, s.guid, s.show, s.title,
               e.published_at, e.emailed_at, e.duration_seconds,
               e.status, e.transcript_path,
               snippet(episode_search, 5, '\x01', '\x02', '…', 24) AS transcript_snippet,
               snippet(episode_search, 4, '\x01', '\x02', '…', 24) AS summary_snippet,
               bm25(episode_search, 0.0, 0.0, 4.0, 5.0, 2.0, 1.0) AS rank
        FROM episode_search s
        LEFT JOIN episodes e ON e.feed_url = s.feed_url AND e.guid = s.guid
        {where}
        ORDER BY rank ASC LIMIT ?
    """
    try:
        total_row = conn.execute(count_sql, params).fetchone()
        total_matches = int(total_row["n"]) if total_row else 0
        rows = conn.execute(sql, [*params, limit]).fetchall()
    except sqlite3.OperationalError:
        return None

    results: list[dict[str, Any]] = []
    for r in rows:
        clean_file, raw_file = _resolve_transcript_paths(data_dir, r["transcript_path"])
        t_kind = (
            "cleaned" if clean_file is not None and os.access(clean_file, os.R_OK)
            else ("raw" if raw_file is not None and os.access(raw_file, os.R_OK) else None)
        )
        t_raw = str(r["transcript_snippet"] or "")
        s_raw = str(r["summary_snippet"] or "")
        t_matched = "\x01" in t_raw
        s_matched = "\x01" in s_raw
        t_snip = t_raw.replace("\x01", "").replace("\x02", "").strip()
        s_snip = s_raw.replace("\x01", "").replace("\x02", "").strip()
        title_lower = f"{r['show']} {r['title']}".casefold()
        if all(term in title_lower for term in terms):
            match_source = "title"
            snippet_text = s_snip or t_snip or str(r["title"])
        elif s_matched and not t_matched:
            match_source = "summary"
            snippet_text = s_snip
        elif t_matched:
            match_source = "transcript"
            snippet_text = t_snip
        elif s_matched:
            match_source = "summary"
            snippet_text = s_snip
        elif any(term in title_lower for term in terms):
            match_source = "title"
            snippet_text = s_snip or t_snip or str(r["title"])
        else:
            match_source = "transcript"
            snippet_text = t_snip or s_snip
        results.append({
            "show": str(r["show"] or ""),
            "title": str(r["title"] or ""),
            "guid": r["guid"],
            "feed_url": r["feed_url"],
            "published_at": r["published_at"],
            "emailed_at": r["emailed_at"],
            "duration_seconds": r["duration_seconds"],
            "status": r["status"],
            "transcript_kind": t_kind,
            "match_source": match_source,
            "snippet": " ".join(snippet_text.split()),
        })
    return total_matches, results


def search_transcripts(
    data_dir: Path | None,
    *,
    query: str,
    feed: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search titles, summaries, and saved transcripts across episodes.

    Uses the SQLite FTS5 BM25 index when populated, falling back transparently
    to a linear scan on databases that have not been indexed yet or for CJK queries.
    """
    db_path = _db_path(data_dir)
    if db_path is None or data_dir is None:
        return {
            "available": False,
            "reason": "state.db is not available (set PODCAST_DIGEST_DATA_DIR)",
        }
    terms = _parse_query_terms(query)
    if not terms:
        return {"error": "query must not be empty"}
    limit = max(1, min(limit, 50))
    feed_filter = feed.casefold().strip() if feed else None

    with _open_readonly(db_path) as conn:
        fts_hit = _search_via_fts5(
            conn, data_dir=data_dir, terms=terms, feed_filter=feed_filter, limit=limit,
        )
        if fts_hit is not None:
            total_matches, fts_results = fts_hit
            return {
                "available": True,
                "engine": "fts5",
                "query": query,
                "total_matches": total_matches,
                "results": fts_results,
            }
        indexed_transcripts: dict[tuple[str, str], str] = {}
        with contextlib.suppress(sqlite3.OperationalError):
            for srow in conn.execute("SELECT feed_url, guid, transcript FROM episode_search"):
                indexed_transcripts[(str(srow["feed_url"]), str(srow["guid"]))] = str(
                    srow["transcript"] or ""
                )
        rows = conn.execute(
            """
            SELECT e.feed_url, e.guid, e.title, e.published_at, e.duration_seconds,
                   e.status, e.emailed_at, e.summary_html, e.transcript_path,
                   COALESCE(f.name, e.feed_url) AS show
            FROM episodes e
            LEFT JOIN feeds f ON f.url = e.feed_url
            WHERE e.summary_html IS NOT NULL OR e.transcript_path IS NOT NULL
            ORDER BY COALESCE(e.emailed_at, e.published_at, e.updated_at) DESC
            """
        ).fetchall()
        seen_keys = {(str(r["feed_url"]), str(r["guid"])) for r in rows}
        missing_keys = [k for k in indexed_transcripts if k not in seen_keys]
        if missing_keys:
            rows = list(rows)
            for m_feed, m_guid in missing_keys:
                extra = conn.execute(
                    """
                    SELECT e.feed_url, e.guid, e.title, e.published_at, e.duration_seconds,
                           e.status, e.emailed_at, e.summary_html, e.transcript_path,
                           COALESCE(f.name, e.feed_url) AS show
                    FROM episodes e
                    LEFT JOIN feeds f ON f.url = e.feed_url
                    WHERE e.feed_url = ? AND e.guid = ?
                    """,
                    (m_feed, m_guid),
                ).fetchone()
                if extra is not None:
                    rows.append(extra)

    matches: list[tuple[int, dict[str, Any]]] = []
    for r in rows:
        show = str(r["show"] or "")
        if feed_filter and feed_filter not in show.casefold():
            continue
        title = str(r["title"] or "")
        summary_text = _html_to_text(r["summary_html"]) if r["summary_html"] else ""
        t_kind, t_text = _read_transcript(data_dir, r["transcript_path"])
        t_text = t_text or indexed_transcripts.get((str(r["feed_url"]), str(r["guid"])), "")

        combined_lower = "\n".join((show, title, summary_text, t_text)).casefold()
        if not all(term in combined_lower for term in terms):
            continue

        title_lower = f"{show} {title}".casefold()
        summary_lower = summary_text.casefold()
        score = 0
        match_source = "transcript"
        snippet_source = t_text
        if any(term in title_lower for term in terms):
            score += 5
            match_source = "title"
            snippet_source = summary_text or t_text or title
        if summary_text and all(term in summary_lower for term in terms):
            score += 3
            if match_source == "transcript":
                match_source = "summary"
                snippet_source = summary_text
        if t_text:
            t_lower = t_text.casefold()
            score += min(10, sum(t_lower.count(term) for term in terms))
            if match_source == "transcript":
                snippet_source = t_text

        matches.append((
            score,
            {
                "show": show,
                "title": title,
                "guid": r["guid"],
                "feed_url": r["feed_url"],
                "published_at": r["published_at"],
                "emailed_at": r["emailed_at"],
                "duration_seconds": r["duration_seconds"],
                "status": r["status"],
                "transcript_kind": t_kind,
                "match_source": match_source,
                "snippet": _snippet(snippet_source, terms),
            },
        ))

    matches.sort(key=lambda item: item[0], reverse=True)
    return {
        "available": True,
        "query": query,
        "total_matches": len(matches),
        "results": [item[1] for item in matches[:limit]],
    }


def get_episode(
    data_dir: Path | None,
    *,
    query_or_guid: str,
    feed: str | None = None,
    include_transcript: bool = True,
    max_chars: int = 40_000,
    offset: int = 0,
) -> dict[str, Any]:
    """Fetch one episode's metadata, summary, and (optionally sliced) transcript."""
    db_path = _db_path(data_dir)
    if db_path is None or data_dir is None:
        return {
            "available": False,
            "reason": "state.db is not available (set PODCAST_DIGEST_DATA_DIR)",
        }
    needle = query_or_guid.strip()
    if not needle:
        return {"error": "query_or_guid must not be empty"}
    max_chars = max(1_000, min(max_chars, 200_000))
    offset = max(0, offset)
    feed_filter = feed.strip() if feed else None

    def _pick_by_feed(rows: list[sqlite3.Row]) -> sqlite3.Row | None:
        if not rows:
            return None
        if not feed_filter:
            return rows[0]
        needle_cf = feed_filter.casefold()
        exact = [
            r for r in rows
            if str(r["feed_url"]) == feed_filter or str(r["show"] or "").casefold() == needle_cf
        ]
        if exact:
            return exact[0]
        partial = [
            r for r in rows
            if needle_cf in str(r["show"] or "").casefold()
        ]
        return partial[0] if partial else None

    with _open_readonly(db_path) as conn:
        guid_candidates = conn.execute(
            """
            SELECT e.feed_url, e.guid, COALESCE(f.name, e.feed_url) AS show
            FROM episodes e
            LEFT JOIN feeds f ON f.url = e.feed_url
            WHERE e.guid = ?
            ORDER BY COALESCE(e.emailed_at, e.updated_at) DESC
            """,
            (needle,),
        ).fetchall()
        chosen = _pick_by_feed(guid_candidates)
        if chosen is None:
            like = f"%{_escape_like(needle)}%"
            like_candidates = conn.execute(
                """
                SELECT e.feed_url, e.guid, COALESCE(f.name, e.feed_url) AS show
                FROM episodes e
                LEFT JOIN feeds f ON f.url = e.feed_url
                WHERE e.title LIKE ? ESCAPE '\\' OR f.name LIKE ? ESCAPE '\\'
                ORDER BY COALESCE(e.emailed_at, e.published_at, e.updated_at) DESC
                """,
                (like, like),
            ).fetchall()
            chosen = _pick_by_feed(like_candidates)
        row = (
            conn.execute(
                """
                SELECT e.*, COALESCE(f.name, e.feed_url) AS show
                FROM episodes e
                LEFT JOIN feeds f ON f.url = e.feed_url
                WHERE e.feed_url = ? AND e.guid = ?
                """,
                (chosen["feed_url"], chosen["guid"]),
            ).fetchone()
            if chosen is not None else None
        )

    if row is None:
        return {"available": True, "found": False, "query_or_guid": query_or_guid}

    t_kind, t_text = _read_transcript(data_dir, row["transcript_path"])
    summary_text = _html_to_text(row["summary_html"]).strip() if row["summary_html"] else None
    result: dict[str, Any] = {
        "available": True,
        "found": True,
        "show": row["show"],
        "title": row["title"],
        "guid": row["guid"],
        "feed_url": row["feed_url"],
        "published_at": row["published_at"],
        "emailed_at": row["emailed_at"],
        "duration_seconds": row["duration_seconds"],
        "status": row["status"],
        "summary": summary_text,
        "transcript_kind": t_kind,
        "transcript_total_chars": len(t_text) if t_text is not None else 0,
    }
    if include_transcript and t_text is not None:
        sliced = t_text[offset : offset + max_chars]
        result["transcript_offset"] = offset
        result["transcript_truncated"] = (offset + len(sliced)) < len(t_text)
        result["transcript"] = sliced
    elif include_transcript:
        result["transcript_offset"] = 0
        result["transcript_truncated"] = False
        result["transcript"] = None
    return result
