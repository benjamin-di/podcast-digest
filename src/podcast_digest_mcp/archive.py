"""Read-only access to state.db and saved transcripts for the MCP server.

The MCP server never mutates SQLite or transcript files. When `/data` is
bind-mounted read-only in Docker and `state.db` uses WAL mode, SQLite cannot
create/map `-shm` in the read-only directory; in that case we snapshot
`state.db` (plus `-wal` if present) into a private temporary directory so
SQLite replays committed WAL frames cleanly without touching `/data`.
"""
from __future__ import annotations

import contextlib
import hashlib
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from podcast_digest.db import MAX_ATTEMPTS, SUMMARIZE_TRANSPORT_HARD_CAP
from podcast_digest.mail import _html_to_text
from podcast_digest.youtube import canonical_watch_url, extract_video_id


@contextlib.contextmanager
def _open_readonly(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Yield a read-only SQLite connection, even on a read-only WAL mount."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        with tempfile.TemporaryDirectory(prefix="podcast-mcp-db-") as tmpdir:
            tmp = Path(tmpdir)
            dst = tmp / db_path.name
            shutil.copy2(db_path, dst)
            for suffix in ("-wal", "-shm"):
                sidecar = db_path.with_name(db_path.name + suffix)
                if sidecar.is_file():
                    with contextlib.suppress(OSError):
                        shutil.copy2(sidecar, tmp / sidecar.name)
            conn = sqlite3.connect(dst)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
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
    if clean_file is not None:
        with contextlib.suppress(OSError):
            text = clean_file.read_text(encoding="utf-8")
            if text.strip():
                return "cleaned", text
    if raw_file is not None:
        with contextlib.suppress(OSError):
            text = raw_file.read_text(encoding="utf-8")
            if text.strip():
                return "raw", text
    return None, None


def _has_transcript(data_dir: Path, recorded_path: str | None) -> bool:
    clean_file, raw_file = _resolve_transcript_paths(data_dir, recorded_path)
    return clean_file is not None or raw_file is not None


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
    limit = max(1, min(limit, 50))
    with _open_readonly(db_path) as conn:
        stage_rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM episodes GROUP BY status"
        ).fetchall()
        counts: dict[str, int] = {str(r["status"]): int(r["n"]) for r in stage_rows}
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
                   e.last_error, e.updated_at, COALESCE(f.name, e.feed_url) AS show
            FROM episodes e
            LEFT JOIN feeds f ON f.url = e.feed_url
            WHERE e.status != 'emailed'
            ORDER BY e.updated_at DESC
            LIMIT ?
            """,
            (limit,),
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
                "last_error": r["last_error"],
                "updated_at": r["updated_at"],
            }
            for r in in_progress_rows
        ],
        "held_deliveries": [dict(r) for r in held_rows],
        "feed_errors": [dict(r) for r in feed_error_rows],
    }


def search_transcripts(
    data_dir: Path | None,
    *,
    query: str,
    feed: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search titles, summaries, and saved transcripts across episodes."""
    db_path = _db_path(data_dir)
    if db_path is None or data_dir is None:
        return {
            "available": False,
            "reason": "state.db is not available (set PODCAST_DIGEST_DATA_DIR)",
        }
    terms = [t.casefold() for t in query.split() if t.strip()]
    if not terms:
        return {"error": "query must not be empty"}
    limit = max(1, min(limit, 50))
    feed_filter = feed.casefold().strip() if feed else None

    with _open_readonly(db_path) as conn:
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

    matches: list[tuple[int, dict[str, Any]]] = []
    for r in rows:
        show = str(r["show"] or "")
        if feed_filter and feed_filter not in show.casefold():
            continue
        title = str(r["title"] or "")
        summary_text = _html_to_text(r["summary_html"]) if r["summary_html"] else ""
        t_kind, t_text = _read_transcript(data_dir, r["transcript_path"])
        t_text = t_text or ""

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

    with _open_readonly(db_path) as conn:
        row = conn.execute(
            """
            SELECT e.*, COALESCE(f.name, e.feed_url) AS show
            FROM episodes e
            LEFT JOIN feeds f ON f.url = e.feed_url
            WHERE e.guid = ?
            ORDER BY COALESCE(e.emailed_at, e.updated_at) DESC
            LIMIT 1
            """,
            (needle,),
        ).fetchone()
        if row is None:
            like = f"%{needle}%"
            row = conn.execute(
                """
                SELECT e.*, COALESCE(f.name, e.feed_url) AS show
                FROM episodes e
                LEFT JOIN feeds f ON f.url = e.feed_url
                WHERE e.title LIKE ? OR f.name LIKE ?
                ORDER BY COALESCE(e.emailed_at, e.published_at, e.updated_at) DESC
                LIMIT 1
                """,
                (like, like),
            ).fetchone()

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
