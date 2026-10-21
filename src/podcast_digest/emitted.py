"""Shared delivery log: the episodes already emailed, in an append-only file.

An instance records every episode it emails to this file; every instance reads
it and treats those episodes as already sent, so none of them re-emails what
another delivered. Typical uses: two instances sharing one replicated copy (one
active, one standby), or moving to a new host without re-sending the backlog.
Replicate the file however suits you (a shared volume, a git-synced folder
committed by a host-side job); the container only *writes* it.

This is the shared complement to ``state.db``, which stays local: state.db
holds the full per-stage status machine; this log carries only the fact that
matters across instances — that an episode was emailed.

Master off-switch: an empty path → a disabled no-op log (the single-instance
default), exactly like ``PODCAST_DIGEST_WEBHOOK_URL``.

Dedup keys — an episode is "already emitted" if ANY matches:
  - ``(feed_url, guid)`` — the exact key, recorded for every normal send.
  - a globally strong GUID — UUIDs, long hexadecimal content IDs, absolute
    URL GUIDs, and explicitly namespaced platform IDs. These identifiers are
    self-namespacing or carry enough entropy to be safe across shows.
  - ``(norm(show), guid)`` — covers two instances resolving the same show to
    different feed URLs: the GUID comes from the feed item itself, so it's
    instance-independent. Scoped by show because RSS GUIDs are only unique within
    a feed — a global guid match would let show B's episode be suppressed by
    show A's identical guid string (bare numeric/slug guids are common).
  - ``(norm(show), norm(title))`` — a FALLBACK key consulted only against
    records that carry no GUID. This lets the log be *seeded from an
    independent record of what was sent* (e.g. the digest emails in a
    mailbox, which carry show+title but no GUID — see ``record_title``).
    Records that DO carry a GUID never gate by title: a show legitimately
    reusing a title across episodes (recurring "Mailbag" segments, yearly
    recaps) has distinct GUIDs, and letting an old GUID'd record's title
    suppress a new GUID'd episode silently dark-holes those forever.

Accepted tradeoff: a publisher that REGENERATES its item GUIDs between two
instances' fetches defeats both guid keys and (for GUID'd records) the title
key — worst case one duplicate digest. Chosen deliberately over
the old any-record-title gate, which silently ate recurring-title episodes.

Format: append-only TSV, one record per line::

    <feed_url>\t<guid>\t<emitted_at_iso>\t<show>\t<title>
        [\t<content_sha256>\t<minhash_v1>\t<token_count>]

Older 3-column lines (no show/title) and seeded lines with an empty
feed_url/guid (title-key only) are both tolerated. Content columns are optional,
so old versions can read enriched records and new ones can read old history.
Reads are union-on-read and skip blank/short lines, so a git merge that
concatenates two instances' appends is always safe.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .dedupe import ContentCandidate, TranscriptFingerprint

logger = logging.getLogger(__name__)


# A trailing duration suffix as produced by pipeline._fmt_duration: "(44m)" or
# "(1h20m)" (or a bare "(1h)"). Requires a real unit letter adjacent to digits so
# it strips ONLY a duration — NOT a year "(2025)", a sequence "(1)"/"(2)", or a
# qualifier "(Q3)". The old `\(\d+\s*h?\s*\d*\s*m?\)` matched any trailing
# parenthesized number, collapsing "Year in Review (2024)" and "(2025)" to the
# same key → the second episode silently deduped away forever.
_DURATION_SUFFIX_RE = re.compile(
    r"\(\s*(?:\d+\s*h\s*)?\d+\s*m\s*\)\s*$|\(\s*\d+\s*h\s*\)\s*$", re.I
)
_UUID_RE = re.compile(
    r"^(?:urn:uuid:)?"
    r"([0-9a-f]{8})-([0-9a-f]{4})-([1-8][0-9a-f]{3})-"
    r"([89ab][0-9a-f]{3})-([0-9a-f]{12})$",
    re.I,
)
_LONG_HEX_RE = re.compile(r"^[0-9a-f]{24,64}$", re.I)
_YOUTUBE_VIDEO_RE = re.compile(r"^yt:video:([A-Za-z0-9_-]{6,})$")


def _strong_guid_key(guid: str | None) -> str | None:
    """Return a globally safe identity for a strong RSS/platform GUID.

    RSS only promises GUID uniqueness within a feed, so arbitrary slugs and
    integers remain show-scoped. Accepted forms either carry a namespace
    (URL / YouTube) or enough structured entropy that an accidental cross-show
    collision is negligible (UUID / 96+-bit hexadecimal content ID).
    """
    if not guid:
        return None
    value = guid.strip()
    if not value:
        return None

    uuid_match = _UUID_RE.fullmatch(value)
    if uuid_match:
        return "uuid:" + "-".join(part.lower() for part in uuid_match.groups())

    # Requiring a hexadecimal letter avoids treating long decimal sequence
    # numbers or timestamps as globally unique. Acast episode IDs are
    # 24-character Mongo-style object IDs and match this branch.
    if _LONG_HEX_RE.fullmatch(value) and any(c in "abcdefABCDEF" for c in value):
        return f"hex:{value.lower()}"

    youtube_match = _YOUTUBE_VIDEO_RE.fullmatch(value)
    if youtube_match:
        # YouTube IDs are case-sensitive; normalize only our namespace prefix.
        return f"yt:video:{youtube_match.group(1)}"

    try:
        parsed = urlsplit(value)
    except ValueError:
        # A malformed URL-shaped RSS GUID is simply not globally strong. It
        # must not make the whole feed fail while evaluating a dedup hint.
        return None
    if parsed.scheme.lower() in {"http", "https"} and parsed.hostname:
        # Absolute URL GUIDs are self-namespacing. Normalize only components
        # whose case is defined as insignificant and drop the client fragment;
        # keep path/query byte-for-byte because publishers may use them as ID.
        normalized = urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                parsed.query,
                "",
            )
        )
        return f"url:{normalized}"

    return None


def _norm(s: str | None) -> str:
    """Normalize show/title for the secondary key: casefold, drop a trailing
    ``(1h20m)``-style duration, strip non-alphanumerics, collapse runs.

    Stable across the small cosmetic differences between an email subject and the
    feed's own title (case, punctuation, a duration suffix) without being so loose
    it collides distinct episodes. Unicode letters and numbers are retained:
    reducing every CJK title to an empty string would make unrelated shows
    collide after a seeded record."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = _DURATION_SUFFIX_RE.sub("", s)  # trailing (44m)/(1h20m)/(1h)
    s = "".join(c if c.isalnum() else " " for c in s)
    return " ".join(s.split())


def _title_key(show: str | None, title: str | None) -> tuple[str, str]:
    """The one definition of the (show, title) key — record-time and
    lookup-time must agree byte-for-byte or records written under one shape
    are never matched under the other."""
    return (_norm(show), _norm(title))


class EmittedLog:
    """Append-only, git-replicated dedup log over emitted episodes.

    Construct via :meth:`load`. When the configured path is empty the log is
    *disabled*: :meth:`contains` is always ``False`` and the writers are no-ops,
    so single-instance deployments behave exactly as before.
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._guid_keys: set[tuple[str, str]] = set()
        # Strong GUIDs are safe across feed/show boundaries. Weak values remain
        # in the existing show-scoped index below.
        self._strong_guids: set[str] = set()
        self._content_by_episode: dict[tuple[str, str], ContentCandidate] = {}
        # (norm(show), guid) — cross-instance feed-URL drift, show-scoped so guid
        # strings can't collide across shows.
        self._show_guids: set[tuple[str, str]] = set()
        # (norm(show), norm(title)) -> "is a suppressor": True only when some
        # GUID-less record (seeded record) carries this title — those gate
        # suppression; GUID'd records' titles are known (False) for seeding
        # idempotency but never gate. One dict instead of two parallel sets so
        # the subset invariant can't be broken by a future edit.
        self._titles: dict[tuple[str, str], bool] = {}

    @property
    def enabled(self) -> bool:
        return self.path is not None

    @classmethod
    def load(cls, path: str | Path | None) -> EmittedLog:
        """Load the log from ``path``. Empty/None path → a disabled log.

        A missing file is fine (an enabled-but-empty log): the first record
        creates it. Malformed lines are skipped, not fatal — the log is a
        best-effort dedup hint, never a hard gate that could dark-hole a send.
        """
        if not path:
            return cls(None)
        log = cls(Path(path))
        assert log.path is not None
        if log.path.exists():
            try:
                for line in log.path.read_text(encoding="utf-8").splitlines():
                    log._index(line.split("\t"))
            except OSError as e:
                logger.warning("emitted.load_failed path=%s err=%s", log.path, e)
        return log

    def _index(self, parts: list[str]) -> None:
        """Fold one parsed TSV row into the in-memory key sets."""
        feed_url = parts[0] if len(parts) > 0 else ""
        guid = parts[1] if len(parts) > 1 else ""
        show = parts[3] if len(parts) > 3 else ""
        title = parts[4] if len(parts) > 4 else ""
        if guid:
            if feed_url:
                self._guid_keys.add((feed_url, guid))
            strong_guid = _strong_guid_key(guid)
            if strong_guid:
                self._strong_guids.add(strong_guid)
            if show:
                self._show_guids.add((_norm(show), guid))
        if show and title:
            # Only a GUID-less record (seeded record) makes a title a
            # SUPPRESSOR. A GUID'd record's show/title stay in the file for
            # humans and for seeding idempotency, but must not suppress a
            # future distinct-GUID episode that reuses the title.
            key = _title_key(show, title)
            self._titles[key] = self._titles.get(key, False) or not guid
        if feed_url and guid and len(parts) >= 8:
            try:
                fingerprint = TranscriptFingerprint.from_storage(
                    parts[5], parts[6], int(parts[7]),
                )
            except (TypeError, ValueError):
                # Identity columns remain useful even when optional enrichment
                # is corrupt or from an unsupported future version.
                return
            self._content_by_episode[(feed_url, guid)] = ContentCandidate(
                feed_url=feed_url,
                guid=guid,
                title=title,
                emitted_at=parts[2] if len(parts) > 2 else "",
                fingerprint=fingerprint,
            )

    def contains(
        self,
        feed_url: str,
        guid: str,
        show: str | None = None,
        title: str | None = None,
    ) -> bool:
        """True if this episode was already emitted.

        Matched by exact (feed_url, guid), by show-scoped guid (cross-instance
        feed-URL drift), or — as a fallback against GUID-less seeding records
        only — by normalized (show, title)."""
        if (feed_url, guid) in self._guid_keys:
            return True
        strong_guid = _strong_guid_key(guid)
        if strong_guid and strong_guid in self._strong_guids:
            return True
        if guid and show and (_norm(show), guid) in self._show_guids:
            return True
        return bool(show and title and self._titles.get(_title_key(show, title)))

    def has_title(self, show: str, title: str) -> bool:
        """True if ANY record (GUID'd or not) carries this (show, title).

        For seeding idempotency only — "have we already recorded this send in
        some form?" — which is a different question from :meth:`contains`'s
        "should this episode be suppressed?". The seeded record must use
        THIS check: gating it on `contains` would re-append (and thereby
        re-arm as suppressors) title-only lines for sends already recorded
        with a GUID."""
        return _title_key(show, title) in self._titles

    def record(
        self,
        feed_url: str,
        guid: str,
        emitted_at: str,
        show: str | None = None,
        title: str | None = None,
        fingerprint: TranscriptFingerprint | None = None,
    ) -> None:
        """Append one emitted episode (both keys). No-op when disabled or dup.

        The append is a single ``write()`` of one newline-terminated line, which
        POSIX guarantees atomic for small writes — so a concurrent reader never
        sees a torn line and the file stays merge-safe. A write failure is logged
        but never raised: the email already went out, and re-emailing on the next
        run (the worst case of a lost record) beats failing a successful send.
        """
        if self.path is None:
            return
        existing_content = self._content_by_episode.get((feed_url, guid))
        if (feed_url, guid) in self._guid_keys and (
            fingerprint is None
            or (
                existing_content is not None
                and existing_content.fingerprint == fingerprint
            )
        ):
            return
        columns = [feed_url, guid, emitted_at, show or "", title or ""]
        if fingerprint is not None:
            columns.extend(
                [
                    fingerprint.sha256,
                    fingerprint.serialize_minhash(),
                    str(fingerprint.token_count),
                ]
            )
        self._append(columns)

    def content_candidates(self, since: datetime) -> list[ContentCandidate]:
        """Return delivered fingerprints at or after the UTC cutoff."""
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)
        candidates: list[ContentCandidate] = []
        for candidate in self._content_by_episode.values():
            try:
                emitted_at = datetime.fromisoformat(
                    candidate.emitted_at.replace("Z", "+00:00")
                )
            except ValueError:
                continue
            if emitted_at.tzinfo is None:
                emitted_at = emitted_at.replace(tzinfo=UTC)
            if emitted_at >= since:
                candidates.append(candidate)
        return candidates

    def has_content(self, feed_url: str, guid: str) -> bool:
        """Whether this exact emitted episode already has content enrichment."""
        return (feed_url, guid) in self._content_by_episode

    def record_title(self, show: str, title: str, emitted_at: str = "") -> None:
        """Append a title-only seeding entry (no GUID) — used to seed the log
        from an independent record of sends (the mailbox). No-op when disabled or
        when this (show, title) is already known."""
        if self.path is None or not (show and title):
            return
        if _title_key(show, title) in self._titles:
            return
        self._append(["", "", emitted_at, show, title])

    def _append(self, cols: list[str]) -> None:
        line = "\t".join(c.replace("\t", " ").replace("\n", " ") for c in cols) + "\n"
        assert self.path is not None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
            self._index(cols)
        except OSError as e:
            logger.warning("emitted.append_failed path=%s err=%s", self.path, e)
