"""Accepted and rejected operations converge without duplicate mail."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from podcast_digest.db import Database
from podcast_digest.dedupe import find_content_match, fingerprint_transcript
from podcast_digest.delivery import DeliveryHeld, Envelope, reconcile_sent_item, send_digest
from podcast_digest.emitted import EmittedLog
from podcast_digest.mail import GraphMailer, MailError
from podcast_digest.pipeline import finalize_deliveries, run_tick

from .conftest import FEED_URL, FakeMailer
from .test_delivery import fake_graph_read, seed

GUID = "ep-3-guid"


@pytest.mark.parametrize("evidence", ["identity", "local_content", "shared_content"])
@pytest.mark.parametrize("unknown", [False, True])
async def test_positive_dedupe_suppresses_safe_retry_but_never_resolves_unknown(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, tmp_path,
    evidence, unknown,
):
    settings.PODCAST_DIGEST_EMITTED_LOG = str(tmp_path / "emitted.tsv")
    failure = RuntimeError("lost reply") if unknown else MailError("not submitted")
    await run_tick(settings=settings, feeds_file=feeds_file, db=db,
                   mailer=FakeMailer(fail=True, exc=failure))
    original = db.get_delivery(FEED_URL, GUID)
    ep = db.get_episode(FEED_URL, GUID)
    fingerprint = fingerprint_transcript(" ".join(f"synthetic{i}" for i in range(100)))
    db.set_content_fingerprint(FEED_URL, GUID, fingerprint)
    emitted = EmittedLog.load(settings.PODCAST_DIGEST_EMITTED_LOG)
    now = datetime.now(UTC).isoformat()
    if evidence == "identity":
        assert emitted.record(FEED_URL, GUID, now, feeds_file.feeds[0].display_show, ep.title)
        assert emitted.contains(FEED_URL, GUID)
    elif evidence == "shared_content":
        assert emitted.record("https://example.com/other.xml", "copy", now,
                              "Synthetic copy", "Same program", fingerprint)
    else:
        envelope = seed(db, guid="copy")
        db.set_content_fingerprint(FEED_URL, "copy", fingerprint)
        await send_digest(db, FakeMailer(), FEED_URL, "copy", envelope)
        cutoff = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        assert find_content_match(fingerprint, db.delivered_content_since(cutoff)) is not None
    mailer = FakeMailer()
    stats = await run_tick(settings=settings, feeds_file=feeds_file, db=db, mailer=mailer)
    assert mailer.sent == [] and stats["emailed"] == 0
    saved = db.get_delivery(FEED_URL, GUID)
    assert (saved.operation_id, saved.envelope, saved.attempts, saved.receipt) == (
        original.operation_id, original.envelope, 1, None,
    )
    ep = db.get_episode(FEED_URL, GUID)
    assert ep.emailed_at is None
    if unknown:
        assert saved.state == "unknown" and saved.suppression_reason is None
        assert ep.status == "summarized"
    else:
        reason = "duplicate_identity" if evidence == "identity" else "duplicate_transcript_exact"
        assert (saved.state, saved.suppression_reason) == ("not_submitted", reason)
        assert (ep.status, ep.terminal_reason) == ("emailed", reason)
        with pytest.raises(DeliveryHeld):
            await send_digest(db, mailer, FEED_URL, GUID)
        assert mailer.sent == []


async def test_reconciled_delivery_finalizes_on_tick_and_suppresses_standby(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, monkeypatch, tmp_path,
):
    graph = GraphMailer(tenant_id="test", client_id="test", client_secret="test",
                        sender="bot@example.com")

    class LostReply(FakeMailer):
        def send(self, **kwargs):
            super().send(**kwargs)
            raise RuntimeError("reply lost after acceptance")

    first = LostReply()
    first.delivery_target = graph.delivery_target
    settings.PODCAST_DIGEST_EMITTED_LOG = str(tmp_path / "emitted.tsv")
    await run_tick(settings=settings, feeds_file=feeds_file, db=db, mailer=first)
    row = db.get_delivery(FEED_URL, GUID)
    frozen = Envelope.parse(row.envelope)
    audio = Path(db.get_episode(FEED_URL, GUID).audio_path)
    assert audio.exists() and len(first.sent) == 1
    fake_graph_read(monkeypatch, graph, {
        "id": "synthetic-sent-id", "isDraft": False,
        "sentDateTime": "2026-11-08T00:00:00Z", "subject": frozen.subject,
        "from": {"emailAddress": {"address": "bot@example.com"}},
        "toRecipients": [{"emailAddress": {"address": frozen.to}}],
        "internetMessageHeaders": [{"name": "x-podcast-digest-operation", "value": row.operation_id}],
    })
    assert reconcile_sent_item(db, graph, row, "synthetic-sent-id")
    assert db.delivery_by_id(row.operation_id).finalized_at is None
    # Simulate exit immediately after the atomic acceptance commit. Even a
    # tick with the feed removed must finish publication without a mailer.
    await run_tick(settings=settings, feeds_file=feeds_file.model_copy(update={"feeds": []}),
                   db=db, mailer=None)
    assert db.delivery_by_id(row.operation_id).finalized_at is not None
    log = EmittedLog.load(settings.PODCAST_DIGEST_EMITTED_LOG)
    assert log.contains(FEED_URL, GUID, feeds_file.feeds[0].display_show, "")
    assert not audio.exists() and db.get_episode(FEED_URL, GUID).audio_path is None
    standby_settings = settings.model_copy(update={"PODCAST_DIGEST_DATA_DIR": tmp_path / "standby"})
    standby_settings.ensure_dirs()
    standby_db = Database(standby_settings.db_path)
    try:
        standby = FakeMailer()
        await run_tick(settings=standby_settings, feeds_file=feeds_file, db=standby_db, mailer=standby)
        assert standby.sent == [] and len(first.sent) == 1
    finally:
        standby_db.close()


@pytest.mark.parametrize("fault", ["append", "cleanup", "watermark"])
async def test_accepted_finalization_retries_after_local_failure_without_resending(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, monkeypatch, tmp_path, fault,
):
    log_path = tmp_path / "emitted.tsv"
    settings.PODCAST_DIGEST_EMITTED_LOG = str(log_path)
    with monkeypatch.context() as failure:
        if fault == "append":
            original_open = Path.open

            def open_path(path, *args, **kwargs):
                if path == log_path:
                    raise OSError("synthetic log failure")
                return original_open(path, *args, **kwargs)

            failure.setattr(Path, "open", open_path)
        elif fault == "cleanup":
            failure.setattr("podcast_digest.pipeline.audio_cleanup", lambda path: None)
        else:
            def failed_watermark(*args):
                raise sqlite3.OperationalError("synthetic finalization commit failure")

            failure.setattr(db, "finalize_delivery", failed_watermark)
        first = FakeMailer()
        await run_tick(settings=settings, feeds_file=feeds_file, db=db, mailer=first)
    row = db.get_delivery(FEED_URL, GUID)
    assert row.state == "accepted" and row.finalized_at is None
    assert db.get_episode(FEED_URL, GUID).terminal_reason == "sent"
    later = FakeMailer()
    await run_tick(settings=settings, feeds_file=feeds_file, db=db, mailer=later)
    assert len(first.sent) == 1 and later.sent == []
    assert db.delivery_by_id(row.operation_id).finalized_at is not None
    assert db.get_episode(FEED_URL, GUID).audio_path is None
    assert EmittedLog.load(log_path).contains(FEED_URL, GUID)
    # A crash after the append but before finalization must not append twice.
    assert len(log_path.read_text().splitlines()) == 1


def test_suppression_rolls_back_with_episode_and_cannot_close_unknown(db, monkeypatch):
    envelope = seed(db)
    row = db.claim_delivery(FEED_URL, "synthetic-digest", target=FakeMailer.delivery_target,
                            envelope=envelope.serialize())
    with pytest.raises(ValueError, match="proven non-submission"):
        db.suppress_delivery(row.operation_id, "duplicate_identity")
    db.delivery_not_submitted(row.operation_id, "rejected")
    with monkeypatch.context() as failure:
        def failed_status(**kwargs):
            raise sqlite3.OperationalError("synthetic failure")

        failure.setattr(db, "set_status", failed_status)
        with pytest.raises(sqlite3.OperationalError):
            db.suppress_delivery(row.operation_id, "duplicate_identity")
    assert db.delivery_by_id(row.operation_id).suppression_reason is None
    assert db.get_episode(FEED_URL, "synthetic-digest").status == "summarized"


async def test_existing_journal_migration_retains_intent_and_pending_acceptance(db):
    envelope = seed(db)
    await send_digest(db, FakeMailer(), FEED_URL, "synthetic-digest", envelope)
    original = db.get_delivery(FEED_URL, "synthetic-digest")
    for column in ("suppression_reason", "show", "finalized_at"):
        db._conn.execute(f"ALTER TABLE digest_deliveries DROP COLUMN {column}")
    reopened = Database(db.path)
    try:
        saved = reopened.delivery_by_id(original.operation_id)
        assert (saved.envelope, saved.receipt, saved.state) == (
            original.envelope, original.receipt, "accepted",
        )
        assert saved.suppression_reason is None and saved.finalized_at is None
        finalize_deliveries(reopened, EmittedLog(None))
        assert reopened.delivery_by_id(original.operation_id).finalized_at is not None
    finally:
        reopened.close()


async def test_failed_log_sync_stays_pending_even_when_a_later_load_sees_the_row(
    db, monkeypatch, tmp_path,
):
    envelope = seed(db)
    await send_digest(db, FakeMailer(), FEED_URL, "synthetic-digest", envelope)
    row = db.get_delivery(FEED_URL, "synthetic-digest")
    path = tmp_path / "emitted.tsv"
    with monkeypatch.context() as failure:
        def failed_sync(fd):
            raise OSError("synthetic sync failure")

        failure.setattr("podcast_digest.emitted.os.fsync", failed_sync)
        finalize_deliveries(db, EmittedLog.load(path))
        assert path.exists()
        loaded = EmittedLog.load(path)
        assert loaded.contains(FEED_URL, "synthetic-digest")
        finalize_deliveries(db, loaded)
        assert db.delivery_by_id(row.operation_id).finalized_at is None
    finalize_deliveries(db, EmittedLog.load(path))
    assert db.delivery_by_id(row.operation_id).finalized_at is not None
    assert len(path.read_text().splitlines()) == 1
