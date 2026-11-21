"""Durable operation and provider observation behavior, using synthetic mail only."""
from __future__ import annotations

import io
import json
import sqlite3
import urllib.error

import pytest

from podcast_digest.db import Database
from podcast_digest.delivery import DeliveryHeld, Envelope, reconcile_sent_item, send_digest
from podcast_digest.mail import GraphMailer, MailAttachment, MailError, MailObservationError

from .conftest import FEED_URL, FakeMailer

GUID = "synthetic-digest"


def seed(db, guid=GUID, feed_url=FEED_URL):
    db.insert_episode(feed_url=feed_url, guid=guid, title="Synthetic episode",
                      published_at="2026-11-01T00:00:00Z", duration_seconds=3600,
                      audio_url="https://example.com/synthetic.mp3")
    db.set_status(feed_url=feed_url, guid=guid, status="summarized",
                  summary_html="<p>Saved summary</p>")
    return Envelope(to="reader@example.com", subject="Synthetic digest",
                    body_html="<p>Saved summary</p>",
                    attachments=(MailAttachment("episode.txt", b"saved bytes"),))


def graph():
    return GraphMailer(tenant_id="test", client_id="test", client_secret="test",
                       sender="bot@example.com")


def sent_message(operation):
    return {
        "id": "synthetic-sent-id", "isDraft": False,
        "sentDateTime": "2026-11-08T00:00:00Z", "subject": "Synthetic digest",
        "from": {"emailAddress": {"address": "bot@example.com"}},
        "toRecipients": [{"emailAddress": {"address": "reader@example.com"}}],
        "internetMessageHeaders": [
            {"name": "X-Podcast-Digest-Operation", "value": operation},
        ],
    }


def fake_graph_read(monkeypatch, mailer, payload):
    captured = []
    monkeypatch.setattr(mailer, "_get_token", lambda: "synthetic-token")

    def observe(req, **_kwargs):
        captured.append(req)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", observe)
    return captured


async def test_proven_non_submission_retries_identical_saved_operation(db):
    envelope = seed(db)
    with pytest.raises(MailError):
        await send_digest(db, FakeMailer(fail=True), FEED_URL, GUID, envelope)
    saved = db.get_delivery(FEED_URL, GUID)
    assert saved.state == "not_submitted" and saved.attempts == 1
    assert db.get_episode(FEED_URL, GUID).status == "summarized"
    mailer = FakeMailer()
    changed = Envelope(to="changed@example.com", subject="Changed", body_html="changed")
    await send_digest(db, mailer, FEED_URL, GUID, changed)
    [sent] = mailer.sent
    assert sent["to"] == envelope.to and sent["subject"] == envelope.subject
    assert sent["body_html"] == envelope.body_html
    assert sent["attachments"] == list(envelope.attachments)
    assert sent["operation_id"] == saved.operation_id
    completed = db.get_delivery(FEED_URL, GUID)
    assert completed.envelope == saved.envelope and completed.attempts == 2
    assert (completed.state, completed.receipt) == ("accepted", "transport:accepted")
    ep = db.get_episode(FEED_URL, GUID)
    assert (ep.status, ep.terminal_reason) == ("emailed", "sent")
    assert ep.emailed_at is not None


async def test_changed_target_never_receives_saved_operation(db):
    envelope = seed(db)
    with pytest.raises(MailError):
        await send_digest(db, FakeMailer(fail=True), FEED_URL, GUID, envelope)
    changed = FakeMailer()
    changed.delivery_target = "test:other@example.com"
    with pytest.raises(ValueError, match="target changed"):
        await send_digest(db, changed, FEED_URL, GUID)
    assert changed.sent == []
    assert db.get_episode(FEED_URL, GUID).status == "summarized"
    assert db.get_delivery(FEED_URL, GUID).state == "not_submitted"


async def test_unknown_exception_never_becomes_a_safe_retry(db):
    envelope = seed(db)
    with pytest.raises(DeliveryHeld):
        await send_digest(db, FakeMailer(fail=True, exc=RuntimeError("unknown")),
                          FEED_URL, GUID, envelope)
    later = FakeMailer()
    with pytest.raises(DeliveryHeld):
        await send_digest(db, later, FEED_URL, GUID)
    assert later.sent == []
    assert db.get_episode(FEED_URL, GUID).emailed_at is None
    assert db.get_delivery(FEED_URL, GUID).state == "unknown"


def test_receipt_and_episode_write_roll_back_together(db, monkeypatch):
    envelope = seed(db)
    row = db.claim_delivery(FEED_URL, GUID, target=FakeMailer.delivery_target,
                            envelope=envelope.serialize())

    def disk_failed(**_kwargs):
        raise sqlite3.OperationalError("synthetic disk error")

    monkeypatch.setattr(db, "set_status", disk_failed)
    with pytest.raises(sqlite3.OperationalError):
        db.complete_delivery(row.operation_id, "transport:accepted")
    saved = db.delivery_by_id(row.operation_id)
    assert saved.state == "unknown" and saved.receipt is None
    assert db.get_episode(FEED_URL, GUID).status == "summarized"
    assert db.get_episode(FEED_URL, GUID).emailed_at is None


def test_unknown_intent_cannot_be_claimed_or_marked_sent(db):
    envelope = seed(db)
    row = db.claim_delivery(FEED_URL, GUID, target=FakeMailer.delivery_target,
                            envelope=envelope.serialize())
    assert db.claim_delivery(FEED_URL, GUID, target=row.target, envelope="changed") is None
    with pytest.raises(ValueError, match="unresolved delivery"):
        db.set_status(feed_url=FEED_URL, guid=GUID, status="emailed", terminal_reason="sent")
    assert db.delivery_by_id(row.operation_id).envelope == row.envelope
    assert db.get_episode(FEED_URL, GUID).status == "summarized"


def test_backward_stage_preserves_summary_and_attempts(db):
    seed(db)
    db.mark_failed(FEED_URL, GUID, "synthetic failure")
    with pytest.raises(ValueError, match="backward stage"):
        db.set_status(feed_url=FEED_URL, guid=GUID, status="downloaded")
    ep = db.get_episode(FEED_URL, GUID)
    assert (ep.status, ep.attempts, ep.summary_html) == ("summarized", 1, "<p>Saved summary</p>")


def test_rekey_preserves_operation_and_payload_and_rejects_unsafe_collision(db):
    envelope = seed(db)
    db.upsert_feed(url=FEED_URL, name="Synthetic show")
    row = db.claim_delivery(FEED_URL, GUID, target=FakeMailer.delivery_target,
                            envelope=envelope.serialize())
    new_url = "https://example.com/moved.xml"
    assert db.rekey_feed(FEED_URL, new_url) == 1
    moved = db.get_delivery(new_url, GUID)
    assert moved.operation_id == row.operation_id and moved.envelope == row.envelope
    assert moved.state == "unknown"
    seed(db, feed_url=FEED_URL)
    # Trying to delete a colliding episode with a durable intent must roll the
    # whole re-key back rather than release its unknown external effect.
    with pytest.raises(sqlite3.IntegrityError):
        db.rekey_feed(new_url, FEED_URL)
    assert db.get_delivery(new_url, GUID).operation_id == row.operation_id
    assert db.get_episode(new_url, GUID).status == "summarized"


def test_two_connections_cannot_claim_one_unknown_intent(db):
    envelope = seed(db)
    first = db.claim_delivery(FEED_URL, GUID, target=FakeMailer.delivery_target,
                              envelope=envelope.serialize())
    other = Database(db.path)
    try:
        assert other.claim_delivery(FEED_URL, GUID, target=first.target,
                                    envelope="changed") is None
        assert other.delivery_by_id(first.operation_id).envelope == first.envelope
    finally:
        other.close()


def test_positive_sent_item_completes_held_operation_and_is_repeatable(db, monkeypatch):
    envelope = seed(db)
    mailer = graph()
    row = db.claim_delivery(FEED_URL, GUID, target=mailer.delivery_target,
                            envelope=envelope.serialize())
    calls = fake_graph_read(monkeypatch, mailer, sent_message(row.operation_id))
    assert reconcile_sent_item(db, mailer, row, "synthetic/sent+id") is True
    assert len(calls) == 1
    assert "/mailFolders/sentitems/messages/synthetic%2Fsent%2Bid?" in calls[0].full_url
    assert calls[0].get_method() == "GET" and calls[0].data is None
    completed = db.delivery_by_id(row.operation_id)
    assert completed.state == "accepted" and completed.receipt == "graph-sent-item:synthetic-sent-id"
    assert db.get_episode(FEED_URL, GUID).status == "emailed"
    assert reconcile_sent_item(db, mailer, completed, "synthetic/sent+id") is True
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["header", "no_header", "draft", "unsent", "sender",
                                    "recipients", "subject"])
def test_readback_without_positive_matching_evidence_remains_held(db, monkeypatch, change):
    envelope = seed(db)
    mailer = graph()
    row = db.claim_delivery(FEED_URL, GUID, target=mailer.delivery_target,
                            envelope=envelope.serialize())
    message = sent_message(row.operation_id)
    if change == "header":
        message["internetMessageHeaders"][0]["value"] = "another-operation"
    elif change == "no_header":
        message["internetMessageHeaders"] = []
    elif change == "draft":
        message["isDraft"] = True
    elif change == "unsent":
        message["sentDateTime"] = None
    elif change == "sender":
        message["from"]["emailAddress"]["address"] = "other@example.com"
    elif change == "recipients":
        message["toRecipients"] = []
    elif change == "subject":
        message["subject"] = "another digest"
    fake_graph_read(monkeypatch, mailer, message)
    assert reconcile_sent_item(db, mailer, row, "synthetic-id") is False
    assert db.delivery_by_id(row.operation_id).state == "unknown"
    assert db.get_episode(FEED_URL, GUID).emailed_at is None


@pytest.mark.parametrize("code", [404, 403, 503])
def test_missing_or_unreadable_receipt_never_releases_intent(db, monkeypatch, code):
    envelope = seed(db)
    mailer = graph()
    row = db.claim_delivery(FEED_URL, GUID, target=mailer.delivery_target,
                            envelope=envelope.serialize())
    monkeypatch.setattr(mailer, "_get_token", lambda: "synthetic-token")

    def unavailable(req, **_kwargs):
        raise urllib.error.HTTPError(req.full_url, code, "synthetic", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    if code == 404:
        assert reconcile_sent_item(db, mailer, row, "missing") is False
    else:
        with pytest.raises(MailObservationError, match=f"HTTP {code}"):
            reconcile_sent_item(db, mailer, row, "unreadable")
    assert db.delivery_by_id(row.operation_id).state == "unknown"
    assert db.get_episode(FEED_URL, GUID).status == "summarized"


def test_observation_transport_error_does_not_change_submission_certainty(db, monkeypatch):
    envelope = seed(db)
    mailer = graph()
    row = db.claim_delivery(FEED_URL, GUID, target=mailer.delivery_target,
                            envelope=envelope.serialize())
    monkeypatch.setattr(mailer, "_get_token", lambda: "synthetic-token")

    def disconnected(*_args, **_kwargs):
        raise ConnectionResetError("synthetic read failure")

    monkeypatch.setattr("urllib.request.urlopen", disconnected)
    with pytest.raises(MailObservationError):
        reconcile_sent_item(db, mailer, row, "synthetic")
    assert db.delivery_by_id(row.operation_id).state == "unknown"
    assert db.get_episode(FEED_URL, GUID).emailed_at is None
