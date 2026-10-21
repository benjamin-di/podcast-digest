"""Pipeline tests for the non-fatal post-email webhook push.

Wires `run_tick` with the webhook client faked (no HTTP) and asserts: emailed
episodes are pushed once with the right payload; per-feed opt-out and the master
off-switch are honored; pushes are idempotent; and a push failure never breaks
the digest and is retried on a later tick.
"""
from __future__ import annotations

import pytest

from podcast_digest.config import Feed, FeedsFile
from podcast_digest.pipeline import run_tick
from podcast_digest.webhook import WebhookError

from .conftest import FEED_URL, FakeMailer

NEWEST_GUID = "ep-3-guid"


class FakeIngestor:
    """Drop-in for WebhookClient — records pushes; can be made to fail."""

    ack_field = "task_id"

    def __init__(self, fail: bool = False):
        self.pushes: list = []
        self.fail = fail

    def push(self, ep) -> str:
        if self.fail:
            raise WebhookError("synthetic webhook failure")
        self.pushes.append(ep)
        return "task-test"


class MissingAckIngestor(FakeIngestor):
    """Model a client that returns no acknowledgement although one is required."""

    def push(self, ep) -> str:
        self.pushes.append(ep)
        return ""


@pytest.fixture
def use_ingestor(monkeypatch):
    """Force the pipeline to use a supplied FakeIngestor regardless of env."""
    def _install(ingestor: FakeIngestor) -> FakeIngestor:
        monkeypatch.setattr(
            "podcast_digest.pipeline._build_webhook", lambda settings: ingestor
        )
        return ingestor
    return _install


async def test_relevant_episode_pushed_after_email(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, use_ingestor,
):
    ing = use_ingestor(FakeIngestor())
    mailer = FakeMailer()

    stats = await run_tick(settings=settings, feeds_file=feeds_file,
                           db=db, mailer=mailer, dry_run=False)

    assert stats["emailed"] == 1
    assert stats["webhook_pushed"] == 1
    assert stats["webhook_failed"] == 0

    # Exactly one push, carrying the episode's payload.
    assert len(ing.pushes) == 1
    pushed = ing.pushes[0]
    assert pushed.show == "The Test Podcast"
    assert pushed.title.startswith("Episode 3")
    assert pushed.guid == NEWEST_GUID
    assert pushed.listen_url  # the audio/listen URL
    assert "Test summary of" in pushed.summary
    # Local mode has no .clean.md → raw transcript text is pushed.
    assert "test transcript" in pushed.transcript

    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None and ep.wiki_pushed_at is not None


async def test_feature_off_when_url_unset_no_push_no_stats_keys(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    """With PODCAST_DIGEST_WEBHOOK_URL unset, _build_webhook returns None: no
    push attempted and the tick stats carry no webhook keys."""
    mailer = FakeMailer()
    stats = await run_tick(settings=settings, feeds_file=feeds_file,
                           db=db, mailer=mailer, dry_run=False)
    assert stats["emailed"] == 1
    assert "webhook_pushed" not in stats
    assert "webhook_failed" not in stats


async def test_feed_opt_out_is_not_pushed(
    settings, db, mock_llm, mock_transcribe, mock_http, use_ingestor,
):
    ing = use_ingestor(FakeIngestor())
    feeds_optout = FeedsFile(
        recipient="reader@example.com", poll_seconds=21600,
        feeds=[Feed(name="The Test Podcast", search="The Test Podcast",
                    min_minutes=60, webhook=False)],
    )
    mailer = FakeMailer()

    stats = await run_tick(settings=settings, feeds_file=feeds_optout,
                           db=db, mailer=mailer, dry_run=False)

    assert stats["emailed"] == 1
    assert ing.pushes == []          # opted out → never pushed
    assert "webhook_pushed" not in stats  # sweep didn't run for this feed
    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None and ep.wiki_pushed_at is None


async def test_dry_run_does_not_push(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, use_ingestor,
):
    ing = use_ingestor(FakeIngestor())
    mailer = FakeMailer()
    await run_tick(settings=settings, feeds_file=feeds_file,
                   db=db, mailer=mailer, dry_run=True)
    assert ing.pushes == []


async def test_push_is_idempotent_across_ticks(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, use_ingestor,
):
    ing = use_ingestor(FakeIngestor())
    mailer = FakeMailer()
    await run_tick(settings=settings, feeds_file=feeds_file,
                   db=db, mailer=mailer, dry_run=False)
    # Second tick: nothing new, already-pushed episode is not re-pushed.
    await run_tick(settings=settings, feeds_file=feeds_file,
                   db=db, mailer=mailer, dry_run=False)
    assert len(ing.pushes) == 1


async def test_push_failure_is_nonfatal_and_retried_next_tick(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, monkeypatch,
):
    """A failing push must NOT break the digest (email still sent), and the
    episode is retried on the next tick once the receiver recovers."""
    failing = FakeIngestor(fail=True)
    monkeypatch.setattr(
        "podcast_digest.pipeline._build_webhook", lambda settings: failing
    )
    mailer = FakeMailer()

    stats = await run_tick(settings=settings, feeds_file=feeds_file,
                           db=db, mailer=mailer, dry_run=False)
    # Digest unaffected.
    assert stats["emailed"] == 1
    assert len(mailer.sent) == 1
    assert stats["webhook_pushed"] == 0
    assert stats["webhook_failed"] == 1

    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None
    assert ep.status == "emailed"
    assert ep.wiki_pushed_at is None
    assert ep.wiki_push_attempts == 1

    # Next tick: the receiver recovers → the still-pending episode is retried and pushed.
    good = FakeIngestor()
    monkeypatch.setattr(
        "podcast_digest.pipeline._build_webhook", lambda settings: good
    )
    stats2 = await run_tick(settings=settings, feeds_file=feeds_file,
                            db=db, mailer=mailer, dry_run=False)
    assert stats2["webhook_pushed"] == 1
    assert len(good.pushes) == 1
    assert db.get_episode(FEED_URL, NEWEST_GUID).wiki_pushed_at is not None


async def test_missing_task_ack_does_not_persist_push_watermark(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, use_ingestor,
):
    """A malformed success response must remain retryable in SQLite."""
    ing = use_ingestor(MissingAckIngestor())

    stats = await run_tick(
        settings=settings,
        feeds_file=feeds_file,
        db=db,
        mailer=FakeMailer(),
        dry_run=False,
    )

    assert len(ing.pushes) == 1
    assert stats["emailed"] == 1
    assert stats["webhook_pushed"] == 0
    assert stats["webhook_failed"] == 1
    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None
    assert ep.wiki_pushed_at is None
    assert ep.wiki_push_attempts == 1


async def test_enabling_the_webhook_later_does_not_post_history(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, use_ingestor,
):
    """An episode emailed while the webhook was off is never posted when it is
    switched on later — only episodes emailed from then on are."""
    mailer = FakeMailer()
    # Webhook off: the episode is emailed, nothing is pushed.
    stats = await run_tick(settings=settings, feeds_file=feeds_file,
                           db=db, mailer=mailer, dry_run=False)
    assert stats["emailed"] == 1
    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None and ep.wiki_pushed_at is None
    # Make the earlier send visibly older than the switch-on moment.
    db._conn.execute(
        "UPDATE episodes SET emailed_at='2020-01-01T00:00:00+00:00' WHERE guid=?",
        (NEWEST_GUID,),
    )

    # Webhook on: the already-emailed episode is history, not posted.
    ing = use_ingestor(FakeIngestor())
    await run_tick(settings=settings, feeds_file=feeds_file,
                   db=db, mailer=mailer, dry_run=False)
    assert ing.pushes == []
    assert db.webhook_enabled_since(True) is not None
