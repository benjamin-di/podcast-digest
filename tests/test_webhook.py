"""Unit tests for the webhook client and the transcript-selection helper."""
from __future__ import annotations

import json
import urllib.error

import pytest

from podcast_digest import webhook
from podcast_digest.webhook import DeliveredEpisode, WebhookClient, WebhookError


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def read(self) -> bytes:
        return self._body


def _episode() -> DeliveredEpisode:
    return DeliveredEpisode(
        show="Market Hours",
        title="A strategist on bubbles",
        published_at="2026-06-18T00:00:00+00:00",
        guid="https://publisher.example/ep/abc",
        listen_url="https://cdn.example.com/abc.mp3",
        summary="<p>TL;DR</p>",
        transcript="[00:00:00] Speaker: hello world",
    )


def _client(**over) -> WebhookClient:
    kw = dict(
        url="https://kb.example.com/ingest", token="secret-token", timeout=5,
        fields={"vault": "research"}, ack_field="task_id",
    )
    kw.update(over)
    return WebhookClient(**kw)


def _capture(monkeypatch, body: bytes = b'{"task_id": "task-789"}') -> dict:
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data)
        captured["timeout"] = timeout
        return _FakeResp(body)

    monkeypatch.setattr(webhook.urllib.request, "urlopen", fake_urlopen)
    return captured


def test_push_sends_bearer_fields_and_payload_and_returns_ack(monkeypatch):
    captured = _capture(monkeypatch)

    ack = _client().push(_episode())

    assert ack == "task-789"
    assert captured["url"] == "https://kb.example.com/ingest"
    assert captured["method"] == "POST"
    # Timeout is floored at the configured value and scales up with the payload
    # (a small transcript here → ≈ the floor). See WebhookClient.push.
    assert 5 <= captured["timeout"] < 6
    # Bearer token present (header key capitalization is urllib's business).
    assert any(v == "Bearer secret-token" for v in captured["headers"].values())
    body = captured["body"]
    assert body["vault"] == "research"          # static extra field
    assert body["show"] == "Market Hours"
    assert body["title"] == "A strategist on bubbles"
    assert body["guid"] == "https://publisher.example/ep/abc"
    assert body["listen_url"] == "https://cdn.example.com/abc.mp3"
    assert body["summary"] == "<p>TL;DR</p>"
    assert body["transcript"].startswith("[00:00:00]")


def test_episode_fields_win_over_static_fields(monkeypatch):
    captured = _capture(monkeypatch)
    _client(fields={"title": "spoofed", "source": "podcasts"}).push(_episode())
    assert captured["body"]["title"] == "A strategist on bubbles"
    assert captured["body"]["source"] == "podcasts"


def test_no_token_sends_no_authorization_header(monkeypatch):
    captured = _capture(monkeypatch, b"OK")
    assert _client(token="", ack_field="").push(_episode()) == ""
    assert not any(k.lower() == "authorization" for k in captured["headers"])


def test_push_raises_on_http_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(webhook.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(WebhookError, match="HTTP 401"):
        _client().push(_episode())


def test_push_raises_on_network_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(webhook.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(WebhookError, match="request failed"):
        _client().push(_episode())


@pytest.mark.parametrize("body", [b"OK", b"{}", b'{"task_id": "  "}'])
def test_push_rejects_2xx_without_required_ack(monkeypatch, body):
    """With an ack field configured, only its value proves the receiver queued
    the episode — a proxy's success page must not count as delivery."""
    _capture(monkeypatch, body)
    with pytest.raises(WebhookError, match="missing 'task_id'"):
        _client().push(_episode())


@pytest.mark.parametrize("body", [b"OK", b"{}", b""])
def test_push_without_ack_field_accepts_any_2xx(monkeypatch, body):
    _capture(monkeypatch, body)
    assert _client(ack_field="").push(_episode()) == ""


def test_read_transcript_prefers_clean_md(tmp_path):
    """The push helper prefers the cleaned .md over the raw .txt."""
    from podcast_digest.db import EpisodeRow
    from podcast_digest.pipeline import _read_transcript_for_webhook

    txt = tmp_path / "ep.txt"
    txt.write_text("raw asr text", encoding="utf-8")
    clean = tmp_path / "ep.clean.md"
    clean.write_text("# cleaned\n[00:00] A: hi", encoding="utf-8")

    ep = EpisodeRow(
        feed_url="f", guid="g", title="T", published_at="2026", duration_seconds=1,
        audio_url="u", audio_path=None, transcript_path=str(txt), segments_path=None,
        summary_html="<p>s</p>", content_sha256=None, content_minhash=None,
        content_token_count=None, status="emailed", terminal_reason="sent",
        attempts=0, clean_attempts=0,
        summarize_attempts=0, wiki_push_attempts=0, last_error=None, failed_at=None,
        emailed_at="2026", wiki_pushed_at=None,
    )
    assert _read_transcript_for_webhook(ep) == "# cleaned\n[00:00] A: hi"

    # With no clean.md, falls back to the raw .txt.
    clean.unlink()
    assert _read_transcript_for_webhook(ep) == "raw asr text"

    # With nothing on disk, None.
    txt.unlink()
    assert _read_transcript_for_webhook(ep) is None
