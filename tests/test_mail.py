"""Graph sendMail POST classification: the transmit/response phase split.

Graph sendMail is not idempotent, so _post's job is to classify every failure
into exactly one of: retryable (nothing accepted — MailError) or ambiguous
(request fully out, verdict unreadable — MailProbablyDelivered, which the
pipeline treats as sent to avoid duplicate digests). These tests fake
http.client.HTTPSConnection to pin each classification.
"""
from __future__ import annotations

import http.client

import pytest

from podcast_digest.mail import GraphMailer, MailError, MailProbablyDelivered


class _FakeResp:
    def __init__(self, status=202, body=b"", read_exc=None):
        self.status = status
        self._body = body
        self._read_exc = read_exc

    def read(self):
        if self._read_exc:
            raise self._read_exc
        return self._body


class _FakeConn:
    """Stands in for HTTPSConnection; behavior injected per test."""

    request_exc = None
    getresponse_exc = None
    resp = None

    def __init__(self, netloc, timeout=None):
        pass

    def request(self, method, path, body=None, headers=None):
        if type(self).request_exc:
            raise type(self).request_exc

    def getresponse(self):
        if type(self).getresponse_exc:
            raise type(self).getresponse_exc
        return type(self).resp

    def close(self):
        pass


@pytest.fixture
def mailer(monkeypatch):
    m = GraphMailer(tenant_id="t", client_id="c", client_secret="s", sender="bot@x")
    # Skip the token round-trip; _post only needs a bearer string.
    m._token = "tok"
    m._token_expiry = 2**62
    monkeypatch.setattr(http.client, "HTTPSConnection", _FakeConn)
    # Reset class-level injection between tests.
    _FakeConn.request_exc = None
    _FakeConn.getresponse_exc = None
    _FakeConn.resp = _FakeResp()
    return m


def test_post_202_succeeds(mailer):
    mailer._post("/users/bot@x/sendMail", {"k": "v"})  # no raise


def test_transmit_timeout_is_retryable(mailer):
    # Failure before/while the body goes out: Graph accepted nothing → retry.
    _FakeConn.request_exc = TimeoutError("send buffer stalled")
    with pytest.raises(MailError):
        mailer._post("/p", {})


def test_response_timeout_is_probably_delivered(mailer):
    _FakeConn.getresponse_exc = TimeoutError("no status line")
    with pytest.raises(MailProbablyDelivered):
        mailer._post("/p", {})


def test_garbled_response_is_probably_delivered(mailer):
    # BadStatusLine is an HTTPException, NOT an OSError — it must land in the
    # same ambiguous bucket as a response timeout, not escape uncategorized
    # (which would re-send and bring back the duplicate storm).
    _FakeConn.getresponse_exc = http.client.BadStatusLine("garbage")
    with pytest.raises(MailProbablyDelivered):
        mailer._post("/p", {})


def test_remote_disconnected_is_retryable(mailer):
    # Zero response bytes = Graph never started a reply → treat as rejected.
    _FakeConn.getresponse_exc = http.client.RemoteDisconnected("closed")
    with pytest.raises(MailError):
        mailer._post("/p", {})


def test_error_status_with_stalled_body_is_still_a_failure(mailer):
    # Status line received and it's a 500: the outcome is DEFINITIVE even if
    # reading the error body times out — this must be a retryable MailError,
    # never probably-delivered (marking a provably-failed send as emailed
    # would silently lose the digest).
    _FakeConn.resp = _FakeResp(status=500, read_exc=TimeoutError("body stalled"))
    with pytest.raises(MailError, match="HTTP 500"):
        mailer._post("/p", {})


def test_error_status_includes_body_detail(mailer):
    _FakeConn.resp = _FakeResp(status=429, body=b'{"error":"throttled"}')
    with pytest.raises(MailError, match="throttled"):
        mailer._post("/p", {})
