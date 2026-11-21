"""Graph transport certainty: only proven non-submission permits retry."""
from __future__ import annotations

import http.client

import pytest

from podcast_digest.mail import GraphMailer, MailError, MailOutcomeUnknown


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

    connect_exc = None
    request_exc = None
    getresponse_exc = None
    resp = None

    def __init__(self, netloc, timeout=None):
        pass

    def connect(self):
        if type(self).connect_exc:
            raise type(self).connect_exc

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
    _FakeConn.connect_exc = None
    _FakeConn.request_exc = None
    _FakeConn.getresponse_exc = None
    _FakeConn.resp = _FakeResp()
    return m


def test_post_202_succeeds(mailer):
    mailer._post("/users/bot@x/sendMail", {"k": "v"})  # no raise


def test_connect_failure_proves_non_submission(mailer):
    _FakeConn.connect_exc = TimeoutError("connect stalled")
    with pytest.raises(MailError):
        mailer._post("/p", {})


@pytest.mark.parametrize("exc", [TimeoutError(), ConnectionResetError(), BrokenPipeError(),
                                 http.client.RemoteDisconnected()])
def test_request_socket_error_is_unknown_even_if_sendall_raises(mailer, exc):
    _FakeConn.request_exc = exc
    with pytest.raises(MailOutcomeUnknown):
        mailer._post("/p", {})


def test_response_timeout_is_unknown(mailer):
    _FakeConn.getresponse_exc = TimeoutError("no status line")
    with pytest.raises(MailOutcomeUnknown):
        mailer._post("/p", {})


def test_garbled_response_is_unknown(mailer):
    # BadStatusLine is an HTTPException, NOT an OSError — it must land in the
    # same ambiguous bucket as a response timeout, not escape uncategorized
    # (which would re-send and bring back the duplicate storm).
    _FakeConn.getresponse_exc = http.client.BadStatusLine("garbage")
    with pytest.raises(MailOutcomeUnknown):
        mailer._post("/p", {})


@pytest.mark.parametrize("exc", [http.client.RemoteDisconnected("closed"),
                                 ConnectionResetError("reset"), OSError("socket closed")])
def test_response_socket_closure_is_unknown(mailer, exc):
    _FakeConn.getresponse_exc = exc
    with pytest.raises(MailOutcomeUnknown):
        mailer._post("/p", {})


@pytest.mark.parametrize("status", [408, 500, 502, 503, 504, 200, 302])
def test_unproven_status_is_unknown_even_when_body_stalls(mailer, status):
    _FakeConn.resp = _FakeResp(status=status, read_exc=TimeoutError("body stalled"))
    with pytest.raises(MailOutcomeUnknown, match=f"HTTP {status}"):
        mailer._post("/p", {})


@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 413, 415, 422, 429])
def test_explicit_rejection_proves_non_submission(mailer, status):
    _FakeConn.resp = _FakeResp(status=status, read_exc=TimeoutError("body stalled"))
    with pytest.raises(MailError, match=f"HTTP {status}"):
        mailer._post("/p", {})


def test_202_with_unreadable_body_still_proves_acceptance(mailer):
    _FakeConn.resp = _FakeResp(read_exc=TimeoutError("body stalled"))
    mailer._post("/p", {})


def test_error_status_includes_body_detail(mailer):
    _FakeConn.resp = _FakeResp(status=429, body=b'{"error":"throttled"}')
    with pytest.raises(MailError, match="throttled"):
        mailer._post("/p", {})


def test_operation_header_is_correlation_and_sent_copy_is_requested(mailer, monkeypatch):
    captured = []
    monkeypatch.setattr(mailer, "_post", lambda path, body: captured.append(body))
    mailer.send(to="reader@example.com", subject="Synthetic digest", body_html="<p>x</p>",
                operation_id="a" * 32)
    assert captured[0]["saveToSentItems"] is True
    assert captured[0]["message"]["internetMessageHeaders"] == [
        {"name": "x-podcast-digest-operation", "value": "a" * 32},
    ]
