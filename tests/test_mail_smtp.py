"""SMTP delivery against a minimal in-process SMTP server.

The server speaks just enough SMTP for smtplib (EHLO, MAIL, RCPT, DATA, QUIT)
and can misbehave after the message body arrives, so the tests pin the same
classification as the Graph mailer: a definite rejection or a failure before
the body left is a retryable MailError; a lost verdict after the full body went
out is MailProbablyDelivered (the pipeline treats it as sent).
"""
from __future__ import annotations

import email
import socket
import threading
from email import policy

import pytest

from podcast_digest.mail import (
    MailAttachment,
    MailError,
    MailProbablyDelivered,
    SmtpMailer,
    _html_to_text,
)


class _Server:
    """One-connection SMTP server. ``after_data``: "ok", "reject" or "drop"."""

    def __init__(self, after_data: str = "ok", rcpt: str = "ok"):
        self.after_data = after_data
        self.rcpt = rcpt
        self.messages: list[bytes] = []
        self.commands: list[str] = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        conn, _ = self.sock.accept()
        f = conn.makefile("rb")

        def say(line: str) -> None:
            conn.sendall(line.encode() + b"\r\n")

        say("220 test ESMTP")
        try:
            while True:
                raw = f.readline()
                if not raw:
                    return
                cmd = raw.decode().strip()
                self.commands.append(cmd.split(" ", 1)[0].upper())
                verb = cmd.upper()
                if verb.startswith(("EHLO", "HELO")):
                    say("250-test")
                    say("250 8BITMIME")
                elif verb.startswith("MAIL"):
                    say("250 ok")
                elif verb.startswith("RCPT"):
                    say("250 ok" if self.rcpt == "ok" else "550 no such user")
                elif verb == "DATA":
                    say("354 go ahead")
                    body = b""
                    while not body.endswith(b"\r\n.\r\n"):
                        chunk = f.readline()
                        if not chunk:
                            return
                        body += chunk
                    self.messages.append(body[:-5])
                    if self.after_data == "drop":
                        return  # close without a verdict
                    say("250 queued" if self.after_data == "ok" else "554 rejected")
                elif verb == "QUIT":
                    say("221 bye")
                    return
                else:
                    say("250 ok")
        finally:
            conn.close()
            self.sock.close()


def _mailer(port: int) -> SmtpMailer:
    return SmtpMailer(host="127.0.0.1", port=port, sender="bot@example.com", security="none")


def _send(mailer: SmtpMailer, **over) -> None:
    kw = dict(
        to="reader@example.com, second@example.com",
        subject="[Podcast] Show — Episode",
        body_html="<h2>TL;DR</h2><p>One &amp; two.</p><ul><li>First</li></ul>",
        attachments=[MailAttachment(
            name="Episode.md", content=b"# Episode\n\ncleaned",
            content_type="text/markdown; charset=utf-8",
        )],
    )
    kw.update(over)
    mailer.send(**kw)


def test_sends_html_with_text_part_and_attachment():
    server = _Server()
    _send(_mailer(server.port))
    server.thread.join(timeout=5)

    assert len(server.messages) == 1
    msg = email.message_from_bytes(server.messages[0], policy=policy.default)
    assert msg["From"] == "bot@example.com"
    assert msg["To"] == "reader@example.com, second@example.com"
    assert msg["Subject"] == "[Podcast] Show — Episode"
    assert msg["Message-ID"].endswith("@example.com>")
    html = msg.get_body(preferencelist=("html",)).get_content()
    text = msg.get_body(preferencelist=("plain",)).get_content()
    assert "<h2>TL;DR</h2>" in html
    assert "One & two." in text and "- First" in text
    attachments = list(msg.iter_attachments())
    assert [a.get_filename() for a in attachments] == ["Episode.md"]
    assert attachments[0].get_content_type() == "text/markdown"
    assert server.commands.count("RCPT") == 2


def test_lost_verdict_after_full_body_is_probably_delivered():
    server = _Server(after_data="drop")
    with pytest.raises(MailProbablyDelivered):
        _send(_mailer(server.port))
    assert len(server.messages) == 1  # the body did arrive


def test_rejection_after_body_is_a_retryable_error():
    server = _Server(after_data="reject")
    with pytest.raises(MailError, match="rejected"):
        _send(_mailer(server.port))


def test_refused_recipients_are_a_retryable_error():
    server = _Server(rcpt="refuse")
    with pytest.raises(MailError):
        _send(_mailer(server.port))


def test_connection_failure_is_a_retryable_error():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens here now
    with pytest.raises(MailError, match="connect"):
        _send(_mailer(port))


def test_invalid_recipient_fails_before_connecting():
    with pytest.raises(MailError, match="invalid recipient"):
        _send(_mailer(1), to="not-an-address")


def test_html_to_text_is_readable():
    text = _html_to_text("<h2>A</h2><p>x<br>y</p><ul><li>one</li><li>two</li></ul>")
    assert text == "A\nx\ny\n- one\n- two\n"
