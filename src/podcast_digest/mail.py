"""Mail delivery: SMTP or Microsoft Graph, behind one ``send()``.

  - :class:`SmtpMailer` — any mail server (your provider's SMTP, a relay, a
    transactional service with SMTP credentials). stdlib ``smtplib``.
  - :class:`GraphMailer` — Microsoft 365 via Graph ``sendMail`` with an app
    registration (OAuth2 client_credentials). stdlib urllib, ~one endpoint, so
    no msal / msgraph-sdk.

Both separate "nothing was accepted" (:class:`MailError`, safe to retry) from
"the message was fully transmitted but no verdict came back"
(:class:`MailProbablyDelivered`): sending is not idempotent, so the pipeline
treats the latter as delivered rather than risk duplicate digests.
"""
from __future__ import annotations

import base64
import html
import http.client
import json
import logging
import re
import smtplib
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from typing import TYPE_CHECKING, Literal, Protocol

from .log import event
from .net import upload_timeout

if TYPE_CHECKING:
    from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class MailAttachment:
    """A file to attach to the digest (e.g. the cleaned transcript)."""

    name: str
    content: bytes
    content_type: str = "application/octet-stream"

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
_GRAPH = urllib.parse.urlsplit(GRAPH_BASE)  # static netloc/path, split once
TOKEN_ENDPOINT = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
SCOPE = "https://graph.microsoft.com/.default"
HTTP_TIMEOUT = 30
# sendMail carries the base64-inlined transcript attachment (up to Graph's ~4 MB
# inline cap, ~1.33× inflated). A fixed 30 s trips mid-upload on a slow
# uplink; scale the timeout to the payload (base + ~1 s per 40 KB, capped —
# shared helper in net.py) so a large-but-healthy upload isn't cut short.
_UPLOAD_TIMEOUT_BASE = 30.0


class MailError(RuntimeError):
    """Graph returned a non-2xx response or auth failed."""


class MailProbablyDelivered(RuntimeError):
    """The request was fully transmitted to Graph but no readable verdict came.

    Graph sendMail is not idempotent, so this is the one genuinely ambiguous
    outcome. Raised as a distinct type (NOT a MailError) so the pipeline can
    choose the duplicate-avoiding branch: treat the episode as sent rather than
    re-send up to MAX_ATTEMPTS copies. Timeouts during connect or while the
    body is still uploading are NOT this — nothing was accepted, so they stay
    ordinary retryable MailErrors.

    Known, accepted imprecision: "fully transmitted" means the body left our
    socket buffer, not that Graph ACKed it — for a small body the two can
    differ if the network dies in the ~second between the TLS handshake and
    the 202. That window is narrow (the handshake itself just round-tripped),
    the failure is WARNING-logged with the episode title, and the worst case
    is one missing digest — chosen deliberately over the alternative's N
    duplicates. If a lost digest is ever actually observed, the escalation
    path is a post-timeout Sent Items check via Graph (saveToSentItems=True
    makes the send verifiable); not built now because the window hasn't been
    seen in practice and it would add a permission dependency.
    """


class Mailer(Protocol):
    """What the pipeline needs from a mail backend."""

    def send(
        self,
        *,
        to: str | list[str],
        subject: str,
        body_html: str,
        reply_to: str | None = None,
        attachments: list[MailAttachment] | None = None,
    ) -> None:
        """Deliver one message. Raises MailError or MailProbablyDelivered."""
        ...


class GraphMailer:
    """Send mail from a fixed sender mailbox using application credentials.

    The Entra ID app registration needs the `Mail.Send` application permission;
    scope it to the sender mailbox with an application access policy.
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        sender: str,
    ):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.sender = sender
        self._token: str | None = None
        self._token_expiry: float = 0
        self._token_lock = threading.Lock()

    def send(
        self,
        *,
        to: str | list[str],
        subject: str,
        body_html: str,
        reply_to: str | None = None,
        attachments: list[MailAttachment] | None = None,
    ) -> None:
        recipients = _recipients(to)
        if not recipients:
            raise MailError("no valid recipients")
        message: dict = {
            "subject": subject,
            "body": {"contentType": "HTML", "content": body_html},
            "toRecipients": recipients,
        }
        if reply_to:
            message["replyTo"] = _recipients(reply_to)
        if attachments:
            # Inline fileAttachment (base64). Graph caps inline attachments at
            # ~3MB total per message; a cleaned transcript is tens of KB.
            message["attachments"] = [
                {
                    "@odata.type": "#microsoft.graph.fileAttachment",
                    "name": a.name,
                    "contentType": a.content_type,
                    "contentBytes": base64.b64encode(a.content).decode("ascii"),
                }
                for a in attachments
            ]
        body = {"message": message, "saveToSentItems": True}

        path = f"/users/{urllib.parse.quote(self.sender)}/sendMail"
        self._post(path, body)
        event(
            logger, logging.INFO, "mail.sent",
            to=[r["emailAddress"]["address"] for r in recipients],
            subject=subject[:120],
            bytes=len(body_html),
            attachments=[a.name for a in (attachments or [])],
        )

    # ---- internals --------------------------------------------------------

    def _get_token(self) -> str:
        with self._token_lock:
            if self._token and time.time() < self._token_expiry:
                return self._token
            url = TOKEN_ENDPOINT.format(tenant=self.tenant_id)
            data = urllib.parse.urlencode({
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": SCOPE,
            }).encode()
            req = urllib.request.Request(url, data=data, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                    payload = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = _safe_read(e)
                raise MailError(f"token request failed: HTTP {e.code} {detail}") from e
            self._token = payload["access_token"]
            self._token_expiry = time.time() + payload.get("expires_in", 3600) - 60
            return self._token

    def _post(self, path: str, body: dict) -> None:
        """POST with the send/response phases separated.

        urlopen collapses connect + upload + response-wait into one opaque
        timeout, which makes a post-accept timeout indistinguishable from a
        dead network — the ambiguity behind the duplicate-digest bug. With
        http.client we know exactly how far the request got: a failure before
        the body is fully written means Graph accepted nothing (safe to retry,
        raise MailError); a timeout AFTER the body was transmitted means Graph
        almost certainly queued the send (raise MailProbablyDelivered so the
        caller doesn't re-send a duplicate).
        """
        token = self._get_token()
        data = json.dumps(body).encode()
        timeout = upload_timeout(len(data), base=_UPLOAD_TIMEOUT_BASE)
        conn = http.client.HTTPSConnection(_GRAPH.netloc, timeout=timeout)
        try:
            try:
                # Connect + transmit the full request (headers + body). Any
                # failure here — DNS, TCP, TLS, mid-upload timeout/reset —
                # happened before Graph could accept the message.
                conn.request(
                    "POST",
                    _GRAPH.path + path,
                    body=data,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
            except (TimeoutError, OSError) as e:
                raise MailError(f"sendMail transmit failed: {e}") from e
            try:
                resp = conn.getresponse()
            except http.client.RemoteDisconnected as e:
                # Connection closed with ZERO response bytes: Graph never even
                # started a reply, which reads as rejected-at-the-door (LB
                # reset, server bounce) — retry rather than risk a lost digest.
                raise MailError(
                    f"sendMail: server closed connection without a response: {e}"
                ) from e
            except (TimeoutError, http.client.HTTPException) as e:
                # No status line arrived (timeout), or what arrived was garbage
                # (BadStatusLine through a flaky middlebox) — both mean the
                # request went out but the verdict is unknowable. Bias
                # anti-duplicate, loudly.
                raise MailProbablyDelivered(
                    f"sendMail response unreadable after full transmit "
                    f"(timeout {timeout:.0f}s): {type(e).__name__}: {e}"
                ) from e
            except OSError as e:
                raise MailError(f"sendMail response failed: {e}") from e
            # Status line is in — from here the outcome is DEFINITIVE, so the
            # body read must never re-enter the probably-delivered path: a
            # stalled error-body on a 4xx/5xx is still a failed send (retry),
            # and on a 2xx the body is empty/ignorable anyway.
            try:
                detail = resp.read()  # /sendMail returns 202 No Content
            except (TimeoutError, OSError, http.client.HTTPException):
                detail = b""
            if not (200 <= resp.status < 300):
                raise MailError(
                    f"sendMail failed: HTTP {resp.status} "
                    f"{detail.decode('utf-8', errors='replace')[:500]}"
                )
        finally:
            conn.close()


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _recipients(value: str | list[str]) -> list[dict]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    else:
        parts = [str(p).strip() for p in value]
    cleaned: list[dict] = []
    for p in parts:
        if not p:
            continue
        if not _EMAIL_RE.match(p):
            # Don't ship to Graph and let it 400 the whole call after we burned
            # a summarization. Surface the typo here.
            raise MailError(f"invalid recipient address: {p!r}")
        cleaned.append({"emailAddress": {"address": p}})
    return cleaned


def _safe_read(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode("utf-8", errors="replace")[:500]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# SMTP
# ---------------------------------------------------------------------------


class _PayloadTracking:
    """Remembers whether the DATA payload (which ends in CRLF.CRLF) has left
    our socket: a failure after that point is ambiguous, before it is not."""

    payload_sent = False

    def send(self, s):  # type: ignore[no-untyped-def]
        super().send(s)  # type: ignore[misc]
        if isinstance(s, bytes | bytearray) and bytes(s).endswith(b"\r\n.\r\n"):
            self.payload_sent = True


class _SMTP(_PayloadTracking, smtplib.SMTP):
    pass


class _SMTP_SSL(_PayloadTracking, smtplib.SMTP_SSL):
    pass


class SmtpMailer:
    """Send mail through an SMTP server (STARTTLS, implicit TLS, or plain)."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        sender: str,
        username: str = "",
        password: str = "",
        security: Literal["starttls", "ssl", "none"] = "starttls",
    ):
        self.host = host
        self.port = port
        self.sender = sender
        self.username = username
        self.password = password
        self.security = security

    def send(
        self,
        *,
        to: str | list[str],
        subject: str,
        body_html: str,
        reply_to: str | None = None,
        attachments: list[MailAttachment] | None = None,
    ) -> None:
        recipients = [r["emailAddress"]["address"] for r in _recipients(to)]
        if not recipients:
            raise MailError("no valid recipients")
        msg = EmailMessage()
        msg["From"] = self.sender
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg["Date"] = formatdate(usegmt=True)
        msg["Message-ID"] = make_msgid(domain=self.sender.rpartition("@")[2] or None)
        if reply_to:
            msg["Reply-To"] = ", ".join(
                r["emailAddress"]["address"] for r in _recipients(reply_to)
            )
        # A plain-text part alongside the HTML: some clients and filters want it.
        msg.set_content(_html_to_text(body_html))
        msg.add_alternative(body_html, subtype="html")
        for a in attachments or []:
            maintype, _, subtype = a.content_type.split(";", 1)[0].strip().partition("/")
            msg.add_attachment(
                a.content,
                maintype=maintype or "application",
                subtype=subtype or "octet-stream",
                filename=a.name,
            )

        timeout = upload_timeout(len(bytes(msg)), base=_UPLOAD_TIMEOUT_BASE)
        smtp: _SMTP | _SMTP_SSL | None = None
        try:
            # Connect, negotiate TLS, authenticate: nothing is accepted yet.
            try:
                context = ssl.create_default_context()
                if self.security == "ssl":
                    smtp = _SMTP_SSL(self.host, self.port, timeout=timeout, context=context)
                else:
                    smtp = _SMTP(self.host, self.port, timeout=timeout)
                    if self.security == "starttls":
                        smtp.starttls(context=context)
                if self.username:
                    smtp.login(self.username, self.password)
            except (smtplib.SMTPException, OSError) as e:
                raise MailError(f"SMTP connect/login to {self.host}:{self.port} failed: {e}") from e
            try:
                refused = smtp.send_message(msg, from_addr=self.sender, to_addrs=recipients)
            except (smtplib.SMTPResponseException, smtplib.SMTPRecipientsRefused,
                    smtplib.SMTPSenderRefused) as e:
                # The server answered with a definite rejection.
                raise MailError(f"SMTP send rejected: {e}") from e
            except (smtplib.SMTPException, OSError) as e:
                if smtp.payload_sent:
                    # The whole message went out; only the server's verdict was
                    # lost. Bias anti-duplicate, loudly.
                    raise MailProbablyDelivered(
                        f"SMTP verdict lost after full transmit: {type(e).__name__}: {e}"
                    ) from e
                raise MailError(f"SMTP send failed: {e}") from e
        finally:
            if smtp is not None:
                try:
                    smtp.quit()
                except (smtplib.SMTPException, OSError):
                    smtp.close()
        if refused:
            logger.warning("mail.partially_refused refused=%s", sorted(refused))
        event(
            logger, logging.INFO, "mail.sent",
            to=recipients, subject=subject[:120], bytes=len(body_html),
            attachments=[a.name for a in (attachments or [])], via="smtp",
        )


_BLOCK_END_RE = re.compile(r"</(p|h[1-6]|li|ul|ol|div|tr)>|<br\s*/?>", re.IGNORECASE)


def _html_to_text(body_html: str) -> str:
    """A readable plain-text rendering of the digest HTML."""
    text = re.sub(r"<li[^>]*>", "- ", body_html, flags=re.IGNORECASE)
    text = _BLOCK_END_RE.sub("\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip() + "\n"


def build_mailer(settings: Settings) -> Mailer:
    """The configured mail backend. Raises ValueError when it is incomplete."""
    sender = settings.PODCAST_DIGEST_MAIL_FROM.strip()
    if not sender:
        raise ValueError("PODCAST_DIGEST_MAIL_FROM (the sender address) is required")
    if settings.mailer_backend == "graph":
        missing = [
            name for name, value in (
                ("PODCAST_DIGEST_GRAPH_TENANT_ID", settings.PODCAST_DIGEST_GRAPH_TENANT_ID),
                ("PODCAST_DIGEST_GRAPH_CLIENT_ID", settings.PODCAST_DIGEST_GRAPH_CLIENT_ID),
                ("PODCAST_DIGEST_GRAPH_CLIENT_SECRET",
                 settings.PODCAST_DIGEST_GRAPH_CLIENT_SECRET),
            ) if not value.strip()
        ]
        if missing:
            raise ValueError(f"PODCAST_DIGEST_MAILER=graph needs {', '.join(missing)}")
        return GraphMailer(
            tenant_id=settings.PODCAST_DIGEST_GRAPH_TENANT_ID,
            client_id=settings.PODCAST_DIGEST_GRAPH_CLIENT_ID,
            client_secret=settings.PODCAST_DIGEST_GRAPH_CLIENT_SECRET,
            sender=sender,
        )
    if not settings.PODCAST_DIGEST_SMTP_HOST.strip():
        raise ValueError(
            "PODCAST_DIGEST_SMTP_HOST is required (or set the Graph credentials "
            "and PODCAST_DIGEST_MAILER=graph)"
        )
    return SmtpMailer(
        host=settings.PODCAST_DIGEST_SMTP_HOST.strip(),
        port=settings.PODCAST_DIGEST_SMTP_PORT,
        sender=sender,
        username=settings.PODCAST_DIGEST_SMTP_USERNAME,
        password=settings.PODCAST_DIGEST_SMTP_PASSWORD,
        security=settings.PODCAST_DIGEST_SMTP_SECURITY,
    )
