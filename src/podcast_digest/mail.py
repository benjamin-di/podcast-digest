"""Mail delivery: SMTP or Microsoft Graph, behind one ``send()``.

  - :class:`SmtpMailer` — any mail server (your provider's SMTP, a relay, a
    transactional service with SMTP credentials). stdlib ``smtplib``.
  - :class:`GraphMailer` — Microsoft 365 via Graph ``sendMail`` with an app
    registration (OAuth2 client_credentials). stdlib urllib, ~one endpoint, so
    no msal / msgraph-sdk.

MailError proves non-submission. MailOutcomeUnknown means that submission may
have happened and must be held for positive observation, never treated as sent.
A correlation key identifies an operation; neither backend deduplicates it.
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
    """The provider accepted nothing; retrying the saved envelope is safe."""


class MailOutcomeUnknown(RuntimeError):
    """Submission may have happened; neither success nor retry is justified."""


class MailObservationError(RuntimeError):
    """Read-back failed; it says nothing about an outstanding send's outcome."""


class Mailer(Protocol):
    """What the pipeline needs from a mail backend."""

    @property
    def delivery_target(self) -> str:
        """Stable transport and sender identity, excluding renewable credentials."""
        ...

    def send(
        self,
        *,
        to: str | list[str],
        subject: str,
        body_html: str,
        reply_to: str | None = None,
        attachments: list[MailAttachment] | None = None,
        operation_id: str | None = None,
        date_header: str | None = None,
    ) -> None:
        """Deliver one message. Raises MailError or MailOutcomeUnknown."""
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
        operation_id: str | None = None,
        date_header: str | None = None,
    ) -> None:
        recipients = _recipients(to)
        if not recipients:
            raise MailError("no valid recipients")
        message: dict = {
            "subject": subject,
            "body": {"contentType": "HTML", "content": body_html},
            "toRecipients": recipients,
        }
        if operation_id:
            message["internetMessageHeaders"] = [
                {"name": "x-podcast-digest-operation", "value": operation_id},
            ]
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

        path = f"/users/{urllib.parse.quote(self.sender, safe="")}/sendMail"
        self._post(path, body)
        event(
            logger, logging.INFO, "mail.sent",
            to=[r["emailAddress"]["address"] for r in recipients],
            subject=subject[:120],
            bytes=len(body_html),
            attachments=[a.name for a in (attachments or [])],
        )

    @property
    def delivery_target(self) -> str:
        return f"graph:{self.tenant_id}:{self.sender.casefold()}"

    def sent_receipt(
        self, *, operation_id: str, message_id: str, to: str, subject: str,
    ) -> str | None:
        """Observe a particular Sent Items message; absence never authorizes a retry.

        Requires Mail.Read for this mailbox, in addition to Mail.Send. Custom
        internet headers are supported on sendMail and readable with $select.
        """
        sender = urllib.parse.quote(self.sender, safe="")
        item = urllib.parse.quote(message_id, safe="")
        select = "id,isDraft,sentDateTime,internetMessageHeaders,subject,toRecipients,from"
        url = f"{GRAPH_BASE}/users/{sender}/mailFolders/sentitems/messages/{item}?"
        try:
            req = urllib.request.Request(
                url + urllib.parse.urlencode({"$select": select}),
                headers={"Authorization": f"Bearer {self._get_token()}"},
            )
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                message = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise MailObservationError(f"Sent Items observation failed: HTTP {exc.code}") from exc
        except (MailError, OSError, http.client.HTTPException, ValueError, KeyError) as exc:
            raise MailObservationError("Sent Items observation unavailable") from exc
        headers = message.get("internetMessageHeaders", [])
        keys = [h.get("value") for h in headers
                if h.get("name", "").casefold() == "x-podcast-digest-operation"]
        expected = {r["emailAddress"]["address"].casefold() for r in _recipients(to)}
        actual = {r["emailAddress"]["address"].casefold()
                  for r in message.get("toRecipients", [])}
        source = message.get("from", {}).get("emailAddress", {}).get("address", "")
        if (
            message.get("id") and message.get("isDraft") is False
            and message.get("sentDateTime") and keys == [operation_id]
            and message.get("subject") == subject and actual == expected
            and source.casefold() == self.sender.casefold()
        ):
            return f"graph-sent-item:{message['id']}"
        return None

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
        """Only failures before request I/O or explicit rejections permit retry."""
        try:
            token = self._get_token()
        except MailError:
            raise
        except (OSError, http.client.HTTPException, ValueError, KeyError) as exc:
            raise MailError("authentication failed before sendMail submission") from exc
        data = json.dumps(body).encode()
        timeout = upload_timeout(len(data), base=_UPLOAD_TIMEOUT_BASE)
        conn = http.client.HTTPSConnection(_GRAPH.netloc, timeout=timeout)
        try:
            try:
                conn.connect()
            except (OSError, http.client.HTTPException) as exc:
                raise MailError("sendMail connection failed before submission") from exc
            try:
                conn.request(
                    "POST", _GRAPH.path + path, body=data,
                    headers={"Authorization": f"Bearer {token}",
                             "Content-Type": "application/json"},
                )
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                # sendall can submit every byte and still raise. A reset, EOF,
                # or timeout at either boundary cannot prove non-submission.
                raise MailOutcomeUnknown("sendMail submission verdict unavailable") from exc
            try:
                detail = resp.read()
            except (OSError, http.client.HTTPException):
                detail = b""
            if resp.status == 202:
                return  # Graph's documented acceptance, even if body read failed.
            if resp.status in {400, 401, 403, 404, 405, 413, 415, 422, 429}:
                raise MailError(
                    f"sendMail rejected: HTTP {resp.status} "
                    f"{detail.decode('utf-8', errors='replace')[:500]}"
                )
            # A server/proxy 5xx or unexpected status is no proof that its
            # upstream did not queue this non-idempotent operation.
            raise MailOutcomeUnknown(f"sendMail acceptance unknown: HTTP {resp.status}")
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
    """Track possible DATA submission before entering the socket write."""

    payload_started = False

    def send(self, s):  # type: ignore[no-untyped-def]
        if isinstance(s, bytes | bytearray) and bytes(s).endswith(b"\r\n.\r\n"):
            # Mark BEFORE sendall: an exception does not say how much arrived.
            self.payload_started = True
        super().send(s)  # type: ignore[misc]


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

    @property
    def delivery_target(self) -> str:
        return f"smtp:{self.host.casefold()}:{self.port}:{self.sender.casefold()}"

    def send(
        self,
        *,
        to: str | list[str],
        subject: str,
        body_html: str,
        reply_to: str | None = None,
        attachments: list[MailAttachment] | None = None,
        operation_id: str | None = None,
        date_header: str | None = None,
    ) -> None:
        recipients = [r["emailAddress"]["address"] for r in _recipients(to)]
        if not recipients:
            raise MailError("no valid recipients")
        msg = EmailMessage()
        msg["From"] = self.sender
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg["Date"] = date_header or formatdate(usegmt=True)
        domain = self.sender.rpartition("@")[2] or "localhost"
        msg["Message-ID"] = (
            f"<{operation_id}@{domain}>" if operation_id else make_msgid(domain=domain)
        )
        if operation_id:
            msg["X-Podcast-Digest-Operation"] = operation_id
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
                if smtp.payload_started:
                    # A socket error cannot prove how much DATA arrived.
                    # Hold until positive evidence is available.
                    raise MailOutcomeUnknown(
                        f"SMTP verdict unknown after possible submission: {type(e).__name__}: {e}"
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


_STYLE_BLOCK_RE = re.compile(r"<style[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
_BLOCK_END_RE = re.compile(r"</(p|h[1-6]|li|ul|ol|div|tr)>|<br\s*/?>", re.IGNORECASE)


def _html_to_text(body_html: str) -> str:
    """A readable plain-text rendering of the digest HTML."""
    text = _STYLE_BLOCK_RE.sub("", body_html)
    text = re.sub(r"<li[^>]*>", "- ", text, flags=re.IGNORECASE)
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
