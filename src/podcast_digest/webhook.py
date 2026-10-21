"""Delivery webhook client.

After an episode is emailed, POST it (metadata + summary + transcript) as JSON
to a configured URL — a knowledge-base ingest route, an automation tool (n8n,
Zapier, Make), a notes service. The receiver decides what to do with it; this
client only ships the episode and reports whether the receiver accepted it.
See docs/webhook.md for the payload and the retry contract.

stdlib `urllib` only, like `mail.py` — no new dependency.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .log import event
from .net import upload_timeout

logger = logging.getLogger(__name__)


class WebhookError(RuntimeError):
    """The receiver returned a non-2xx response, did not acknowledge, or the
    request failed."""


@dataclass
class DeliveredEpisode:
    """The webhook payload for one emailed episode."""

    show: str
    title: str
    published_at: str
    guid: str
    listen_url: str
    summary: str
    transcript: str


@dataclass
class WebhookClient:
    """POST an episode to the webhook, with an optional static bearer token.

    Fire-and-return: a receiver that does slow work (an ingest agent, say)
    should queue it and answer at once. ``ack_field`` names the response key
    that proves the receiver queued the episode (e.g. "task_id"); when set, a
    2xx without it is a failure, so a proxy's generic success page never counts
    as delivery. When unset, any 2xx counts.
    """

    url: str
    token: str = ""
    timeout: int = 30
    fields: dict[str, Any] = field(default_factory=dict)
    ack_field: str = ""

    def push(self, ep: DeliveredEpisode) -> str:
        """Push one episode. Returns the acknowledgement value ("" when no
        ack_field is configured). Raises WebhookError."""
        body = json.dumps({
            **self.fields,
            "show": ep.show,
            "title": ep.title,
            "published_at": ep.published_at,
            "guid": ep.guid,
            "listen_url": ep.listen_url,
            "summary": ep.summary,
            "transcript": ep.transcript,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        # Scale the timeout to the payload: the body carries the full transcript
        # (can be hundreds of KB), and a timeout after the receiver already
        # accepted the POST (2xx lost in transit) re-POSTs next tick — a
        # duplicate delivery. Shared, hard-capped helper (net.py): an uncapped
        # scale would let a wedged socket stall the tick for minutes on a
        # multi-MB body. `self.timeout` is the floor.
        timeout = upload_timeout(len(body), base=float(self.timeout))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = _parse(resp.read())
        except urllib.error.HTTPError as e:
            raise WebhookError(f"webhook failed: HTTP {e.code} {_safe_read(e)}") from e
        except urllib.error.URLError as e:
            raise WebhookError(f"webhook request failed: {e.reason}") from e
        ack = ""
        if self.ack_field:
            ack = str(payload.get(self.ack_field) or "").strip()
            if not ack:
                # A truncated response or a proxy-generated success page gives
                # no evidence that the receiver accepted the episode; treating
                # it as success would set the delivery watermark and lose the
                # episode forever. Leave it unset so the pipeline retries.
                raise WebhookError(f"webhook response missing {self.ack_field!r}")
        event(
            logger, logging.INFO, "webhook.pushed",
            show=ep.show, title=ep.title[:120],
            guid=ep.guid, ack=ack or None,
            bytes=len(ep.transcript),
        )
        return ack


def _parse(raw: bytes) -> dict:
    try:
        data = json.loads(raw or b"{}")
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _safe_read(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode("utf-8", errors="replace")[:500]
    except Exception:
        return ""
