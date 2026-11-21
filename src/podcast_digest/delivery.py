"""One immutable digest operation, held until its submission outcome is known."""
from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from email.utils import formatdate

from .db import Database, DeliveryRow
from .mail import GraphMailer, MailAttachment, Mailer, MailError


class DeliveryHeld(RuntimeError):
    def __init__(self, operation_id: str):
        super().__init__(
            f"delivery {operation_id} held; run podcast-digest delivery inspect {operation_id}; "
            "reconcile positive Sent Items evidence with delivery reconcile"
        )


@dataclass(frozen=True)
class Envelope:
    to: str
    subject: str
    body_html: str
    attachments: tuple[MailAttachment, ...] = ()
    date_header: str = ""

    def serialize(self) -> str:
        return json.dumps({
            "to": self.to, "subject": self.subject, "body_html": self.body_html,
            "date_header": self.date_header or formatdate(usegmt=True),
            "attachments": [
                {"name": a.name, "content_type": a.content_type,
                 "content": base64.b64encode(a.content).decode("ascii")}
                for a in self.attachments
            ],
        }, sort_keys=True)

    @classmethod
    def parse(cls, value: str) -> Envelope:
        data = json.loads(value)
        return cls(
            to=data["to"], subject=data["subject"], body_html=data["body_html"],
            date_header=data["date_header"],
            attachments=tuple(MailAttachment(
                name=a["name"], content_type=a["content_type"],
                content=(
                    base64.b64decode(a["content"], validate=True)
                    if "content" in a else b""
                ),
            ) for a in data["attachments"]),
        )


async def send_digest(
    db: Database, mailer: Mailer, feed_url: str, guid: str,
    envelope: Envelope | None = None, *, show: str = "",
) -> bool:
    """Called only while the existing tick lock excludes every other sender."""
    previous = db.get_delivery(feed_url, guid)
    if previous is not None and (
        previous.state != "not_submitted" or previous.suppression_reason is not None
    ):
        raise DeliveryHeld(previous.operation_id)
    if previous is None and envelope is None:
        raise ValueError("first submission requires an envelope")
    if previous is not None:
        payload = previous.envelope
    else:
        assert envelope is not None
        payload = envelope.serialize()
    claimed = db.claim_delivery(
        feed_url, guid, target=mailer.delivery_target, envelope=payload, show=show,
    )
    if claimed is None:
        current = db.get_delivery(feed_url, guid)
        assert current is not None
        raise DeliveryHeld(current.operation_id)
    frozen = Envelope.parse(claimed.envelope)
    try:
        await asyncio.to_thread(
            mailer.send,
            to=frozen.to, subject=frozen.subject, body_html=frozen.body_html,
            attachments=list(frozen.attachments), operation_id=claimed.operation_id,
            date_header=frozen.date_header,
        )
    except MailError as exc:
        db.delivery_not_submitted(claimed.operation_id, type(exc).__name__)
        raise
    except Exception as exc:
        # The committed unknown record also survives BaseException/SIGKILL and
        # a storage error while recording positive acceptance below.
        raise DeliveryHeld(claimed.operation_id) from exc
    db.complete_delivery(claimed.operation_id, "transport:accepted")
    return True


def reconcile_sent_item(
    db: Database, mailer: GraphMailer, delivery: DeliveryRow, message_id: str,
) -> bool:
    """Read-only provider observation; never submits or releases an unknown send."""
    if delivery.target != mailer.delivery_target:
        raise ValueError("configured Graph mailbox does not match saved delivery target")
    if delivery.state == "accepted":
        return True
    if delivery.state != "unknown":
        raise ValueError("delivery has no unknown submission to reconcile")
    frozen = Envelope.parse(delivery.envelope)
    receipt = mailer.sent_receipt(
        operation_id=delivery.operation_id, message_id=message_id,
        to=frozen.to, subject=frozen.subject,
    )
    if receipt is None:
        return False
    db.complete_delivery(delivery.operation_id, receipt)
    return True
