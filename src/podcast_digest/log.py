"""Structured logging — single-line JSON when stderr is not a TTY, plain otherwise.

One JSON object per line: timestamp + level + name + msg,
with context promoted to real, queryable fields (not interpolated into the message).

Call sites use :func:`event` so context lands as top-level JSON keys:

    event(logger, logging.INFO, "feed.polled", feed=name, parsed=n, first_run=False)
    # -> {"ts": "...", "lvl": "INFO", "name": "...", "msg": "feed.polled",
    #     "feed": "...", "parsed": 626, "first_run": false}

That makes logs `jq`-filterable (`select(.feed=="…")`, `select(.auth_failure==true)`)
instead of grep-over-a-string. JSON output is one object per line; timestamps are
UTC with millisecond resolution and a trailing ``Z`` so logs from several hosts can be
ordered without ambiguity.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import TypedDict

# Keys owned by the formatter; extras may never clobber them. A colliding extra
# (e.g. a field literally named "name") is renamed with a trailing underscore.
_RESERVED = ("ts", "lvl", "name", "msg", "exc")

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def strip_ansi(s: str) -> str:
    """Remove ANSI SGR/CSI escapes — agent CLIs colour their error output."""
    return _ANSI_RE.sub("", s)


def oneline(s: str, *, limit: int = 200) -> str:
    """Collapse whitespace to single spaces and truncate — for `err_brief`."""
    return " ".join(s.split())[:limit]


class ErrorFields(TypedDict, total=False):
    err_class: str
    err_brief: str
    err_reason: str
    auth_failure: bool


def err_fields(exc: BaseException, *, limit: int = 200) -> ErrorFields:
    """Structured fields for an exception: class, a one-line brief, and the
    machine reason / auth flag when the exception carries them (SummarizerUnavailable).

    ``err_reason`` is the alerting field (auth_failure / exit_nonzero / timeout /
    …) and is only emitted for a *string* ``.reason`` — so a foreign exception
    that happens to carry a non-string ``.reason`` (e.g. ``URLError.reason`` can
    be an ``OSError``) doesn't pollute it. Free-text detail lives in ``err_brief``.
    """
    fields: ErrorFields = {
        "err_class": type(exc).__name__,
        "err_brief": oneline(strip_ansi(str(exc)), limit=limit),
    }
    reason = getattr(exc, "reason", None)
    if isinstance(reason, str):
        fields["err_reason"] = reason
    if getattr(exc, "auth_failure", False):
        fields["auth_failure"] = True
    return fields


class _UtcFormatter(logging.Formatter):
    """Shared base: UTC ISO-8601 timestamps with millisecond resolution + Z."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
        return f"{base}.{int(record.msecs):03d}Z"


class _KVFormatter(_UtcFormatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = getattr(record, "extras", None)
        if extras:
            kv = " ".join(f"{k}={_fmt(v)}" for k, v in extras.items())
            return f"{base} {kv}"
        return base


class _JsonFormatter(_UtcFormatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record),
            "lvl": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        extras = getattr(record, "extras", None)
        if isinstance(extras, dict):
            for k, v in extras.items():
                payload[f"{k}_" if k in _RESERVED else k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def _fmt(v: object) -> str:
    if isinstance(v, str) and (" " in v or "=" in v):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def configure(level: str | None = None) -> None:
    """Idempotent root logger setup."""
    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    root.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    if sys.stderr.isatty():
        h.setFormatter(_KVFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    else:
        h.setFormatter(_JsonFormatter())
    root.addHandler(h)
    root.setLevel(level)
    # quiet noisy deps
    for name in ("httpx", "httpcore", "anthropic._base_client"):
        logging.getLogger(name).setLevel(logging.WARNING)


def event(
    logger: logging.Logger,
    level: int,
    name: str,
    /,
    *,
    exc_info: bool = False,
    **fields: object,
) -> None:
    """Log a ``noun.verb`` event with context promoted to queryable fields.

    ``name`` is the event identifier (the JSON ``msg``); ``fields`` become
    top-level JSON keys. Pass ``exc_info=True`` to attach the active traceback.
    """
    logger.log(level, name, extra={"extras": fields}, exc_info=exc_info)


def log_with(logger: logging.Logger, level: int, msg: str, /, **extras: object) -> None:
    """Back-compat alias for :func:`event`."""
    logger.log(level, msg, extra={"extras": extras})
