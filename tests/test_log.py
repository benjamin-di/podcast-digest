"""Tests for the structured logging layer (log.py)."""
from __future__ import annotations

import json
import logging

from podcast_digest.log import (
    _JsonFormatter,
    err_fields,
    event,
    oneline,
    strip_ansi,
)


def _emit(logger: logging.Logger, level: int, event_name: str, **fields):
    """Capture one record produced by event() and return its parsed JSON."""
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        event(logger, level, event_name, **fields)
    finally:
        logger.removeHandler(handler)
    return json.loads(_JsonFormatter().format(records[-1]))


def test_event_promotes_fields_to_top_level():
    out = _emit(logging.getLogger("t.promote"), logging.INFO, "feed.polled",
                parsed=626, first_run=False)
    assert out["msg"] == "feed.polled"
    assert out["lvl"] == "INFO"
    assert out["parsed"] == 626          # real int, queryable
    assert out["first_run"] is False     # real bool


def test_timestamp_is_utc_millis_z():
    out = _emit(logging.getLogger("t.ts"), logging.INFO, "x")
    ts = out["ts"]
    assert ts.endswith("Z")
    # 2026-06-29T14:10:36.123Z -> the millis are 3 digits before the Z.
    assert ts[-5] == "." and ts[19] == "." and len(ts) == 24


def test_reserved_key_collision_is_renamed():
    # A field literally named "name" must not clobber the logger name.
    out = _emit(logging.getLogger("t.collide"), logging.INFO, "x", name="GS")
    assert out["name"] == "t.collide"   # logger name preserved
    assert out["name_"] == "GS"          # extra renamed, not lost


def test_strip_ansi_and_oneline():
    assert strip_ansi("\x1b[31mred\x1b[0m") == "red"
    assert oneline("a\n  b\t c", limit=100) == "a b c"
    assert oneline("x" * 50, limit=10) == "x" * 10


def test_err_fields_plain_exception():
    f = err_fields(ValueError("\x1b[31mboom\x1b[0m happened"))
    assert f["err_class"] == "ValueError"
    assert f["err_brief"] == "boom happened"   # ANSI stripped
    assert "err_reason" not in f               # plain exc has no reason
    assert "auth_failure" not in f


def test_err_fields_carries_reason_and_auth():
    class FakeVps(RuntimeError):
        reason = "auth_failure"
        auth_failure = True

    f = err_fields(FakeVps("nope"))
    assert f["err_reason"] == "auth_failure"
    assert f["auth_failure"] is True


def test_err_fields_ignores_non_string_reason():
    # A foreign exception (e.g. URLError) can carry a non-string .reason —
    # it must not pollute the alerting field.
    class ForeignErr(OSError):
        reason = OSError("connection refused")

    f = err_fields(ForeignErr("boom"))
    assert "err_reason" not in f
    assert f["err_class"] == "ForeignErr"
