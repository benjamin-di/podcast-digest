"""Regression tests for standalone local_transcriber Rich logging attribution."""
from __future__ import annotations

import io
import json
import logging

import pytest

rich_console = pytest.importorskip("rich.console")
rich_logging = pytest.importorskip("rich.logging")

from local_transcriber import log as lt_log  # noqa: E402


def _make_handler(console: object) -> logging.Handler:
    return rich_logging.RichHandler(
        console=console,
        show_time=False,
        show_level=False,
        show_path=False,
        highlighter=None,
    )


def test_local_transcriber_rich_capture_is_preserved():
    sink = io.StringIO()
    console = rich_console.Console(file=sink, force_terminal=False, color_system=None)
    logger = logging.getLogger("local_transcriber.test.capture")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(_make_handler(console))

    lt_log.stamp_handlers()
    assert isinstance(logger.handlers[0], lt_log._RichSink)

    with console.capture() as captured:
        logger.info("captured diagnostic")

    output = captured.get().strip()
    assert sink.getvalue() == ""
    payload = json.loads(output)
    assert payload["msg"].strip() == "captured diagnostic"
    assert payload["service"] == "local-transcriber"
    assert "source_revision" in payload


def test_local_transcriber_rich_quiet_suppression_is_preserved():
    sink = io.StringIO()
    console = rich_console.Console(file=sink, quiet=True, force_terminal=False, color_system=None)
    logger = logging.getLogger("local_transcriber.test.quiet")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(_make_handler(console))

    lt_log.stamp_handlers()
    logger.info("suppressed diagnostic")
    assert sink.getvalue() == ""


def test_local_transcriber_rich_record_export_is_preserved():
    sink = io.StringIO()
    console = rich_console.Console(file=sink, record=True, force_terminal=False, color_system=None)
    logger = logging.getLogger("local_transcriber.test.record")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(_make_handler(console))

    lt_log.stamp_handlers()
    logger.info("recorded diagnostic")

    exported = console.export_text().strip()
    assert "recorded diagnostic" in exported
    assert "local-transcriber" in exported
