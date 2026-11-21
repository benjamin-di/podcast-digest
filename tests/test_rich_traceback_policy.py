"""Rich owns traceback disclosure before build identity is attached."""

import io
import json
import logging
import sys

import pytest
from rich.console import Console
from rich.logging import RichHandler

from podcast_digest import log


@pytest.mark.parametrize("suppressed", [False, True])
@pytest.mark.parametrize("with_formatter", [False, True])
@pytest.mark.parametrize("tty", [False, True])
def test_attribution_retains_rich_traceback_policy(
    tmp_path, monkeypatch, suppressed, with_formatter, tty
):
    source = tmp_path / "synthetic_failure.py"
    source.write_text(
        "\n".join(
            [
                f"def policy_frame_{i}():\n    return policy_frame_{i + 1}()"
                + ("  # HIDDEN_MIDDLE_SOURCE" if i == 4 else "")
                for i in range(8)
            ]
            + [
                'def policy_frame_8():\n'
                '    raise RuntimeError("synthetic failure")  # SUPPRESSED_SOURCE'
            ]
        )
        + "\n"
    )
    namespace = {}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    try:
        namespace["policy_frame_0"]()
    except RuntimeError:
        exc_info = sys.exc_info()

    class Stream(io.StringIO):
        def isatty(self):
            return tty

    output = Stream()
    console = Console(file=output, width=240, force_terminal=False, color_system=None)
    original = RichHandler(
        console=console,
        show_time=False,
        show_level=False,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_suppress=[str(tmp_path)] if suppressed else [],
        tracebacks_max_frames=3,
    )
    format_calls = []

    class Redact(logging.Formatter):
        def format(self, record):
            format_calls.append(record)
            return super().format(record).replace("synthetic-private", "[redacted]")

    if with_formatter:
        original.setFormatter(Redact("%(message)s"))

    def record():
        item = logging.LogRecord(
            "fastmcp.policy.synthetic", logging.ERROR, __file__, 1,
            "synthetic-private diagnostic", (), exc_info,
        )
        item.extras = {"hidden_extra": "must-never-be-restored"}
        return item

    original.handle(record())
    retained = output.getvalue().rstrip("\n")
    assert "HIDDEN_MIDDLE_SOURCE" not in retained
    assert ("SUPPRESSED_SOURCE" not in retained) is suppressed
    assert "frames hidden" in retained
    output.seek(0)
    output.truncate(0)
    format_calls.clear()

    logger = logging.getLogger("fastmcp.policy.synthetic")
    monkeypatch.setattr(logger, "handlers", [original])
    monkeypatch.setattr(logger, "propagate", False)
    log.stamp_handlers()
    log.stamp_handlers()
    logger.handlers[0].handle(record())

    attributed = output.getvalue()
    assert "HIDDEN_MIDDLE_SOURCE" not in attributed
    assert ("SUPPRESSED_SOURCE" not in attributed) is suppressed
    assert "must-never-be-restored" not in attributed
    assert len(format_calls) == int(with_formatter)
    if tty:
        lines = attributed.splitlines()
        expected = retained.splitlines()
        assert len(lines) == len(expected)
        assert all(
            actual.endswith(rendered) and "source_revision=" in actual
            for actual, rendered in zip(lines, expected, strict=True)
        )
    else:
        payload = json.loads(attributed)
        assert payload["msg"] == retained
        assert payload["source_revision"]
        assert "exc" not in payload and "hidden_extra" not in payload

def test_shared_rich_sink_retains_policy_and_console_destination(monkeypatch):
    first, second = io.StringIO(), io.StringIO()
    console = Console(file=first, width=120, force_terminal=False)
    original = RichHandler(
        console=console, show_time=False, show_level=False, show_path=False
    )
    original.setLevel(logging.WARNING)
    seen = []

    def allowed(record):
        seen.append(record.msg)
        return record.msg != "drop"

    original.addFilter(allowed)
    logger = logging.getLogger("fastmcp.policy.shared")
    peer = logging.getLogger("mcp.policy.shared")
    for target in (logger, peer):
        monkeypatch.setattr(target, "handlers", [original])
        monkeypatch.setattr(target, "level", logging.DEBUG)
        monkeypatch.setattr(target, "propagate", False)
    log.stamp_handlers()
    log.stamp_handlers()
    adopted = logger.handlers[0]
    assert adopted is peer.handlers[0]
    assert adopted.level == logging.WARNING
    assert adopted.filters == [allowed]
    logger.debug("below threshold")
    logger.warning("drop")
    console.quiet = True
    logger.warning("quiet")
    assert first.getvalue() == ""
    console.quiet = False
    adopted.setFormatter(logging.Formatter("sanitized", validate=False))
    logger.warning("synthetic-private")
    assert "sanitized" in first.getvalue() and "synthetic-private" not in first.getvalue()
    previous = first.getvalue()
    console.file = second
    peer.warning("synthetic-private again")
    assert first.getvalue() == previous
    assert "sanitized" in second.getvalue() and "synthetic-private" not in second.getvalue()
    assert adopted.stream is second
    assert seen == ["drop", "quiet", "synthetic-private", "synthetic-private again"]


@pytest.mark.parametrize("operation", ["write", "flush"])
def test_rich_sink_failure_uses_logging_error_policy(monkeypatch, operation):
    class Broken(io.StringIO):
        def write(self, text):
            if operation == "write":
                raise OSError("synthetic sink failure")
            return super().write(text)

        def flush(self):
            if operation == "flush":
                raise OSError("synthetic sink failure")
            return super().flush()

    original = RichHandler(console=Console(file=Broken()))
    logger = logging.getLogger("fastmcp.policy.broken")
    monkeypatch.setattr(logger, "handlers", [original])
    log.stamp_handlers()
    failures = []
    monkeypatch.setattr(original, "handleError", failures.append)
    record = logging.LogRecord(logger.name, logging.ERROR, __file__, 1, "retained", (), None)
    logger.handlers[0].handle(record)
    assert failures == [record]
