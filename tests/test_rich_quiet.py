"""Regression for dynamic Rich console suppression during build attribution."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module,function", [('podcast_digest.log', 'stamp_handlers'), ('local_transcriber.log', 'stamp_handlers')])
def test_rich_console_quiet_policy_survives_adaptation(module, function):
    program = r"""
import importlib
import io
import logging
import sys
from rich.console import Console
from rich.logging import RichHandler

module = importlib.import_module(sys.argv[1])
stream = io.StringIO()
console = Console(file=stream, quiet=True)
handler = RichHandler(console=console)
logger = logging.getLogger("local_transcriber" if sys.argv[1] == "local_transcriber.log" else "fastmcp")
logger.handlers, logger.propagate, logger.level = [handler], False, logging.DEBUG
function = getattr(module, sys.argv[2])
function("dine") if sys.argv[1] == "dine_mcp.logs" else function()
logger.critical("must remain silent")
assert stream.getvalue() == ""
console.quiet = False
logger.critical("visible diagnostic")
visible = stream.getvalue()
assert "visible diagnostic" in visible and "source_revision" in visible
console.quiet = True
logger.critical("must remain silent again")
assert stream.getvalue() == visible
"""
    result = subprocess.run([sys.executable, "-c", program, module, function],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
