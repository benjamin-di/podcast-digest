"""Cross-process "process now" signal between the MCP and the digest scheduler.

The MCP and the digest are separate containers that share only the config
directory (MCP read-write, digest read-only). To let an ingest kick off
processing immediately — instead of waiting up to `poll_seconds` for the next
scheduled tick — the MCP bumps a sentinel file's content to a fresh `time_ns()`
token; the digest scheduler reads that token every few seconds and runs a tick
early when it advances past the token it last consumed.

Race safety: this only *wakes* the scheduler early. It does not run the pipeline
itself. The existing per-tick flock (`pipeline._acquire_tick_lock`) still makes
ticks single-flight, so a triggered tick and a scheduled tick can never run
concurrently — the loser returns ``skipped`` and retries. The MCP always writes
feeds.toml *before* signalling, so the feed is visible when the tick loads it;
a signal that lands mid-tick simply advances the token again and the scheduler
runs one more tick (the work is idempotent + DB-deduped either way).

Keep the sentinel out of any sync: if the directory holding ``feeds.toml`` is
replicated (say, a git-synced folder shared between hosts), its per-signal
``time_ns()`` token churn would conflict between copies and could break parsing
of the files next to it. The filename ends in ``.tmp`` so a ``*.tmp`` ignore rule
keeps it untracked — the sentinel is pure ephemeral local IPC.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# Ends in `.tmp` so a `*.tmp` ignore rule keeps it out of any sync — see the
# module docstring.
WAKE_FILENAME = ".process-now.tmp"


def wake_path(config_file: Path | str) -> Path:
    """The sentinel lives next to feeds.toml so it rides the same config mount."""
    return Path(config_file).parent / WAKE_FILENAME


def signal_wake(config_file: Path | str) -> int:
    """Bump the sentinel to a fresh monotonic-ish token. Returns the token.

    Atomic (tempfile + os.replace) so the scheduler never reads a torn value.
    """
    p = wake_path(config_file)
    token = time.time_ns()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(str(token), encoding="utf-8")
    os.replace(tmp, p)
    return token


def read_wake(config_file: Path | str) -> int:
    """Read the current token, or 0 if the sentinel is missing/unreadable."""
    try:
        return int(wake_path(config_file).read_text(encoding="utf-8").strip() or "0")
    except (FileNotFoundError, ValueError, OSError):
        return 0
