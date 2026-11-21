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
from dataclasses import asdict
from typing import Any, TypedDict

from . import identity
from .identity import BUILD

# Keys owned by the formatter; extras may never clobber them. A colliding extra
# (e.g. a field literally named "name") is renamed with a trailing underscore.
_IDENTITY_FIELDS = ("service", "version", "source_revision", "instance")
_RESERVED = ("ts", "lvl", "name", "logger", "msg", "exc", *_IDENTITY_FIELDS)
_MCP = False


def identity_fields(record_name: str = "") -> dict[str, str]:
    fields = asdict(BUILD)
    if record_name.split(".")[0] == "local_transcriber":
        from local_transcriber.identity import BUILD as transcriber_build
        fields = asdict(transcriber_build)
    elif _MCP or record_name.split(".")[0] == "podcast_digest_mcp":
        from podcast_digest_mcp.identity import BUILD as mcp_build
        fields = asdict(mcp_build)
    return {**fields, "instance": identity.log_fields()["instance"]}


def diagnostic(message: str, *, record_name: str = "") -> str:
    fields = " ".join(f"{key}={value}" for key, value in identity_fields(record_name).items())
    return "\n".join(f"{line} [{fields}]" for line in message.splitlines() or [""])

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
            kv = " ".join(f"{k + "_" if k in _IDENTITY_FIELDS else k}={_fmt(v)}" for k, v in extras.items())
            base = f"{base} {kv}"
        identity = " ".join(f"{k}={v}" for k, v in identity_fields(record.name).items())
        header = f"{self.formatTime(record)} {record.levelname} {record.name}: "
        if base.startswith(header):
            base = base[len(header):]
        return "\n".join(header + identity + " " + line for line in base.splitlines() or [""])


class _JsonFormatter(_UtcFormatter):
    def __init__(self, name_field: str = "name") -> None:
        super().__init__()
        self.name_field = name_field

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record),
            "lvl": record.levelname,
            self.name_field: record.name,
            "msg": record.getMessage(),
        }
        extras = getattr(record, "extras", None)
        if isinstance(extras, dict):
            for k, v in extras.items():
                payload[f"{k}_" if k in _RESERVED else k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        payload.update(identity_fields(record.name))
        return json.dumps(payload, default=str, ensure_ascii=False)


def _fmt(v: object) -> str:
    if isinstance(v, str) and (" " in v or "=" in v):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


class _SinkFormatter(_UtcFormatter):
    """Attach identity only after an installed sink has formatted/redacted."""

    def __init__(self, original: logging.Formatter | None, *, name_field: str, tty: bool):
        super().__init__()
        self.original = original or logging.Formatter()
        self.name_field = name_field
        self.tty = tty

    def format(self, record: logging.LogRecord) -> str:
        rendered = self.original.format(record)
        try:
            payload = json.loads(rendered)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            # Preserve the sink's names, types and sanitized exception fields.
            fields = identity_fields(record.name)
            for key in fields:
                if key in payload:
                    displaced = key + "_"
                    while displaced in payload:
                        displaced += "_"
                    payload[displaced] = payload[key]
            payload.update(fields)
            return json.dumps(payload, default=str, ensure_ascii=False)
        if self.tty:
            header = f"{self.formatTime(record)} {record.levelname} {record.name}: "
            identity = " ".join(f"{k}={v}" for k, v in identity_fields(record.name).items())
            return "\n".join(header + identity + " " + line for line in rendered.splitlines() or [""])
        return json.dumps({"ts": self.formatTime(record), "lvl": record.levelname,
                           self.name_field: record.name, "msg": rendered,
                           **identity_fields(record.name)}, ensure_ascii=False)


def _native_console_dispatch(console: Any, render: Any, hook: Any) -> Any:
    """Scope render provenance to this thread without replacing print dispatch."""
    from contextlib import contextmanager
    from threading import local

    @contextmanager
    def dispatch() -> Any:
        key = "_build_identity_render_dispatch"
        with console._lock:
            state = vars(console).get(key)
            if state is None:
                missing = object()
                state = {
                    "local": local(),
                    "users": 0,
                    "render": console.render,
                    "saved_render": vars(console).get("render", missing),
                    "missing": missing,
                    "hooks": console._render_hooks,
                }

                def current() -> Any:
                    stack = getattr(state["local"], "stack", ())
                    return stack[-1] if stack else None

                def dispatch_render(*args: Any, **kwargs: Any) -> Any:
                    context = current()
                    if context is None:
                        yield from state["render"](*args, **kwargs)
                        return
                    stack = state["local"].stack
                    stack.append(None)
                    try:
                        yield from context[0](*args, **kwargs)
                    finally:
                        stack.pop()

                class DispatchHook:
                    def __init__(self, original: Any) -> None:
                        self.original = original

                    def process_renderables(self, renderables: list[Any]) -> list[Any]:
                        context = current()
                        target = self.original
                        if context is not None:
                            target = context[1](target)
                        return target.process_renderables(renderables)

                class DispatchHooks:
                    def __getattr__(self, name: str) -> Any:
                        return getattr(state["hooks"], name)

                    def __getitem__(self, index: Any) -> Any:
                        items = state["hooks"][index]
                        if current() is None:
                            return items
                        if isinstance(index, slice):
                            return [DispatchHook(item) for item in items]
                        return DispatchHook(items)

                    def __len__(self) -> int:
                        return len(state["hooks"])

                    def __iter__(self) -> Any:
                        return iter(self[:])

                state["dispatch_render"] = dispatch_render
                state["dispatch_hooks"] = DispatchHooks()
                console.render = dispatch_render
                console._render_hooks = state["dispatch_hooks"]
                vars(console)[key] = state
            state["users"] += 1
            stack = getattr(state["local"], "stack", None)
            if stack is None:
                stack = state["local"].stack = []
            stack.append((render, hook))
        try:
            yield
        finally:
            with console._lock:
                stack.pop()
                state["users"] -= 1
                if not state["users"]:
                    if console.render is state["dispatch_render"]:
                        if state["saved_render"] is state["missing"]:
                            del console.render
                        else:
                            console.render = state["saved_render"]
                    if console._render_hooks is state["dispatch_hooks"]:
                        console._render_hooks = state["hooks"]
                    del vars(console)[key]

    return dispatch()


class _RichSink(logging.Handler):
    """Attach identity only after Rich applies its rendering and traceback policy."""

    def __init__(self, original: Any, identity_formatter: logging.Formatter) -> None:
        super().__init__(original.level)
        self.original = original
        self.filters = list(original.filters)
        self.setFormatter(original.formatter)
        self.identity_formatter = identity_formatter

    @property
    def stream(self) -> Any:
        return self.original.console.file

    def setFormatter(self, formatter: logging.Formatter | None) -> None:
        super().setFormatter(formatter)
        self.original.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        owner = self
        original: Any = self.original
        original.acquire()
        console = original.console

        class RecordConsole:
            def __getattr__(self, name: str) -> Any:
                return getattr(console, name)

            def __setattr__(self, name: str, value: Any) -> None:
                setattr(console, name, value)

            def __delattr__(self, name: str) -> None:
                delattr(console, name)

            def print(self, *objects: Any, **kwargs: Any) -> None:
                from rich.control import Control, ControlType
                from rich.live import Live
                from rich.live_render import LiveRender
                from rich.segment import Segment

                host_renderables: set[int] = set()
                markers: dict[int, tuple[Any, bool]] = {}

                class Hook:
                    def __init__(self, original_hook: Any) -> None:
                        self.original_hook = original_hook

                    def process_renderables(self, renderables: list[Any]) -> list[Any]:
                        incoming = {id(item) for item in renderables}
                        result = self.original_hook.process_renderables(renderables)
                        if isinstance(self.original_hook, Live):
                            host_renderables.update(
                                id(item)
                                for item in result
                                if id(item) not in incoming
                                and isinstance(item, (Control, LiveRender))
                            )
                        return result

                def marker(owned: bool) -> Any:
                    # Fresh tuple identity carries provenance through native cropping.
                    controls = [(ControlType.BELL,)]
                    token = tuple(controls)
                    markers[id(token)] = (token, owned)
                    return Segment("", None, token)

                def render(renderable: Any, options: Any = None) -> Any:
                    yield marker(id(renderable) not in host_renderables)
                    yield from console.render(renderable, options)
                    yield marker(True)

                with console:
                    # Rich owns a thread-local buffer; isolate this print from host
                    # output and restore it even if a hook or renderer raises.
                    prefix = console._buffer[:]
                    console._buffer.clear()
                    try:
                        with _native_console_dispatch(console, render, Hook):
                            console.print(*objects, **kwargs)
                        raw = console._buffer[:]
                    finally:
                        console._buffer[:] = prefix
                    groups: list[tuple[bool, list[Any]]] = []
                    owned = True
                    for segment in raw:
                        provenance = markers.get(id(segment.control))
                        if provenance is not None:
                            owned = provenance[1]
                        elif groups and groups[-1][0] == owned:
                            groups[-1][1].append(segment)
                        else:
                            groups.append((owned, [segment]))
                    for owned, segments in groups:
                        if not owned:
                            console._buffer.extend(segments)
                            continue
                        rendered = console._render_buffer(segments)
                        if not rendered:
                            continue
                        retained = logging.makeLogRecord(
                            {
                                "name": record.name,
                                "levelno": record.levelno,
                                "levelname": record.levelname,
                                "created": record.created,
                                "msecs": record.msecs,
                                "msg": rendered.rstrip("\n"),
                                "args": (),
                            }
                        )
                        attributed = owner.identity_formatter.format(retained)
                        try:
                            structured = json.loads(attributed)
                        except (ValueError, TypeError):
                            structured = None
                        if getattr(owner.identity_formatter, "tty", False) and not isinstance(
                            structured, dict
                        ):
                            # These formatters declare a constant prefix
                            # for every native line.
                            native_line = (rendered.rstrip("\n").splitlines() or [""])[0]
                            attributed_line = attributed.splitlines()[0]
                            header = (
                                attributed_line[: len(attributed_line) - len(native_line)]
                                if native_line
                                else attributed_line
                            )
                            retained_segments = segments[:]
                            for index in range(len(retained_segments) - 1, -1, -1):
                                segment = retained_segments[index]
                                if segment.control:
                                    continue
                                trimmed = segment.text.rstrip("\n")
                                retained_segments[index] = Segment(trimmed, segment.style)
                                if trimmed or trimmed == segment.text:
                                    break
                            line_start = True
                            for segment in retained_segments:
                                if segment.control:
                                    console._buffer.append(segment)
                                    continue
                                for part in segment.text.splitlines(keepends=True):
                                    if line_start:
                                        console._buffer.append(Segment(header))
                                    console._buffer.append(Segment(part, segment.style))
                                    line_start = part.endswith("\n")
                            console._buffer.append(Segment.line())
                        else:
                            from rich.ansi import AnsiDecoder

                            text = AnsiDecoder().decode_line(attributed + "\n")
                            text.end = ""
                            console._buffer.extend(
                                console.render(
                                    text, console.options.update(no_wrap=True, overflow="ignore")
                                )
                            )

        try:
            original.console = RecordConsole()
            original.emit(record)
        finally:
            original.console = console
            original.release()


def stamp_handlers(*, mcp: bool = False) -> None:
    """Keep streams, filters, levels and propagation on existing owned/SDK sinks."""
    names = {"podcast_digest", "podcast_digest_mcp", "local_transcriber", "fastmcp", "mcp", "uvicorn"}
    loggers = [logging.getLogger(), *(logger for name, logger in logging.Logger.manager.loggerDict.items()
               if isinstance(logger, logging.Logger) and name.split(".")[0] in names)]
    rich = sys.modules.get("rich.logging")
    replacements: dict[logging.Handler, logging.Handler] = {}
    for logger in loggers:
        for index, existing in enumerate(logger.handlers):
            handler = replacements.get(existing, existing)
            if rich is not None and isinstance(handler, rich.RichHandler):
                tty = not mcp and bool(getattr(handler.console.file, "isatty", lambda: False)())
                handler = _RichSink(handler, _SinkFormatter(logging.Formatter(), name_field="logger" if mcp else "name", tty=tty))
                replacements[existing] = handler
            logger.handlers[index] = handler
            if isinstance(handler, _RichSink):
                continue
            if isinstance(handler.formatter, (_JsonFormatter, _KVFormatter, _SinkFormatter)):
                continue
            tty = not mcp and bool(getattr(getattr(handler, "stream", None), "isatty", lambda: False)())
            handler.setFormatter(_SinkFormatter(handler.formatter, name_field="logger" if mcp else "name", tty=tty))


def configure(level: str | None = None, *, mcp: bool = False) -> None:
    """Idempotent setup; installed sinks retain their suppression/redaction policy."""
    global _MCP
    _MCP = mcp
    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        if not mcp and sys.stderr.isatty():
            h.setFormatter(_KVFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        else:
            h.setFormatter(_JsonFormatter("logger" if mcp else "name"))
        root.addHandler(h)
        root.setLevel(level)
    if not mcp:
        for name in ("httpx", "httpcore", "anthropic._base_client"):
            noisy = logging.getLogger(name)
            if noisy.level == logging.NOTSET:
                noisy.setLevel(logging.WARNING)
    stamp_handlers(mcp=mcp)


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
