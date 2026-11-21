"""CLI entry point.

  podcast-digest serve                    # long-running scheduler (the container default)
  podcast-digest run [--dry-run] [--feed=NAME]   # one tick, then exit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tomllib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Never

from pydantic import ValidationError
from pydantic_settings import SettingsError

from . import __version__
from . import log as logmod
from .config import Settings, load_feeds
from .db import Database
from .delivery import reconcile_sent_item
from .emitted import EmittedLog
from .llm import build_completer
from .mail import GraphMailer, Mailer, build_mailer
from .pipeline import _acquire_tick_lock, finalize_deliveries, run_tick
from .scheduler import run_forever

logger = logging.getLogger("podcast_digest")

class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        # Never echo argument values: they can contain credentials.
        logmod.configure("ERROR")
        logmod.event(logger, logging.ERROR, "Invalid command arguments; use --help for usage.",
                     usage=self.format_usage().strip())
        raise SystemExit(2)


_SOURCE_FIELDS = ("rss_url", "search", "youtube", "youtube_video", "audio_url")


def _startup_fields(error: BaseException) -> dict[str, object]:
    """Retain actionable fixed reasons and setting names, never input values."""
    if isinstance(error, ValidationError):
        fields: set[str] = set()
        for item in error.errors(include_input=False, include_context=False):
            fields.update(str(part) for part in item["loc"] if part in Settings.model_fields)
            if item["type"] == "command_required":
                fields.update(("PODCAST_DIGEST_COMMAND", "PODCAST_DIGEST_SUMMARIZER"))
            elif item["type"] in {"secret_file_unreadable", "secret_file_empty"}:
                msg_field = item["msg"].split(" ", 1)[0]
                if msg_field in Settings.model_fields:
                    fields.add(msg_field)
        return {"reason": "invalid_settings", "settings": sorted(fields)}
    if isinstance(error, SettingsError):
        # Only retain a declared field from pydantic-settings' fixed source-error
        # prefix; its exception/cause may contain private JSON or input values.
        match = re.match(r'error (?:parsing|getting) value for field "([A-Z_]+)"', str(error))
        source_fields = [match[1]] if match and match[1] in Settings.model_fields else []
        return {"reason": "invalid_settings", "settings": source_fields}
    message = str(error)
    settings = [name for name in (
        "PODCAST_DIGEST_COMMAND", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "PODCAST_DIGEST_OPENAI_BASE_URL", "PODCAST_DIGEST_SUMMARY_PROMPT_FILE",
        "PODCAST_DIGEST_MAIL_FROM", "PODCAST_DIGEST_MAILER", "PODCAST_DIGEST_SMTP_HOST",
        "PODCAST_DIGEST_GRAPH_TENANT_ID", "PODCAST_DIGEST_GRAPH_CLIENT_ID",
        "PODCAST_DIGEST_GRAPH_CLIENT_SECRET",
    ) if name in message]
    for prefix, reason in (
        ("podcast-digest: summarizer misconfigured:", "summarizer_configuration"),
        ("podcast-digest: summary prompt file is empty:", "summary_prompt_empty"),
        ("podcast-digest: cannot read summary prompt file:", "summary_prompt_unreadable"),
        ("podcast-digest: mail misconfigured:", "mail_configuration"),
    ):
        if isinstance(error, SystemExit) and message.startswith(prefix):
            return {"reason": reason, "settings": settings}
    return {"reason": "command_failed"}


def _startup(settings: Settings, *, need_mail: bool) -> Mailer | None:
    """Check the backends before doing any work; exit with a clear message on
    a configuration error instead of failing every episode later."""
    completer = build_completer(settings)
    problem = completer.config_problem()
    if problem:
        raise SystemExit(logmod.diagnostic(f"podcast-digest: summarizer misconfigured: {problem}"))
    prompt_file = settings.PODCAST_DIGEST_SUMMARY_PROMPT_FILE
    if prompt_file:
        try:
            if not Path(prompt_file).read_text(encoding="utf-8").strip():
                raise SystemExit(logmod.diagnostic(f"podcast-digest: summary prompt file is empty: {prompt_file}"))
        except OSError as exc:
            raise SystemExit(logmod.diagnostic(f"podcast-digest: cannot read summary prompt file: {exc}")) from exc
    mailer: Mailer | None = None
    if need_mail:
        try:
            mailer = build_mailer(settings)
        except ValueError as exc:
            raise SystemExit(logmod.diagnostic(f"podcast-digest: mail misconfigured: {exc}")) from exc
    logger.info(
        "startup summarizer=%s mailer=%s attach=%s webhook=%s",
        completer.label,
        settings.mailer_backend if need_mail else "off (dry run)",
        settings.PODCAST_DIGEST_ATTACH_TRANSCRIPT,
        "on" if settings.webhook_url else "off",
    )
    return mailer


async def _cmd_run(args: argparse.Namespace) -> int:
    settings = Settings()  # type: ignore[call-arg]  # pydantic-settings loads from env
    logmod.configure(settings.LOG_LEVEL)
    settings.ensure_dirs()
    mailer = _startup(settings, need_mail=not args.dry_run)
    feeds_file = load_feeds(settings.PODCAST_DIGEST_CONFIG)
    db = Database(settings.db_path)
    try:
        stats = await run_tick(
            settings=settings, feeds_file=feeds_file,
            db=db, mailer=mailer, dry_run=args.dry_run, only_feed=args.feed,
        )
        logger.info(
            "run.done feeds=%d new=%d emailed=%d failed=%d",
            stats["feeds"], stats["new_episodes"], stats["emailed"], stats["failed"],
        )
        return 0 if stats["failed"] == 0 else 1
    finally:
        db.close()


async def _cmd_serve(_args: argparse.Namespace) -> int:
    settings = Settings()  # type: ignore[call-arg]
    logmod.configure(settings.LOG_LEVEL)
    settings.ensure_dirs()
    mailer = _startup(settings, need_mail=True)
    assert mailer is not None
    db = Database(settings.db_path)
    try:
        await run_forever(settings, db, mailer)
        return 0
    finally:
        db.close()


async def _cmd_check_config(args: argparse.Namespace) -> int:
    """Validate a feeds file without credentials, network calls, or state writes."""
    path: Path = args.path
    try:
        feeds_file = load_feeds(path)
    except (OSError, RuntimeError, tomllib.TOMLDecodeError) as exc:
        print(logmod.diagnostic(f"config invalid: {exc}"), file=sys.stderr)
        return 1

    counts = Counter(
        field
        for feed in feeds_file.feeds
        for field in _SOURCE_FIELDS
        if getattr(feed, field)
    )
    sources = ", ".join(f"{field}={counts[field]}" for field in _SOURCE_FIELDS if counts[field])
    print(
        f"config ok: {path} | feeds={len(feeds_file.feeds)}"
        f" | sources={sources or 'none'} | poll_seconds={feeds_file.poll_seconds}"
        f" | min_minutes={feeds_file.min_minutes}"
    )
    return 0


async def _cmd_delivery(args: argparse.Namespace) -> int:
    settings = Settings()  # type: ignore[call-arg]
    logmod.configure(settings.LOG_LEVEL)
    settings.ensure_dirs()
    lock = _acquire_tick_lock(settings)
    if lock is None:
        print("another tick or recovery command is running; try later", file=sys.stderr)
        return 1
    try:
        db = Database(settings.db_path)
        try:
            if args.action == "list":
                for listed_delivery in db.deliveries():
                    print(
                        f"{listed_delivery.operation_id} {listed_delivery.state} "
                        f"attempts={listed_delivery.attempts} "
                        f"suppression={listed_delivery.suppression_reason or 'none'}"
                    )
                return 0
            row = db.delivery_by_id(args.operation_id)
            if row is None:
                print("unknown delivery operation", file=sys.stderr)
                return 1
            if args.action == "inspect":
                data = asdict(row)
                data["envelope"] = json.loads(row.envelope)
                print(json.dumps(data, indent=2))
                return 0
            mailer = build_mailer(settings)
            if not isinstance(mailer, GraphMailer):
                print("Sent Items reconciliation requires the original Graph backend", file=sys.stderr)
                return 1
            if reconcile_sent_item(db, mailer, row, args.graph_message_id):
                finalize_deliveries(db, EmittedLog.load(settings.PODCAST_DIGEST_EMITTED_LOG))
                print(f"{row.operation_id} accepted")
                return 0
            print(f"{row.operation_id} held: no matching positive Sent Items receipt", file=sys.stderr)
            return 1
        finally:
            db.close()
    finally:
        lock.close()


def main(argv: list[str] | None = None) -> int:
    parser = _ArgumentParser(prog="podcast-digest")
    parser.add_argument("--version", action="version", version=f"podcast-digest {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run one tick and exit")
    p_run.add_argument("--dry-run", action="store_true",
                       help="do everything except send mail")
    p_run.add_argument("--feed", help="restrict to a single feed name")
    p_run.set_defaults(func=_cmd_run)

    p_serve = sub.add_parser("serve", help="run forever; tick on the configured interval")
    p_serve.set_defaults(func=_cmd_serve)

    default_config = Path(os.environ.get("PODCAST_DIGEST_CONFIG", "feeds.toml"))
    p_check = sub.add_parser(
        "check-config",
        help="validate feeds.toml without credentials, network calls, or writes",
    )
    p_check.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=default_config,
        help=f"feeds file (default: {default_config})",
    )
    p_check.set_defaults(func=_cmd_check_config)

    p_delivery = sub.add_parser("delivery", help="inspect and reconcile saved digest operations")
    delivery_sub = p_delivery.add_subparsers(dest="action", required=True)
    p_list = delivery_sub.add_parser("list", help="list saved operation IDs and outcomes")
    p_list.set_defaults(func=_cmd_delivery)
    p_inspect = delivery_sub.add_parser("inspect", help="show the saved envelope and outcome")
    p_inspect.add_argument("operation_id")
    p_inspect.set_defaults(func=_cmd_delivery)
    p_reconcile = delivery_sub.add_parser("reconcile", help="observe positive Graph Sent Items evidence")
    p_reconcile.add_argument("operation_id")
    p_reconcile.add_argument("--graph-message-id", required=True)
    p_reconcile.set_defaults(func=_cmd_delivery)

    args = parser.parse_args(argv)
    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:
        return 130
    except SystemExit as exc:
        if not isinstance(exc.code, str):
            raise
        logmod.configure("ERROR")
        details = _startup_fields(exc)
        logmod.event(logger, logging.ERROR, "Podcast command failed; details withheld.",
                     err_class="SystemExit", reason=details["reason"], settings=details.get("settings", []))
        return 1
    except Exception as exc:
        logmod.configure("ERROR")
        details = _startup_fields(exc)
        logmod.event(logger, logging.ERROR, "Podcast command failed; details withheld.",
                     err_class=type(exc).__name__, reason=details["reason"], settings=details.get("settings", []))
        return 1


if __name__ == "__main__":
    sys.exit(main())
