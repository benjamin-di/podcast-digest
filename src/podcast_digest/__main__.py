"""CLI entry point.

  podcast-digest serve                    # long-running scheduler (the container default)
  podcast-digest run [--dry-run] [--feed=NAME]   # one tick, then exit
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tomllib
from collections import Counter
from pathlib import Path

from . import __version__
from . import log as logmod
from .config import Settings, load_feeds
from .db import Database
from .llm import build_completer
from .mail import Mailer, build_mailer
from .pipeline import run_tick
from .scheduler import run_forever

logger = logging.getLogger("podcast_digest")

_SOURCE_FIELDS = ("rss_url", "search", "youtube", "youtube_video", "audio_url")


def _startup(settings: Settings, *, need_mail: bool) -> Mailer | None:
    """Check the backends before doing any work; exit with a clear message on
    a configuration error instead of failing every episode later."""
    completer = build_completer(settings)
    problem = completer.config_problem()
    if problem:
        raise SystemExit(f"podcast-digest: summarizer misconfigured: {problem}")
    prompt_file = settings.PODCAST_DIGEST_SUMMARY_PROMPT_FILE
    if prompt_file:
        try:
            if not Path(prompt_file).read_text(encoding="utf-8").strip():
                raise SystemExit(f"podcast-digest: summary prompt file is empty: {prompt_file}")
        except OSError as exc:
            raise SystemExit(f"podcast-digest: cannot read summary prompt file: {exc}") from exc
    mailer: Mailer | None = None
    if need_mail:
        try:
            mailer = build_mailer(settings)
        except ValueError as exc:
            raise SystemExit(f"podcast-digest: mail misconfigured: {exc}") from exc
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
        print(f"config invalid: {exc}", file=sys.stderr)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="podcast-digest")
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

    args = parser.parse_args(argv)
    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
