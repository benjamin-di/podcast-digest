"""Installed application identity, read offline without runtime relabeling."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class BuildIdentity:
    service: str
    version: str
    source_revision: str


def _object(pairs: list[tuple[str, object]]) -> dict:
    data = dict(pairs)
    if len(data) != len(pairs):
        raise ValueError("Duplicate application build identity fields")
    return data


def _source_development() -> bool:
    try:
        installed = distribution("local-transcriber")
    except PackageNotFoundError:
        return True
    try:
        direct = json.loads(installed.read_text("direct_url.json") or "{}")
    except (TypeError, ValueError):
        return False
    return (
        isinstance(direct, dict)
        and isinstance(direct.get("dir_info"), dict)
        and direct["dir_info"].get("editable") is True
    )


def load_build(path: Path | None = None) -> BuildIdentity:
    try:
        installed_version = version("local-transcriber")
    except PackageNotFoundError:
        installed_version = "dev"
    path = path if path is not None else Path(__file__).with_name("build_identity.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    except FileNotFoundError:
        if not _source_development():
            raise ValueError("Missing application build identity in installed package") from None
        return BuildIdentity("local-transcriber", installed_version, "unknown")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid application build identity") from exc
    if (
        not isinstance(data, dict)
        or set(data) != {"service", "version", "source_revision"}
        or data["service"] != "local-transcriber"
        or data["version"] != installed_version
        or not isinstance(data["source_revision"], str)
        or not re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", data["source_revision"])
    ):
        raise ValueError("Application build identity conflicts with the installed package")
    return BuildIdentity(**data)


BUILD = load_build()


_INSTANCE = uuid4().hex


def _after_fork() -> None:
    global _INSTANCE
    _INSTANCE = uuid4().hex


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def log_fields() -> dict[str, str]:
    return {**asdict(BUILD), "instance": _INSTANCE}
