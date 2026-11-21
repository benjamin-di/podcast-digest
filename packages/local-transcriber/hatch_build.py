"""Embed the installed build identity into noneditable wheels and source distributions."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

MISSING = "Missing build identity: build a clean Git checkout or run embed-build-identity.py first"
UNCOMMITTED = "Build identity requires committed source: commit it or use an editable install"


class BuildIdentityHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version == "editable":
            return
        root = Path(self.root)
        stamps: dict[str, str] = self.config["stamps"]
        revision = _clean_commit(root)
        if revision is None:
            # Source distributions and image contexts ship the stamps written before packaging.
            for path in stamps.values():
                if not (root / path).is_file():
                    raise RuntimeError(MISSING)
                build_data["force_include"][str(root / path)] = path
            return
        self._stamps = tempfile.TemporaryDirectory()
        product_version = self.metadata.version
        for service, path in stamps.items():
            stamp = Path(self._stamps.name, f"{service}.json")
            identity = {"service": service, "version": product_version, "source_revision": revision}
            stamp.write_text(json.dumps(identity) + "\n", encoding="utf-8")
            build_data["force_include"][str(stamp)] = path

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        if hasattr(self, "_stamps"):
            self._stamps.cleanup()


def _clean_commit(root: Path) -> str | None:
    if (root / "PKG-INFO").is_file():
        return None
    checkout = any((directory / ".git").exists() for directory in (root, *root.parents))
    # Only the checkout being built may identify it; inherited GIT_DIR and similar would relabel it.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True)

    try:
        if git("ls-files", "--error-unmatch", "pyproject.toml").returncode:
            if checkout:
                raise RuntimeError("Build identity cannot verify this Git checkout")
            return None
    except OSError as error:
        if checkout:
            raise RuntimeError("Build identity cannot verify this Git checkout") from error
        return None
    status = git("status", "--porcelain", "--untracked-files=all", "--", ".")
    untracked = git("ls-files", "--others", "--exclude-per-directory=.gitignore", "--", ".")
    tracked = git("ls-files", "-v", "-z", "--", ".")
    if tracked.returncode or any(
        entry and (entry[0].islower() or entry[0] == "S") for entry in tracked.stdout.split("\0")
    ):
        raise RuntimeError(
            "Build identity cannot verify index-hidden source: disable sparse checkout, "
            "skip-worktree and assume-unchanged"
        )
    head = git("rev-parse", "--verify", "HEAD^{commit}")
    failed = any(result.returncode for result in (status, untracked, head))
    if failed or status.stdout or untracked.stdout:
        raise RuntimeError(UNCOMMITTED)
    return head.stdout.strip()
