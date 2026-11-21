#!/usr/bin/env python3
"""Validate publisher inputs and stamp one installed product before packaging."""
from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path


def embed(root: Path, product: str, version: str, revision: str) -> None:
    if product == "podcast-digest":
        project = root
        targets = {"podcast-digest": root / "src/podcast_digest",
                   "podcast-digest-mcp": root / "src/podcast_digest_mcp"}
    elif product == "local-transcriber":
        project = root / "packages/local-transcriber"
        targets = {product: project / "src/local_transcriber"}
    else:
        raise ValueError("Unknown build identity product")
    declared = tomllib.loads((project / "pyproject.toml").read_text())["project"]["version"]
    normalized = version.strip().removeprefix("v")
    if not normalized or normalized != declared:
        raise ValueError("VERSION must match the product version in pyproject.toml")
    if not re.fullmatch(r"(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})", revision):
        raise ValueError("VCS_REF must be a full 40- or 64-character hexadecimal source revision")
    for service, target in targets.items():
        (target / "build_identity.json").write_text(json.dumps({
            "service": service, "version": normalized, "source_revision": revision.lower()
        }) + "\n", encoding="utf-8")


if __name__ == "__main__":
    embed(Path.cwd(), *sys.argv[1:])
