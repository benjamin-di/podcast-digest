#!/usr/bin/env python3
"""Offline acceptance of the installed stamps inside an exact published image."""
from __future__ import annotations

import sys
from importlib.metadata import version


def check(variant: str, revision: str, product_version: str, transcriber_version: str) -> None:
    from local_transcriber.identity import BUILD as transcriber

    assert transcriber.service == "local-transcriber"
    assert transcriber.version == version("local-transcriber") == transcriber_version
    assert transcriber.source_revision == revision
    if variant in {"podcast", "mcp"}:
        from podcast_digest.identity import BUILD as worker
        from podcast_digest_mcp.identity import BUILD as mcp

        assert worker.service == "podcast-digest"
        assert mcp.service == "podcast-digest-mcp"
        assert worker.version == mcp.version == version("podcast-digest") == product_version
        assert worker.source_revision == mcp.source_revision == revision
    elif variant != "transcriber":
        raise ValueError("Unknown image variant")


if __name__ == "__main__":
    check(*sys.argv[1:])
