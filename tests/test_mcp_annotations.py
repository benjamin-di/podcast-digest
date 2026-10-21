"""Every MCP tool declares whether it reads or changes state."""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastmcp")

from podcast_digest_mcp.server import build_app


def test_tool_annotations(monkeypatch, tmp_path):
    monkeypatch.setenv("PODCAST_MCP_BEARER_TOKEN", "s3cret")
    monkeypatch.setenv("PODCAST_DIGEST_CONFIG", str(tmp_path / "feeds.toml"))
    mcp, _app = build_app()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    read_only = {name for name, t in tools.items() if t.annotations and t.annotations.read_only_hint}
    assert read_only == {"preview", "subscription_list"}
    assert all(t.annotations is not None for t in tools.values())
    assert tools["subscription_remove"].annotations.destructive_hint is True
    assert tools["ingest"].annotations.destructive_hint is False
