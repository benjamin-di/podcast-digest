"""The MCP endpoint serves stateless streamable HTTP with plain JSON replies."""
from __future__ import annotations

import pytest

pytest.importorskip("fastmcp")

from starlette.testclient import TestClient

from podcast_digest_mcp.server import build_app

INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
               "clientInfo": {"name": "transport-test", "version": "1"}},
}


def test_mcp_is_stateless_json_and_authenticated(monkeypatch, tmp_path):
    monkeypatch.setenv("PODCAST_MCP_BEARER_TOKEN", "s3cret")
    monkeypatch.setenv("PODCAST_DIGEST_CONFIG", str(tmp_path / "feeds.toml"))
    _mcp, app = build_app()
    accept = {"Accept": "application/json, text/event-stream"}
    auth = {**accept, "Authorization": "Bearer s3cret"}
    with TestClient(app) as client:
        assert client.post("/mcp", json=INIT, headers=accept).status_code == 401
        response = client.post("/mcp", json=INIT, headers=auth)
        assert response.status_code == 200, response.text
        # No session to lose on restart; plain JSON, not SSE.
        assert "mcp-session-id" not in response.headers
        assert response.headers["content-type"].startswith("application/json")
        tools = client.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            headers={**auth, "mcp-protocol-version": "2025-03-26"},
        )
        assert tools.status_code == 200, tools.text
        assert tools.json()["result"]["tools"]


def test_foreign_host_and_origin_are_served(monkeypatch, tmp_path):
    """The gateway forwards its own Host; Host/Origin checks stay off."""
    monkeypatch.setenv("PODCAST_MCP_BEARER_TOKEN", "s3cret")
    monkeypatch.setenv("PODCAST_DIGEST_CONFIG", str(tmp_path / "feeds.toml"))
    _mcp, app = build_app()
    foreign = {
        "Accept": "application/json, text/event-stream",
        "Authorization": "Bearer s3cret",
        "Host": "mcp.example.com",
        "Origin": "https://elsewhere.example",
    }
    with TestClient(app) as client:
        response = client.post("/mcp", json=INIT, headers=foreign)
        assert response.status_code == 200, response.text
        assert response.json()["result"]["serverInfo"]
        health = client.get("/health", headers={"Host": "10.0.0.7:8000"})
        assert health.status_code == 200, health.text
