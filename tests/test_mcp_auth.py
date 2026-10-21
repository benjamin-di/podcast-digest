"""Bearer auth on the MCP endpoint (FastMCP's StaticTokenVerifier)."""
from __future__ import annotations

import pytest

pytest.importorskip("fastmcp")

from starlette.testclient import TestClient

from podcast_digest_mcp.server import build_app

INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
               "clientInfo": {"name": "auth-test", "version": "1"}},
}
ACCEPT = {"Accept": "application/json, text/event-stream"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("PODCAST_MCP_BEARER_TOKEN", "s3cret")
    monkeypatch.setenv("PODCAST_DIGEST_CONFIG", str(tmp_path / "feeds.toml"))
    _mcp, app = build_app()
    with TestClient(app) as c:
        yield c


def _post(client, headers=None, url="/mcp"):
    return client.post(url, json=INIT, headers={**ACCEPT, **(headers or {})})


def test_header_bearer_is_accepted(client):
    assert _post(client, {"Authorization": "Bearer s3cret"}).status_code == 200


def test_query_token_is_rejected(client):
    # The gateway stopped accepting ?token= (2026-07-11); tokens in URLs leak
    # into access logs, so the backend accepts the header only.
    assert _post(client, url="/mcp?token=s3cret").status_code == 401


@pytest.mark.parametrize("value", ["Bearer nope", "Basic s3cret", ""])
def test_wrong_or_missing_token_is_401(client, value):
    headers = {"Authorization": value} if value else {}
    assert _post(client, headers).status_code == 401


def test_non_ascii_token_is_401_not_500(client):
    assert _post(client, {"Authorization": "Bearer sékrit".encode()}).status_code == 401


def test_health_is_open(client):
    assert client.get("/health").status_code == 200
