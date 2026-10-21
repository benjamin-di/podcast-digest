"""FastMCP server entry: wires HTTP clients, tools and bearer-token auth.

Auth is FastMCP's static bearer verifier on the MCP endpoint
(PODCAST_MCP_BEARER_TOKEN); /health stays open for container healthchecks. Put
it behind a TLS reverse proxy to reach it from a phone or claude.ai.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from starlette.requests import Request
from starlette.responses import JSONResponse

from podcast_digest import USER_AGENT, __version__
from podcast_digest.net import validate_async_request, validate_request

from . import tools

logger = logging.getLogger(__name__)




def build_app():
    """Construct the FastMCP server and the ASGI app to serve."""
    token = os.environ.get("PODCAST_MCP_BEARER_TOKEN")
    if not token:
        raise RuntimeError("PODCAST_MCP_BEARER_TOKEN must be set")

    feeds_path = Path(os.environ.get("PODCAST_DIGEST_CONFIG", "/app/config/feeds.toml"))

    http_async = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        event_hooks={"request": [validate_async_request]},
    )
    http_sync = httpx.Client(
        timeout=httpx.Timeout(15.0, connect=10.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        event_hooks={"request": [validate_request]},
    )

    mcp = FastMCP(
        "podcast-digest",
        version=__version__,  # serverInfo: this package's version, not FastMCP's
        # Header bearer only: tokens in URLs end up in access logs.
        auth=StaticTokenVerifier(tokens={token: {"client_id": "bearer", "scopes": []}}),
        # Unexpected exceptions reach clients as a generic error, not their
        # raw text; ToolError messages and returned error dicts are unchanged.
        mask_error_details=True,
        # Tool lists only change on deploy: 10-minute private cache hint for
        # clients on the 2026-07-28 protocol.
        cache_ttl=600,
    )

    @mcp.custom_route("/health", methods=["GET"])
    async def _health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    tools.register(mcp, feeds_path=feeds_path, http=http_async, http_sync=http_sync)

    # Stateless streamable HTTP with plain JSON replies: no in-memory MCP
    # sessions to lose on every deploy or container restart, and no SSE reply
    # stream that a reverse proxy can cut (seen as ~1 in 500 calls hanging until
    # timeout behind one). Nothing here needs a session.
    app = mcp.http_app(
        stateless_http=True,
        json_response=True,
        # Behind a reverse proxy (foreign Host) and plain health probes: explicit
        # so a future FastMCP default of Host/Origin checks can't reject them.
        host_origin_protection=False,
    )
    return mcp, app


def main() -> None:
    import uvicorn

    # stdlib logging wants an UPPER-case level name ("INFO"); uvicorn wants it
    # lower-case ("info"). Normalise the single LOG_LEVEL env for both so a
    # lower-case value (as the compose sets) doesn't crash basicConfig.
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=log_level.upper(),
        format='{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s","msg":%(message)r}',
    )
    _mcp, app = build_app()
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level=log_level.lower(),
    )


if __name__ == "__main__":
    main()
