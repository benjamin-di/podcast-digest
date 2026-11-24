"""FastMCP server entry: wires HTTP clients, tools and bearer-token auth.

Auth is FastMCP's static bearer verifier on the MCP endpoint
(PODCAST_MCP_BEARER_TOKEN); /health stays open for container healthchecks. Put
it behind a TLS reverse proxy to reach it from a phone or claude.ai.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import httpx
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from starlette.requests import Request
from starlette.responses import JSONResponse

from podcast_digest import USER_AGENT
from podcast_digest.log import configure, diagnostic, stamp_handlers
from podcast_digest.net import build_async_client, build_sync_client

from . import tools
from .identity import BUILD

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "Manage podcast and video digest subscriptions and search saved summaries/transcripts. "
    "Use preview to inspect how a URL will be classified and its detected language before calling ingest. "
    "ingest saves a subscription or one-shot episode and wakes the background digest pipeline asynchronously; "
    "digest_status shows pipeline progress and recent deliveries."
)




def build_app():
    """Construct the FastMCP server and the ASGI app to serve."""
    token = os.environ.get("PODCAST_MCP_BEARER_TOKEN")
    if not token:
        raise RuntimeError(diagnostic("PODCAST_MCP_BEARER_TOKEN must be set", record_name=__name__))

    feeds_path = Path(os.environ.get("PODCAST_DIGEST_CONFIG", "/app/config/feeds.toml"))
    data_dir_env = os.environ.get("PODCAST_DIGEST_DATA_DIR", "").strip()
    data_dir = Path(data_dir_env) if data_dir_env else None

    http_async = build_async_client(
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    http_sync = build_sync_client(
        timeout=httpx.Timeout(15.0, connect=10.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )

    mcp = FastMCP(
        "podcast-digest",
        version=BUILD.version,  # serverInfo: this package's version, not FastMCP's
        instructions=INSTRUCTIONS,
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

    tools.register(
        mcp, feeds_path=feeds_path, http=http_async, http_sync=http_sync, data_dir=data_dir,
    )

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
    from uvicorn.logging import AccessFormatter

    log_level = os.environ.get("LOG_LEVEL", "INFO")
    stage = "logging"
    try:
        configure(log_level, mcp=True)
        stage = "application"
        _mcp, app = build_app()
        stamp_handlers(mcp=True)
        stage = "listener_settings"
        port = int(os.environ.get("PORT", "8000"))
        stage = "listener"
        access = logging.getLogger("uvicorn.access")
        if not access.handlers and access.propagate and not access.disabled:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(AccessFormatter(
                '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                use_colors=False,
            ))
            access.addHandler(handler)
            access.propagate = False
            if access.level == logging.NOTSET:
                access.setLevel(log_level.upper())
        stamp_handlers(mcp=True)
        uvicorn.run(
            app,
            host=os.environ.get("HOST", "0.0.0.0"),
            port=port,
            # Inherit the configured thresholds without replacing SDK sinks.
            log_level=None,
            log_config=None,
        )
    except Exception as exc:
        configure("ERROR", mcp=True)
        fields = {"reason": f"{stage}_startup_failed", "error_type": type(exc).__name__}
        if stage == "logging":
            fields.update(reason="invalid_log_level", setting="LOG_LEVEL")
        elif not os.environ.get("PODCAST_MCP_BEARER_TOKEN"):
            fields.update(reason="missing_bearer_token", setting="PODCAST_MCP_BEARER_TOKEN")
        elif stage == "listener_settings" and isinstance(exc, ValueError):
            fields.update(reason="invalid_listener_port", setting="PORT")
        logger.error("MCP startup failed; details withheld.", extra={"extras": fields})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
