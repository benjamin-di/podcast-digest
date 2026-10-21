from __future__ import annotations

import httpx
import pytest
import respx

from podcast_digest.net import UnsafeOutboundURL, validate_async_request


@respx.mock
async def test_async_request_hook_blocks_redirect_before_internal_request():
    public = respx.get("https://public.example/feed").respond(
        302, headers={"location": "http://127.0.0.1:8080/private"},
    )
    internal = respx.get("http://127.0.0.1:8080/private").respond(
        200, content=b"secret",
    )
    async with httpx.AsyncClient(
        follow_redirects=True,
        event_hooks={"request": [validate_async_request]},
    ) as client:
        with pytest.raises(UnsafeOutboundURL, match="internal"):
            await client.get("https://public.example/feed")

    assert public.called
    assert not internal.called
