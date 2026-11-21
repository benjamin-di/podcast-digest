from __future__ import annotations

import ipaddress
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpcore
import httpx
import pytest
import respx

from podcast_digest import curlfetch
from podcast_digest.net import (
    UnsafeOutboundURL,
    build_async_client,
    build_sync_client,
    validate_async_request,
)


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


async def test_async_client_blocks_dns_rebinding_at_connect_time():
    """Public IP during request hook followed by loopback at connect is rejected."""
    lookups = iter([
        (ipaddress.ip_address("93.184.216.34"),),
        (ipaddress.ip_address("127.0.0.1"),),
    ])
    connected: list[tuple[str, int]] = []

    class RecordingAsyncBackend(httpcore.AsyncNetworkBackend):
        async def connect_tcp(self, host, port, **kwargs):
            connected.append((host, port))
            raise AssertionError("connect_tcp must not be called for rebound loopback")

    async with build_async_client() as client:
        pool = client._transport._pool
        assert isinstance(pool, httpcore.AsyncConnectionPool)
        pool._network_backend._inner = RecordingAsyncBackend()
        with (
            patch(
                "podcast_digest.net._resolve_host_addresses",
                side_effect=lambda _host, _port: next(lookups),
            ),
            pytest.raises(UnsafeOutboundURL, match="127.0.0.1"),
        ):
            await client.get("http://rebind.example/feed")

    assert connected == []


def test_sync_client_blocks_dns_rebinding_at_connect_time():
    lookups = iter([
        (ipaddress.ip_address("93.184.216.34"),),
        (ipaddress.ip_address("127.0.0.1"),),
    ])
    connected: list[tuple[str, int]] = []

    class RecordingSyncBackend(httpcore.NetworkBackend):
        def connect_tcp(self, host, port, **kwargs):
            connected.append((host, port))
            raise AssertionError("connect_tcp must not be called for rebound loopback")

    with build_sync_client() as client:
        pool = client._transport._pool
        assert isinstance(pool, httpcore.ConnectionPool)
        pool._network_backend._inner = RecordingSyncBackend()
        with (
            patch(
                "podcast_digest.net._resolve_host_addresses",
                side_effect=lambda _host, _port: next(lookups),
            ),
            pytest.raises(UnsafeOutboundURL, match="127.0.0.1"),
        ):
            client.get("http://rebind.example/feed")

    assert connected == []


async def test_async_client_pins_validated_public_ip_for_tcp_connect():
    """The underlying socket receives the validated IP literal, never the hostname."""
    connected: list[tuple[str, int]] = []

    class RecordingAsyncBackend(httpcore.AsyncNetworkBackend):
        async def connect_tcp(self, host, port, **kwargs):
            connected.append((host, port))
            raise httpcore.ConnectError("stop after recording pinned ip")

    async with build_async_client() as client:
        pool = client._transport._pool
        assert isinstance(pool, httpcore.AsyncConnectionPool)
        pool._network_backend._inner = RecordingAsyncBackend()
        with (
            patch(
                "podcast_digest.net._resolve_host_addresses",
                return_value=(ipaddress.ip_address("93.184.216.34"),),
            ),
            pytest.raises(httpx.ConnectError),
        ):
            await client.get("https://public.example/feed")

    assert connected == [("93.184.216.34", 443)]


async def test_curlfetch_pins_resolved_ip_on_every_hop(tmp_path: Path):
    calls: list[list[str]] = []

    async def fake_run(cmd: list[str], *, timeout: float, label: str):
        calls.append(cmd)
        if len(calls) == 1:
            return 0, b"302\thttps://cdn.example/ep.mp3", b""
        (tmp_path / "out.bin").write_bytes(b"ok")
        return 0, b"200\t", b""

    def fake_resolve(host: str, _port: int):
        if host == "public.example":
            return (ipaddress.ip_address("93.184.216.34"),)
        return (ipaddress.ip_address("2001:4860:4860::8888"),)

    with (
        patch("podcast_digest.net._resolve_host_addresses", side_effect=fake_resolve),
        patch("podcast_digest.curlfetch.run_capture", new=AsyncMock(side_effect=fake_run)),
    ):
        final = await curlfetch.fetch_to_file(
            "https://public.example/feed",
            dest=tmp_path / "out.bin",
            user_agent="test-agent",
            max_time="30",
            timeout=45.0,
            label="curl test",
        )

    assert final == "https://cdn.example/ep.mp3"
    assert len(calls) == 2
    assert "--resolve" in calls[0]
    assert "public.example:443:93.184.216.34" in calls[0]
    assert "--resolve" in calls[1]
    assert "cdn.example:443:[2001:4860:4860::8888]" in calls[1]
