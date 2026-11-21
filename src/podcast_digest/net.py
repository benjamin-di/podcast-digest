"""Shared URL, timeout, and outbound-request safety helpers."""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpcore
import httpx

HTTP_SCHEMES = frozenset({"http", "https"})

# Conservative sustained-uplink estimate for scaling upload timeouts.
UPLOAD_BYTES_PER_SEC = 40_000
# Hard ceiling: a wedged socket must still bound the (serial) tick.
UPLOAD_TIMEOUT_MAX = 600.0

_INTERNAL_HOSTNAMES = frozenset({"localhost"})
_INTERNAL_SUFFIXES = (".local", ".internal", ".localdomain")


class UnsafeOutboundURL(ValueError):
    """The requested URL is not safe for this service to contact."""


@dataclass(frozen=True)
class ResolvedOutboundTarget:
    """Validated public host, port and concrete IP addresses to pin."""

    host: str
    port: int
    addresses: tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]
    is_ip_literal: bool = False


def is_http_url(url: str) -> bool:
    """True when ``url`` parses and its scheme is HTTP(S)."""
    if not url:
        return False
    try:
        return urlsplit(url).scheme.lower() in HTTP_SCHEMES
    except ValueError:
        return False


def upload_timeout(
    body_bytes: int, *, base: float, cap: float = UPLOAD_TIMEOUT_MAX,
) -> float:
    """Timeout for uploading ``body_bytes``: base + ~1s per 40KB, hard-capped."""
    return min(cap, base + body_bytes / UPLOAD_BYTES_PER_SEC)


def _resolve_host_addresses(host: str, port: int) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise UnsafeOutboundURL(f"could not resolve outbound host {host!r}: {exc}") from exc
    addresses = tuple(
        dict.fromkeys(ipaddress.ip_address(record[4][0]) for record in records)
    )
    if not addresses:
        raise UnsafeOutboundURL(f"outbound host resolved to no addresses: {host!r}")
    return addresses


def _validate_and_resolve_host(host: str, port: int) -> ResolvedOutboundTarget:
    """Validate ``host`` and return only public IP addresses for connection pinning."""
    normalized = host.lower().strip(".")
    if not normalized:
        raise UnsafeOutboundURL("outbound URL has no hostname")
    if normalized in _INTERNAL_HOSTNAMES or normalized.endswith(_INTERNAL_SUFFIXES):
        raise UnsafeOutboundURL(f"refusing to contact internal host: {host!r}")
    try:
        literal = ipaddress.ip_address(normalized)
    except ValueError:
        addresses = _resolve_host_addresses(normalized, port)
        is_literal = False
    else:
        addresses = (literal,)
        is_literal = True
    unsafe = [str(address) for address in addresses if not address.is_global]
    if unsafe:
        raise UnsafeOutboundURL(
            f"refusing to contact internal or non-public address for "
            f"{host!r}: {', '.join(unsafe)}"
        )
    return ResolvedOutboundTarget(
        host=normalized,
        port=port,
        addresses=addresses,
        is_ip_literal=is_literal,
    )


def resolve_outbound_url(value: str) -> ResolvedOutboundTarget:
    """Validate an outbound HTTP(S) URL and return its pinned public addresses."""
    parsed = urlsplit(value)
    if parsed.scheme not in HTTP_SCHEMES:
        raise UnsafeOutboundURL(f"unsupported outbound URL scheme: {parsed.scheme!r}")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeOutboundURL("outbound URLs may not contain credentials")
    host = parsed.hostname
    if not host:
        raise UnsafeOutboundURL("outbound URL has no hostname")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise UnsafeOutboundURL(f"invalid outbound URL port: {exc}") from exc
    return _validate_and_resolve_host(host, port)


async def resolve_outbound_url_async(value: str) -> ResolvedOutboundTarget:
    return await asyncio.to_thread(resolve_outbound_url, value)


def validate_outbound_url(value: str) -> None:
    """Require a credential-free HTTP(S) URL resolving only to public IPs."""
    resolve_outbound_url(value)


async def validate_outbound_url_async(value: str) -> None:
    await resolve_outbound_url_async(value)


def validate_request(request: httpx.Request) -> None:
    """httpx sync request hook; runs for the initial request and every redirect."""
    validate_outbound_url(str(request.url))


async def validate_async_request(request: httpx.Request) -> None:
    """httpx async request hook; runs for the initial request and every redirect."""
    await validate_outbound_url_async(str(request.url))


class _PinnedSyncNetworkBackend(httpcore.NetworkBackend):
    """Resolve and validate public IPs right at TCP connect, pinning the IP literal."""

    def __init__(self, inner: httpcore.NetworkBackend) -> None:
        self._inner = inner

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.NetworkStream:
        target = _validate_and_resolve_host(host, port)
        last_exc: Exception | None = None
        for address in target.addresses:
            try:
                return self._inner.connect_tcp(
                    str(address),
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.NetworkStream:
        raise UnsafeOutboundURL(f"refusing outbound unix socket connection: {path!r}")

    def sleep(self, seconds: float) -> None:
        self._inner.sleep(seconds)


class _PinnedAsyncNetworkBackend(httpcore.AsyncNetworkBackend):
    """Async counterpart: resolve, validate, and connect directly to the pinned IP."""

    def __init__(self, inner: httpcore.AsyncNetworkBackend) -> None:
        self._inner = inner

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        target = await asyncio.to_thread(_validate_and_resolve_host, host, port)
        last_exc: Exception | None = None
        for address in target.addresses:
            try:
                return await self._inner.connect_tcp(
                    str(address),
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise UnsafeOutboundURL(f"refusing outbound unix socket connection: {path!r}")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def pinned_sync_transport() -> httpx.HTTPTransport:
    """Return an httpx sync transport that pins validated public IPs at connect time."""
    transport = httpx.HTTPTransport()
    pool = transport._pool
    if isinstance(pool, httpcore.ConnectionPool):
        pool._network_backend = _PinnedSyncNetworkBackend(pool._network_backend)
    return transport


def pinned_async_transport() -> httpx.AsyncHTTPTransport:
    """Return an httpx async transport that pins validated public IPs at connect time."""
    transport = httpx.AsyncHTTPTransport()
    pool = transport._pool
    if isinstance(pool, httpcore.AsyncConnectionPool):
        pool._network_backend = _PinnedAsyncNetworkBackend(pool._network_backend)
    return transport


def build_sync_client(
    *,
    timeout: httpx.Timeout | float | None = None,
    headers: dict[str, str] | None = None,
    follow_redirects: bool = True,
) -> httpx.Client:
    """Build an httpx sync client with request-hook and connect-time IP pinning."""
    kwargs: dict[str, Any] = {
        "follow_redirects": follow_redirects,
        "event_hooks": {"request": [validate_request]},
        "transport": pinned_sync_transport(),
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if headers is not None:
        kwargs["headers"] = headers
    return httpx.Client(**kwargs)


def build_async_client(
    *,
    timeout: httpx.Timeout | float | None = None,
    headers: dict[str, str] | None = None,
    follow_redirects: bool = True,
) -> httpx.AsyncClient:
    """Build an httpx async client with request-hook and connect-time IP pinning."""
    kwargs: dict[str, Any] = {
        "follow_redirects": follow_redirects,
        "event_hooks": {"request": [validate_async_request]},
        "transport": pinned_async_transport(),
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if headers is not None:
        kwargs["headers"] = headers
    return httpx.AsyncClient(**kwargs)
