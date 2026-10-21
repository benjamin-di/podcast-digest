"""Shared URL, timeout, and outbound-request safety helpers."""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

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


def validate_outbound_url(value: str) -> None:
    """Require a credential-free HTTP(S) URL resolving only to public IPs."""
    parsed = urlsplit(value)
    if parsed.scheme not in HTTP_SCHEMES:
        raise UnsafeOutboundURL(f"unsupported outbound URL scheme: {parsed.scheme!r}")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeOutboundURL("outbound URLs may not contain credentials")
    host = parsed.hostname
    if not host:
        raise UnsafeOutboundURL("outbound URL has no hostname")
    normalized = host.lower().strip(".")
    if normalized in _INTERNAL_HOSTNAMES or normalized.endswith(_INTERNAL_SUFFIXES):
        raise UnsafeOutboundURL(f"refusing to contact internal host: {host!r}")
    try:
        literal = ipaddress.ip_address(normalized)
    except ValueError:
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise UnsafeOutboundURL(f"invalid outbound URL port: {exc}") from exc
        addresses = _resolve_host_addresses(normalized, port)
    else:
        addresses = (literal,)
    unsafe = [str(address) for address in addresses if not address.is_global]
    if unsafe:
        raise UnsafeOutboundURL(
            f"refusing to contact internal or non-public address for "
            f"{host!r}: {', '.join(unsafe)}"
        )


async def validate_outbound_url_async(value: str) -> None:
    await asyncio.to_thread(validate_outbound_url, value)


def validate_request(request: httpx.Request) -> None:
    """httpx sync request hook; runs for the initial request and every redirect."""
    validate_outbound_url(str(request.url))


async def validate_async_request(request: httpx.Request) -> None:
    """httpx async request hook; runs for the initial request and every redirect."""
    await validate_outbound_url_async(str(request.url))
