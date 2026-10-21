"""curl-subprocess HTTP GET with validated, hop-by-hop redirect following.

Some hosts (Substack behind Cloudflare) fingerprint-block httpx's TLS stack
with a 403/429 while serving plain `curl` fine, so feeds.py and audio.py fall
back to a curl subprocess. That block can land on the FIRST hop — before any
redirect has been followed — so the URL handed to the fallback may itself be
an unresolved redirect (Substack 307s `api.substack.com` audio to a signed
`substackcdn.com` URL). curl is deliberately run WITHOUT `-L`: a bare `-L`
would follow redirects with no outbound-URL validation at all, escaping the
guard the httpx paths enforce per-hop via request hooks. Instead we follow
redirects here, one hop at a time, revalidating every Location with the same
guard.

Each hop's fetch doubles as its own redirect probe
(`-w '%{http_code}\\t%{redirect_url}'`): a 3xx writes only a tiny stub body,
which is discarded before following, and a 2xx has already downloaded the
payload — no separate probe request, so a direct 200 costs exactly one fetch.

If a host ever starts fingerprint-blocking plain curl too, the next step is
curl_cffi (curl-impersonate: browser TLS fingerprints, in-process redirect
control) — not more flags here.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .net import validate_outbound_url_async
from .proc import run_capture

logger = logging.getLogger(__name__)

MAX_HOPS = 5


class CurlFetchError(RuntimeError):
    """curl fetch failed: non-zero exit, malformed redirect, or too many hops."""


async def fetch_to_file(
    url: str,
    *,
    dest: Path,
    user_agent: str,
    max_time: str,
    timeout: float,
    label: str,
) -> str:
    """GET ``url`` into ``dest``, following redirects one validated hop at a
    time. Returns the final (post-redirect) URL whose body is in ``dest``.

    ``max_time`` bounds each curl invocation (curl's own --max-time) and
    ``timeout`` is the per-invocation kill backstop — both apply per hop.
    Raises :class:`CurlFetchError` on failure and
    :class:`~.net.UnsafeOutboundURL` on a redirect to a non-public target;
    ``dest`` may be left partially written — callers own cleanup.
    """
    hop_url = url
    for _ in range(MAX_HOPS):
        rc, out, err = await run_capture(
            ["curl", "-fsS", "--max-time", max_time, "-A", user_agent,
             "-o", str(dest), "-w", "%{http_code}\t%{redirect_url}",
             "--", hop_url],
            timeout=timeout, label=label,
        )
        if rc != 0:
            raise CurlFetchError(
                f"{label} failed (rc={rc}) for {hop_url}: "
                f"{err.decode(errors='replace')[:200]}"
            )
        status_s, _, redirect_url = (
            out.decode(errors="replace").strip().partition("\t")
        )
        try:
            status = int(status_s)
        except ValueError:
            raise CurlFetchError(
                f"{label}: unparsable curl status {status_s!r} for {hop_url}"
            ) from None
        if 300 <= status < 400:
            if not redirect_url:
                raise CurlFetchError(
                    f"{label}: HTTP {status} without a Location for {hop_url}"
                )
            # The hop body is a redirect stub — drop it before following, or a
            # later failure could leave it behind masquerading as the payload.
            dest.unlink(missing_ok=True)
            await validate_outbound_url_async(redirect_url)
            logger.info(
                "%s: following redirect %d %s -> %s",
                label, status, hop_url, redirect_url,
            )
            hop_url = redirect_url
            continue
        return hop_url
    raise CurlFetchError(f"{label}: more than {MAX_HOPS} redirects from {url}")
