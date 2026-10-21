"""Audio download: curl fallback on TLS-fingerprint 403/429 (Substack),
validated hop-by-hop redirect following, and pre-publish media verification."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from podcast_digest import audio, curlfetch
from podcast_digest.curlfetch import CurlFetchError
from podcast_digest.net import UnsafeOutboundURL

# Captured at import time, before conftest's autouse mock_ffprobe_ok patches
# the module attribute — tests that exercise the real probe restore this.
_REAL_FFPROBE = audio._ffprobe

GOOD_AUDIO = b"GOOD-AUDIO-PAYLOAD"
STUB_BODY = b"Temporary Redirect. Redirecting to https://cdn.example/real.mp3"


def _http_error(status: int, url: str) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", url)
    resp = httpx.Response(status, request=req)
    return httpx.HTTPStatusError(f"{status}", request=req, response=resp)


def _content_aware_ffprobe(monkeypatch):
    """ffprobe fake that judges by bytes: GOOD_AUDIO decodes, all else is
    rejected — mirroring real ffprobe on a redirect stub saved as .mp3."""
    async def fake(path):
        if GOOD_AUDIO in Path(path).read_bytes():
            return "ok", 3600.0, ""
        return "invalid", None, "Invalid data found when processing input"
    monkeypatch.setattr(audio, "_ffprobe", fake)


def _scripted_curl(monkeypatch, hops: list[tuple[int, str, bytes]]):
    """Replace curlfetch's run_capture with a scripted curl: each hop is
    (status, redirect_url, body-written-to--o). Returns the call log."""
    calls: list[str] = []

    async def fake_run_capture(cmd, *, timeout, label, env=None):
        status, redirect_url, body = hops[len(calls)]
        calls.append(cmd[-1])  # URL is the last argument (after `--`)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(body)
        return 0, f"{status}\t{redirect_url}".encode(), b""

    monkeypatch.setattr(curlfetch, "run_capture", fake_run_capture)
    return calls


async def test_download_falls_back_to_curl_on_403(monkeypatch, tmp_path):
    url = "https://api.substack.com/feed/podcast/1/abc.mp3"

    async def boom_stream(client, audio_url, dest_dir, basename):
        raise _http_error(403, audio_url)

    called = {}

    async def fake_curl(audio_url, *, dest_dir, basename):
        called["url"] = audio_url
        p = dest_dir / f"{basename}.mp3"
        p.write_bytes(b"audio")
        return p

    monkeypatch.setattr(audio, "_download_stream", boom_stream)
    monkeypatch.setattr(audio, "_download_curl", fake_curl)

    out = await audio.download(url, dest_dir=tmp_path, basename="ep")
    assert called["url"] == url
    assert out.exists() and out.read_bytes() == b"audio"


async def test_download_non_4xx_block_reraises(monkeypatch, tmp_path):
    async def boom_stream(client, audio_url, dest_dir, basename):
        raise _http_error(500, audio_url)

    async def fake_curl(audio_url, *, dest_dir, basename):  # pragma: no cover
        raise AssertionError("curl fallback must NOT run for non-403/429")

    monkeypatch.setattr(audio, "_download_stream", boom_stream)
    monkeypatch.setattr(audio, "_download_curl", fake_curl)

    with pytest.raises(httpx.HTTPStatusError):
        await audio.download("https://cdn.example/x.mp3", dest_dir=tmp_path, basename="ep")


async def test_curl_follows_validated_redirect(monkeypatch, tmp_path):
    """The Substack shape: httpx is blocked on hop 1, so curl receives an
    UNRESOLVED redirect — it must follow the 307 (discarding the stub body)
    and publish the payload fetched from the redirect target."""
    _content_aware_ffprobe(monkeypatch)
    calls = _scripted_curl(monkeypatch, [
        (307, "https://cdn.example/real.mp3", STUB_BODY),
        (200, "", GOOD_AUDIO),
    ])

    out = await audio._download_curl(
        "https://api.substack.com/feed/podcast/1/abc.mp3",
        dest_dir=tmp_path, basename="ep",
    )
    assert calls == [
        "https://api.substack.com/feed/podcast/1/abc.mp3",
        "https://cdn.example/real.mp3",
    ]
    assert out.read_bytes() == GOOD_AUDIO
    assert not list(tmp_path.glob("*.partial"))


async def test_curl_redirect_to_internal_host_rejected(monkeypatch, tmp_path):
    """A redirect Location is revalidated with the outbound-URL guard — the
    reason curl is run without `-L` at all."""
    _content_aware_ffprobe(monkeypatch)
    _scripted_curl(monkeypatch, [
        (307, "https://10.0.0.5/loot.mp3", STUB_BODY),
    ])

    with pytest.raises(UnsafeOutboundURL):
        await audio._download_curl(
            "https://api.substack.com/feed/podcast/1/abc.mp3",
            dest_dir=tmp_path, basename="ep",
        )
    assert not list(tmp_path.iterdir())  # no partial, no final


async def test_curl_refuses_to_publish_non_media(monkeypatch, tmp_path):
    """A 200 whose body is not decodable media (interstitial, error page) must
    fail the DOWNLOAD — nothing published, episode retryable."""
    _content_aware_ffprobe(monkeypatch)
    _scripted_curl(monkeypatch, [(200, "", STUB_BODY)])

    with pytest.raises(audio.InvalidMediaError):
        await audio._download_curl(
            "https://api.substack.com/feed/podcast/1/abc.mp3",
            dest_dir=tmp_path, basename="ep",
        )
    assert not list(tmp_path.iterdir())


async def test_poisoned_cached_file_is_refetched(monkeypatch, tmp_path):
    """A stub published as .mp3 by a pre-0.1.47 run must not be re-served from
    cache: it is deleted and the episode re-downloaded."""
    _content_aware_ffprobe(monkeypatch)
    poisoned = tmp_path / "ep.mp3"
    poisoned.write_bytes(STUB_BODY)
    _scripted_curl(monkeypatch, [(200, "", GOOD_AUDIO)])

    out = await audio._download_curl(
        "https://api.substack.com/feed/podcast/1/abc.mp3",
        dest_dir=tmp_path, basename="ep",
    )
    assert out == poisoned and out.read_bytes() == GOOD_AUDIO


async def test_curl_redirect_loop_capped(monkeypatch, tmp_path):
    _content_aware_ffprobe(monkeypatch)
    _scripted_curl(monkeypatch, [
        (307, f"https://cdn.example/hop{i}.mp3", STUB_BODY)
        for i in range(curlfetch.MAX_HOPS + 1)
    ])

    with pytest.raises(CurlFetchError, match="redirects"):
        await audio._download_curl(
            "https://cdn.example/hop.mp3", dest_dir=tmp_path, basename="ep",
        )


async def test_verify_media_unavailable_never_blocks(monkeypatch, tmp_path):
    """ffprobe missing/hung ⇒ validation degrades, publication proceeds —
    the same never-block contract probe_duration has always had."""
    async def unavailable(_path):
        return "unavailable", None, "no ffprobe"
    monkeypatch.setattr(audio, "_ffprobe", unavailable)
    _scripted_curl(monkeypatch, [(200, "", STUB_BODY)])

    out = await audio._download_curl(
        "https://cdn.example/x.mp3", dest_dir=tmp_path, basename="ep",
    )
    assert out.read_bytes() == STUB_BODY  # published unverified, by contract


async def test_stream_refuses_to_publish_non_media(monkeypatch, tmp_path):
    """The httpx path gets the same pre-publish verification: a 200 serving an
    interstitial with an audio content-type must not be published."""
    _content_aware_ffprobe(monkeypatch)
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200, content=STUB_BODY, headers={"content-type": "audio/mpeg"},
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(audio.InvalidMediaError):
            await audio.download(
                "https://cdn.example/x.mp3",
                dest_dir=tmp_path, basename="ep", client=client,
            )
    assert not list(tmp_path.iterdir())


def test_ext_from_url():
    assert audio._ext_from_url("https://h/x.mp3?t=1") == ".mp3"
    assert audio._ext_from_url("https://h/x.m4a") == ".m4a"
    assert audio._ext_from_url("https://h/no-ext") == ".mp3"


async def test_probe_duration_returns_none_when_ffprobe_missing(monkeypatch, tmp_path):
    """A missing/unspawnable ffprobe must never block an episode — return None."""
    monkeypatch.setattr(audio, "_ffprobe", _REAL_FFPROBE)  # undo conftest mock

    async def boom(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ffprobe")

    # ffprobe is spawned via the shared proc.run_capture helper now.
    monkeypatch.setattr("podcast_digest.proc.asyncio.create_subprocess_exec", boom)

    f = tmp_path / "ep.mp3"
    f.write_bytes(b"audio")
    assert await audio.probe_duration(f) is None
