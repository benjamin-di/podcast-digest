"""Pipeline tests for publisher transcripts (Podcasting 2.0).

The sample feed's newest episode (90 min) gets a <podcast:transcript> link.
When the transcript is usable, the episode is summarized from it without a
download or local transcription; when it is not, whisper runs as before.
"""
from __future__ import annotations

from pathlib import Path

import httpx

from podcast_digest.pipeline import run_tick

from .conftest import EPISODE_URLS, FEED_URL, RSS_BYTES, FakeMailer

NEWEST_GUID = "ep-3-guid"
TRANSCRIPT_URL = "https://cdn.example.com/ep3.vtt"


def _feed_with_transcript(tag_attrs: str = f'url="{TRANSCRIPT_URL}" type="text/vtt"') -> bytes:
    body = RSS_BYTES.decode()
    body = body.replace(
        'xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"',
        'xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" '
        'xmlns:podcast="https://podcastindex.org/namespace/1.0"',
    )
    body = body.replace(
        "<itunes:duration>01:30:00</itunes:duration>",
        f"<itunes:duration>01:30:00</itunes:duration>\n      <podcast:transcript {tag_attrs}/>",
    )
    return body.encode()


def _vtt(minutes: int = 90) -> str:
    lines = ["WEBVTT", ""]
    for i in range(minutes * 10):
        s, e = i * 6, i * 6 + 5
        stamp = lambda t: f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}.000"  # noqa: E731
        speaker = "Ana" if (i // 5) % 2 == 0 else "Bo"
        lines += [f"{stamp(s)} --> {stamp(e)}",
                  f"<v {speaker}>publisher words {i} " + "lorem " * 8, ""]
    return "\n".join(lines)


def _serve(mock_http, *, transcript: httpx.Response) -> None:
    mock_http.get(FEED_URL).mock(return_value=httpx.Response(
        200, content=_feed_with_transcript(), headers={"content-type": "application/xml"},
    ))
    mock_http.get(TRANSCRIPT_URL).mock(return_value=transcript)


async def test_publisher_transcript_replaces_download_and_whisper(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    _serve(mock_http, transcript=httpx.Response(200, text=_vtt()))
    mailer = FakeMailer()

    stats = await run_tick(settings=settings, feeds_file=feeds_file,
                           db=db, mailer=mailer, dry_run=False)

    assert stats["emailed"] == 1
    assert mock_transcribe.calls == []                         # no whisper
    audio_calls = [c for c in mock_http.calls
                   if str(c.request.url) == EPISODE_URLS[NEWEST_GUID]]
    assert audio_calls == []                                   # no download
    # The summary was written from the publisher's text, with speakers.
    assert "Ana: publisher words 0" in mock_llm.calls[0]["transcript_text"]
    assert "Summarized from the publisher's transcript" in mailer.sent[0]["body_html"]

    ep = db.get_episode(FEED_URL, NEWEST_GUID)
    assert ep is not None and ep.status == "emailed"
    assert ep.audio_path is None
    assert Path(ep.transcript_path).read_text().startswith("Ana: publisher words 0")


async def test_publisher_transcript_is_the_cleaned_attachment_without_an_llm_pass(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http, monkeypatch,
):
    settings.PODCAST_DIGEST_ATTACH_TRANSCRIPT = "cleaned"
    cleaned: list[str] = []
    monkeypatch.setattr(
        "podcast_digest.pipeline.clean_transcript",
        lambda inp, **_kw: cleaned.append(inp.title) or "# never used",
    )
    _serve(mock_http, transcript=httpx.Response(200, text=_vtt()))
    mailer = FakeMailer()

    await run_tick(settings=settings, feeds_file=feeds_file,
                   db=db, mailer=mailer, dry_run=False)

    assert cleaned == []
    (att,) = mailer.sent[0]["attachments"]
    assert att.name.endswith(".md")
    assert b"transcript from the publisher" in att.content
    assert b"**Ana:** publisher words 0" in att.content


async def test_unusable_publisher_transcript_falls_back_to_whisper_in_the_same_tick(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    _serve(mock_http, transcript=httpx.Response(404))
    mailer = FakeMailer()

    stats = await run_tick(settings=settings, feeds_file=feeds_file,
                           db=db, mailer=mailer, dry_run=False)

    assert stats["emailed"] == 1
    assert len(mock_transcribe.calls) == 1
    assert "publisher" not in mailer.sent[0]["body_html"]


async def test_too_short_publisher_transcript_falls_back_to_whisper(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    _serve(mock_http, transcript=httpx.Response(200, text=_vtt(minutes=5)))
    await run_tick(settings=settings, feeds_file=feeds_file,
                   db=db, mailer=FakeMailer(), dry_run=False)
    assert len(mock_transcribe.calls) == 1


async def test_publisher_transcripts_can_be_switched_off(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    settings.PODCAST_DIGEST_PUBLISHER_TRANSCRIPTS = False
    _serve(mock_http, transcript=httpx.Response(200, text=_vtt()))
    await run_tick(settings=settings, feeds_file=feeds_file,
                   db=db, mailer=FakeMailer(), dry_run=False)
    assert len(mock_transcribe.calls) == 1
    assert not any(str(c.request.url) == TRANSCRIPT_URL for c in mock_http.calls)


async def test_feed_level_opt_out(
    settings, db, feeds_file, mock_llm, mock_transcribe, mock_http,
):
    feeds_file.feeds[0].publisher_transcripts = False
    _serve(mock_http, transcript=httpx.Response(200, text=_vtt()))
    await run_tick(settings=settings, feeds_file=feeds_file,
                   db=db, mailer=FakeMailer(), dry_run=False)
    assert len(mock_transcribe.calls) == 1
