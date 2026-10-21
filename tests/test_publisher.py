"""Publisher transcripts: tag discovery, format choice, parsers, plausibility,
fetching with fallback between candidates, and the on-disk artifacts."""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from podcast_digest import publisher
from podcast_digest.feeds import parse_episodes
from podcast_digest.publisher import TranscriptLink
from podcast_digest.transcribe import Transcript, load_whisper_json

NS = 'xmlns:podcast="https://podcastindex.org/namespace/1.0"'


def _feed(items: str, ns: str = NS) -> bytes:
    return f"""<?xml version="1.0"?>
<rss version="2.0" {ns} xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
<channel><title>T</title>{items}</channel></rss>""".encode()


def _item(guid: str, tags: str = "", duration: str = "10:00") -> str:
    return f"""<item><title>{guid}</title><guid>{guid}</guid>
<pubDate>Tue, 02 Apr 2024 10:00:00 +0000</pubDate>
<enclosure url="https://cdn.example.com/{guid}.mp3" type="audio/mpeg" length="1"/>
<itunes:duration>{duration}</itunes:duration>{tags}</item>"""


def _vtt(minutes: int = 10, words_per_cue: int = 10, speaker: str | None = "Ana") -> str:
    """A timed transcript covering `minutes`, one cue every 6 s."""
    lines = ["WEBVTT", "", "NOTE generated for tests", ""]
    for i in range(minutes * 10):
        start, end = i * 6, i * 6 + 5
        voice = f"<v {speaker}>" if speaker else ""
        body = " ".join(f"word{i}x{j}" for j in range(words_per_cue))
        lines += [f"{start // 60:02d}:{start % 60:02d}.000 --> {end // 60:02d}:{end % 60:02d}.500",
                  f"{voice}{body}", ""]
    return "\n".join(lines)


# ---- discovery --------------------------------------------------------------


def test_every_transcript_tag_of_every_item_is_found():
    body = _feed(
        _item("g1", '<podcast:transcript url="https://t.example.com/1.vtt" type="text/vtt" '
                    'language="en" rel="captions"/>'
                    '<podcast:transcript url="https://t.example.com/1.json" type="application/json"/>')
        + _item("g2")
        + _item("g3", '<podcast:transcript url="https://t.example.com/3.html" type="text/html"/>')
    )
    links = publisher.transcript_links(body)
    assert [link.url for link in links["g1"]] == [
        "https://t.example.com/1.vtt", "https://t.example.com/1.json",
    ]
    assert links["g1"][0] == TranscriptLink(
        url="https://t.example.com/1.vtt", type="text/vtt", language="en", rel="captions",
    )
    assert "g2" not in links
    assert links["#2"] == links["g3"]


def test_namespace_variants_and_unparseable_xml():
    http_ns = 'xmlns:podcast="http://podcastindex.org/namespace/1.0/"'
    body = _feed(_item("g", '<podcast:transcript url="https://t.example.com/t.srt"/>'), http_ns)
    assert publisher.transcript_links(body)["g"][0].url == "https://t.example.com/t.srt"
    # A tag in another namespace is not a Podcasting 2.0 transcript.
    other = _feed(_item("g", '<x:transcript xmlns:x="https://other.example" url="https://t.example.com/x"/>'))
    assert publisher.transcript_links(other) == {}
    assert publisher.transcript_links(b"<rss><channel><item>broken") == {}


def test_parse_episodes_attaches_links():
    body = _feed(_item("g1", '<podcast:transcript url="https://t.example.com/1.vtt" type="text/vtt"/>'))
    (ep,) = parse_episodes(body)
    assert ep.transcripts == (TranscriptLink(url="https://t.example.com/1.vtt", type="text/vtt"),)


# ---- choice ------------------------------------------------------------------


def test_candidates_prefer_the_feed_language_then_timed_formats():
    links = (
        TranscriptLink("https://t.example.com/a.html", "text/html", "en"),
        TranscriptLink("https://t.example.com/a.vtt", "text/vtt", "en"),
        TranscriptLink("https://t.example.com/a.json", "application/json", "de"),
        TranscriptLink("https://t.example.com/a.srt", "application/x-subrip", None),
        TranscriptLink("https://t.example.com/a.pdf", "application/pdf", "en"),
    )
    order = [link.url.rsplit(".", 1)[1] for link, _ in publisher.candidates(links, language=None)]
    assert order == ["vtt", "html", "srt", "json"]   # pdf dropped; English first
    order_de = [fmt for _, fmt in publisher.candidates(links, language="de")]
    assert order_de[0] == "json"


def test_format_from_the_extension_when_the_type_is_missing():
    assert publisher.candidates(
        (TranscriptLink("https://t.example.com/x.srt?sig=1", ""),), language=None,
    )[0][1] == "srt"


# ---- parsers -----------------------------------------------------------------


def test_vtt_with_voices_tags_and_hour_timestamps():
    body = """WEBVTT

STYLE
::cue { color: lime }

1
00:00:01.000 --> 00:00:03.000 align:start
<v.loud Ana Lima>Hello <i>and</i> welcome &amp; hi.</v>

2
01:00:05.250 --> 01:00:07.000
<v Bo>Thanks</v>
"""
    t = publisher.parse(body, "vtt")
    assert [(s.start, s.speaker) for s in t.segments] == [(1.0, "Ana Lima"), (3605.25, "Bo")]
    assert t.segments[0].text == "Ana Lima: Hello and welcome & hi."
    assert t.segments[1].text == "Bo: Thanks"


def test_srt_and_repeated_captions():
    body = "1\n00:00:01,000 --> 00:00:02,000\nSame line\n\n2\n00:00:02,000 --> 00:00:03,000\nSame line\n\n3\n00:00:03,500 --> 00:00:04,000\nNext\n"
    t = publisher.parse(body, "srt")
    assert t.text == "Same line Next"   # rolling repeat dropped, pieces merged
    assert t.segments[0].start == 1.0


def test_json_words_merge_into_speaker_turns():
    body = json.dumps({"version": "1.0.0", "segments": [
        {"speaker": "Ana", "startTime": 0.0, "endTime": 0.4, "body": "Hello"},
        {"speaker": "Ana", "startTime": 0.4, "endTime": 0.9, "body": "there."},
        {"speaker": "Bo", "startTime": 1.0, "endTime": 1.5, "body": "Hi"},
        {"startTime": "bad", "body": "skipped"},
    ]})
    t = publisher.parse(body, "json")
    assert [s.text for s in t.segments] == ["Ana: Hello there.", "Bo: Hi"]


def test_html_and_plain_text():
    html_body = "<html><body><cite>Ana:</cite><time>0:00</time><p>Hello &amp; welcome</p><script>x()</script></body></html>"
    t = publisher.parse(html_body, "html")
    assert t.segments == [] and t.text == "Ana:\n0:00\nHello & welcome"
    assert publisher.parse("  just text \n", "text").text == "just text"
    with pytest.raises(ValueError):
        publisher.parse("WEBVTT\n\n", "vtt")


# ---- plausibility ------------------------------------------------------------


def test_implausible_transcripts_are_rejected():
    full = publisher.parse(_vtt(minutes=10), "vtt")
    assert publisher.implausible(full, declared_seconds=600) is None
    # Words for 10 minutes, but the episode is an hour long.
    assert "words" in publisher.implausible(full, declared_seconds=3600)
    # Enough words but timing ends early: a partial transcript.
    dense = publisher.parse(_vtt(minutes=4, words_per_cue=40), "vtt")
    assert "covers" in publisher.implausible(dense, declared_seconds=600)
    assert "words" in publisher.implausible(Transcript(text="too short", segments=[]),
                                            declared_seconds=None)


# ---- fetch -------------------------------------------------------------------


@pytest.fixture
def client():
    return httpx.AsyncClient()


async def test_fetch_falls_through_to_the_next_candidate(client):
    links = (
        TranscriptLink("https://t.example.com/a.json", "application/json"),
        TranscriptLink("https://t.example.com/a.vtt", "text/vtt"),
    )
    with respx.mock(assert_all_called=True) as router:
        router.get("https://t.example.com/a.json").mock(return_value=httpx.Response(404))
        router.get("https://t.example.com/a.vtt").mock(
            return_value=httpx.Response(200, text=_vtt(minutes=10)),
        )
        pub = await publisher.fetch(links, client=client, language=None, declared_seconds=600)
    assert pub is not None and pub.link.url.endswith(".vtt")
    assert pub.end_seconds == pytest.approx(599.5)


async def test_fetch_returns_none_when_nothing_is_usable(client):
    links = (TranscriptLink("https://t.example.com/a.vtt", "text/vtt"),)
    with respx.mock() as router:
        router.get("https://t.example.com/a.vtt").mock(
            return_value=httpx.Response(200, text=_vtt(minutes=2)),
        )
        assert await publisher.fetch(
            links, client=client, language=None, declared_seconds=3600,
        ) is None


async def test_fetch_refuses_oversized_bodies(client, monkeypatch):
    monkeypatch.setattr(publisher, "MAX_TRANSCRIPT_BYTES", 1000)
    links = (TranscriptLink("https://t.example.com/a.vtt", "text/vtt"),)
    with respx.mock() as router:
        router.get("https://t.example.com/a.vtt").mock(
            return_value=httpx.Response(200, text=_vtt(minutes=10)),
        )
        assert await publisher.fetch(
            links, client=client, language=None, declared_seconds=600,
        ) is None


# ---- artifacts ---------------------------------------------------------------


def test_artifacts_round_trip_through_the_whisper_loader(tmp_path):
    pub = publisher.PublisherTranscript(
        transcript=publisher.parse(_vtt(minutes=1), "vtt"),
        link=TranscriptLink("https://t.example.com/a.vtt", "text/vtt"),
    )
    json_path = tmp_path / "g.json"
    txt_path = tmp_path / "g.txt"
    json_path.write_text(publisher.to_whisper_json(pub), encoding="utf-8")
    txt_path.write_text(pub.transcript.text, encoding="utf-8")
    loaded = load_whisper_json(json_path, fallback_text_path=txt_path)
    assert loaded.text == pub.transcript.text
    assert loaded.segments[0].start == 0.0
    assert publisher.source_of(json_path) == "publisher"
    assert publisher.source_of(tmp_path / "missing.json") == "whisper"

    md = publisher.to_markdown(pub, title="Ep", show="Show")
    assert md.startswith("# Ep\n\n_Show_ — transcript from the publisher")
    assert "[00:00:00] **Ana:** word0x0" in md


def test_untimed_artifacts_fall_back_to_the_text(tmp_path):
    pub = publisher.PublisherTranscript(
        transcript=publisher.parse("Ana: hello there", "text"),
        link=TranscriptLink("https://t.example.com/a.txt", "text/plain"),
    )
    (tmp_path / "g.json").write_text(publisher.to_whisper_json(pub), encoding="utf-8")
    (tmp_path / "g.txt").write_text(pub.transcript.text, encoding="utf-8")
    loaded = load_whisper_json(tmp_path / "g.json", fallback_text_path=tmp_path / "g.txt")
    assert loaded.text == "Ana: hello there" and loaded.segments == []
