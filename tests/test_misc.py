"""Spot-checks on small pure helpers across the codebase."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from podcast_digest.config import Feed
from podcast_digest.mail import _recipients
from podcast_digest.pipeline import (
    _artifact_key,
    _atomic_write_text,
    _build_subject,
    _feed_storage_key,
    _safe_basename,
    _wrap_email,
)
from podcast_digest.sanitize import safe_listen_url, sanitize_summary_html
from podcast_digest.summarize import (
    SummarizationInput,
    _build_user_message,
    _fmt_hms,
    _format_transcript,
    _language_guidance,
    _strip_code_fence,
)
from podcast_digest.transcribe import Segment


def test_feed_identity_is_filesystem_safe():
    assert Feed(name="The Morning Market Desk", search="x").identity() == "the-morning-market-desk"
    assert Feed(name="Market Hours!!", search="x").identity() == "market-hours"
    # Pathological all-special-chars → fallback
    assert Feed(name="!!!", search="x").identity() == "feed"


def test_feed_audio_url_is_a_valid_source():
    f = Feed(name="A clip", audio_url="https://cdn.example/x.mp3")
    assert f.audio_url == "https://cdn.example/x.mp3"


def test_feed_with_no_source_is_rejected():
    # The model enforces exactly one source; audio_url counts as one.
    with pytest.raises(ValidationError):
        Feed(name="no source")


def test_feed_with_multiple_sources_is_rejected():
    with pytest.raises(ValidationError, match="exactly one"):
        Feed(name="ambiguous", rss_url="https://x/rss", search="x")


def test_feed_normalizes_blank_sources_before_cardinality_check():
    feed = Feed(name="  Search feed  ", rss_url="  ", search="  query  ")
    assert feed.name == "Search feed"
    assert feed.rss_url is None
    assert feed.search == "query"


def test_feed_rejects_negative_thresholds_and_misplaced_metadata():
    with pytest.raises(ValidationError):
        Feed(name="negative", search="x", min_minutes=-1)
    with pytest.raises(ValidationError, match="only valid with audio_url"):
        Feed(name="metadata", search="x", duration_seconds=60)


def test_feed_source_language_normalization():
    # Unset / English (any casing, stray whitespace) → None: the English path.
    assert Feed(name="x", search="x").source_language is None
    assert Feed(name="x", search="x", language="").source_language is None
    assert Feed(name="x", search="x", language="en").source_language is None
    assert Feed(name="x", search="x", language=" EN ").source_language is None
    assert Feed(name="x", search="x", language="en-US").source_language is None
    # Anything else → lowercased, region-stripped whisper `-l` code.
    assert Feed(name="x", search="x", language="zh").source_language == "zh"
    assert Feed(name="x", search="x", language=" ZH ").source_language == "zh"
    assert Feed(name="x", search="x", language="ja").source_language == "ja"
    assert Feed(name="x", search="x", language="zh-CN").source_language == "zh"
    assert Feed(name="x", search="x", language="pt_BR").source_language == "pt"


def test_language_guidance_zh_has_pinyin_and_numeric_unit_rules():
    g = _language_guidance("zh")
    assert "English" in g
    assert "pinyin" in g
    assert "万 = 10^4" in g and "亿 = 10^8" in g
    # Non-Chinese languages get the generic English-output instruction.
    g_ja = _language_guidance("ja")
    assert "ja" in g_ja and "English" in g_ja
    assert "pinyin" not in g_ja


def test_build_user_message_appends_guidance_only_for_non_english():
    kw = dict(
        podcast_name="Show", title="Ep", published_at="2026-01-01",
        duration_seconds=None, description=None,
        transcript_text="hello", segments=[],
    )
    assert "Mandarin" not in _build_user_message(SummarizationInput(**kw))
    zh = _build_user_message(SummarizationInput(**kw, source_language="zh"))
    assert "Mandarin Chinese" in zh and "pinyin" in zh


def test_safe_basename_drops_unsafe_chars_and_caps_length():
    base = _safe_basename("https://podcast.example.com/some/long/url/with/many/segments-and-a-very-long-tail-that-goes-on-and-on-and-on-and-on")
    assert "/" not in base
    assert ":" not in base
    assert len(base) <= 100
    # Empty / all-symbol input gives a fallback
    assert _safe_basename("???") == "episode"


def test_artifact_key_is_ascii_and_collision_resistant():
    first = _artifact_key("épisode/42")
    second = _artifact_key("épisode?42")
    assert first != second
    assert first.isascii()
    assert len(first) <= 200
    assert all(c.isalnum() or c in "._-" for c in first)


def test_feed_storage_key_distinguishes_same_name_sources():
    feed = Feed(name="同名 Podcast", search="x")
    first = _feed_storage_key(feed, "https://one.example/rss")
    second = _feed_storage_key(feed, "https://two.example/rss")
    assert first != second
    assert first.isascii() and second.isascii()


def test_atomic_write_preserves_final_file_on_replace_failure(
    tmp_path: Path, monkeypatch,
):
    target = tmp_path / "episode.clean.md"
    target.write_text("known good", encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("podcast_digest.pipeline.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        _atomic_write_text(target, "partial replacement")

    assert target.read_text(encoding="utf-8") == "known good"
    assert list(tmp_path.glob("*.tmp")) == []


def test_build_subject_includes_duration():
    s = _build_subject("Show", "Title", 5400)
    assert s == "[Podcast] Show — Title (1h30m)"
    s2 = _build_subject("Show", "Title", 2700)
    assert s2 == "[Podcast] Show — Title (45m)"
    s3 = _build_subject("Show", "Title", None)
    assert s3 == "[Podcast] Show — Title"


def test_wrap_email_includes_episode_metadata():
    html = _wrap_email(
        "<p>SUMMARY</p>", "The Show", "Ep Title", "https://cdn.example/a.mp3",
        published_at="2025-06-10T14:30:00+00:00", duration_seconds=5400,
    )
    assert "Ep Title" in html                 # title header
    assert "The Show" in html                  # podcast name
    assert "Released 10 Jun 2025" in html      # release date metadata
    assert "1h30m" in html                     # runtime metadata
    assert "<p>SUMMARY</p>" in html            # model output preserved
    assert "cdn.example/a.mp3" in html         # listen link
    # Missing metadata degrades gracefully (no crash, no stray separators)
    html2 = _wrap_email("<p>S</p>", "Show", "T", "https://x/a.mp3")
    assert "Released" not in html2 and "·" not in html2
    # English episodes (no original_language) carry no translation note.
    assert "Translated from" not in html


def test_wrap_email_notes_original_language():
    html = _wrap_email(
        "<p>S</p>", "示例播客", "某一集", "https://x/a.mp3",
        original_language="zh",
    )
    assert "Translated from Mandarin" in html
    # Unknown code falls back to the raw code rather than being dropped.
    html2 = _wrap_email(
        "<p>S</p>", "Show", "T", "https://x/a.mp3", original_language="sw",
    )
    assert "Translated from sw" in html2


def test_sanitize_summary_html_strips_dangerous_markup():
    """F8: the model output is untrusted (prompt-injectable). Disallowed tags,
    event handlers, and non-http(s) hrefs must be stripped server-side."""
    dirty = (
        "<h2>TL;DR</h2><p onclick=\"steal()\">hi</p>"
        "<script>evil()</script>"
        "<a href=\"javascript:phish()\">Re-authenticate</a>"
        "<a href=\"https://ok.example\">real link</a>"
        "<img src=x onerror=alert(1)>"
        "<iframe src=\"https://evil\"></iframe>"
    )
    clean = sanitize_summary_html(dirty)
    assert "<script" not in clean
    assert "onclick" not in clean and "onerror" not in clean
    assert "javascript:" not in clean
    assert "<iframe" not in clean and "<img" not in clean
    # Allowed structure + a safe https link survive.
    assert "<h2>TL;DR</h2>" in clean
    assert 'href="https://ok.example"' in clean


def test_wrap_email_sanitizes_and_validates_listen_url():
    """F8: injected markup in the summary is neutralized, and a non-http(s)
    enclosure URL yields no "Listen" link at all."""
    html = _wrap_email(
        "<p>ok</p><script>x</script>", "Show", "T",
        "javascript:alert(document.cookie)",
    )
    assert "<script" not in html
    # A javascript: listen URL is dropped entirely — no anchor emitted.
    assert "javascript:" not in html
    assert "Listen to the episode" not in html
    # An http(s) URL still renders the link.
    html2 = _wrap_email("<p>ok</p>", "Show", "T", "https://cdn/a.mp3")
    assert 'href="https://cdn/a.mp3"' in html2 and "Listen to the episode" in html2


def test_safe_listen_url():
    assert safe_listen_url("https://x/a.mp3") == "https://x/a.mp3"
    assert safe_listen_url("http://x/a.mp3") == "http://x/a.mp3"
    assert safe_listen_url("javascript:alert(1)") is None
    assert safe_listen_url("data:text/html,evil") is None
    assert safe_listen_url("ftp://x/a.mp3") is None


def test_recipients_handles_strings_and_lists():
    assert _recipients("a@example.com") == [{"emailAddress": {"address": "a@example.com"}}]
    assert _recipients("a@example.com, b@example.com") == [
        {"emailAddress": {"address": "a@example.com"}},
        {"emailAddress": {"address": "b@example.com"}},
    ]
    assert _recipients(["a@example.com", "b@example.com"]) == [
        {"emailAddress": {"address": "a@example.com"}},
        {"emailAddress": {"address": "b@example.com"}},
    ]
    assert _recipients(",,,") == []


def test_fmt_hms():
    assert _fmt_hms(0) == "00:00:00"
    assert _fmt_hms(61) == "00:01:01"
    assert _fmt_hms(3725) == "01:02:05"


def test_format_transcript_inserts_periodic_timestamps():
    segs = [
        Segment(start=0.0, end=2.0, text="Hello."),
        Segment(start=2.0, end=4.0, text="World."),
        # Big gap — should get a new timestamp marker
        Segment(start=120.0, end=121.0, text="Later."),
    ]
    out = _format_transcript(segs, fallback="")
    assert "[00:00:00]" in out
    assert "[00:02:00]" in out
    assert "Hello." in out
    assert "Later." in out


def test_strip_code_fence_handles_html_fence():
    s = "```html\n<h2>Hi</h2>\n```"
    assert _strip_code_fence(s) == "<h2>Hi</h2>"
    # No fence — passthrough
    assert _strip_code_fence("<p>hi</p>") == "<p>hi</p>"


def test_upload_timeout_scales_and_caps():
    from podcast_digest.net import UPLOAD_TIMEOUT_MAX, upload_timeout

    assert upload_timeout(0, base=30.0) == 30.0
    assert upload_timeout(400_000, base=30.0) == 40.0     # +1s per 40KB
    # A multi-MB body hits the hard cap — a wedged socket must still bound
    # the serial tick (this is what the wiki path was missing).
    assert upload_timeout(100_000_000, base=30.0) == UPLOAD_TIMEOUT_MAX


def test_whisper_timeout_scales_with_audio_duration():
    from podcast_digest.transcribe import (
        _WAV_BYTES_PER_SECOND,
        _WHISPER_SECONDS_PER_AUDIO_SECOND,
        _WHISPER_TIMEOUT_FLOOR,
        _whisper_timeout,
    )

    # Short clip: floor dominates (model load time etc.).
    assert _whisper_timeout(_WAV_BYTES_PER_SECOND * 600) == _WHISPER_TIMEOUT_FLOOR
    # A 3h episode gets 4x realtime — room for the CPU fallback path, where the
    # old fixed 2h ceiling would have SIGKILLed a slow-but-succeeding run.
    three_hours = 3 * 3600
    assert _whisper_timeout(_WAV_BYTES_PER_SECOND * three_hours) == (
        three_hours * _WHISPER_SECONDS_PER_AUDIO_SECOND
    )
