"""Tests for MCP source-language detection (podcast_digest_mcp.lang)."""
from __future__ import annotations

import pytest

pytest.importorskip("tomlkit")  # the `mcp` extra gates the whole package

from podcast_digest_mcp import lang

# -- normalize ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("en", None),
        (" EN ", None),
        ("en-US", None),
        ("zh", "zh"),
        (" ZH ", "zh"),
        ("zh-CN", "zh"),          # region stripped to bare subtag
        ("zh_Hans_CN", "zh"),
        ("pt-BR", "pt"),
        ("ja", "ja"),
        ("gibberish", None),       # not a 2–3 letter code
        ("x", None),
    ],
)
def test_normalize(raw, expected):
    assert lang.normalize(raw) == expected


# -- script detection via resolve_language (declared=None) -------------------


def _sniff(*texts: str) -> str | None:
    return lang.resolve_language(None, None, list(texts))


def test_detect_mandarin_from_han_titles():
    # A Mandarin show: Han-only titles, no declared language.
    assert _sniff("口述创业史：用人观、科技与未来") == "zh"


def test_detect_mandarin_survives_mixed_latin_metadata():
    # A Mandarin channel with a romanized name + some English-translated
    # titles alongside the Chinese ones. The Latin metadata must not dilute the
    # Han signal away — the audio is Mandarin.
    assert _sniff(
        "Example Mandarin Show",
        "An Oral History of a Startup: Technology and the Future",
        "145. 口述创业史：和前高管聊用人观、上市与未来",
        "A Founder: Consumer Electronics, Death & Birth",
    ) == "zh"


def test_detect_japanese_beats_chinese_on_kana():
    # Japanese mixes kana with Han; kana presence must win over the Han→zh path.
    assert _sniff("ことばのラジオ 第100回 日本語の起源") == "ja"


def test_detect_korean_hangul():
    assert _sniff("아침 뉴스 공장 오늘의 소식") == "ko"


def test_detect_cyrillic_russian():
    assert _sniff("Разговоры о главном — большой выпуск про космос") == "ru"


def test_english_title_stays_undetected():
    assert _sniff("The Weekly Roundtable: markets, tech, and politics") is None


def test_stray_foreign_char_does_not_flip_english():
    # One Han char (a name) in an otherwise-English title must not force zh:
    # it fails the absolute floor and the minimum-share guard.
    assert _sniff("Interview with 李 about startups and venture capital") is None


def test_stray_hangul_token_does_not_flip_english():
    # A Korean proper noun in an English feed must not flip it to `ko` — kana and
    # hangul go through the same floor/share gate as Han, not mere presence.
    assert _sniff(
        "Some English Tech Channel",
        "Why 오징어 게임 broke Netflix",
        "The future of streaming and AI",
        "Interviewing a startup founder about growth",
    ) is None


def test_stray_kana_token_does_not_flip_english():
    # Same for a Japanese-script token buried in English titles.
    assert _sniff(
        "English Anime Reviews",
        "Reviewing けものフレンズ and why it matters",
        "Top 10 shows of the year, ranked and reviewed",
        "A deep dive into animation studios and budgets",
    ) is None


def test_empty_texts_undetected():
    assert _sniff("", "  ") is None


# -- resolve_language precedence ---------------------------------------------


def test_explicit_override_wins_over_script():
    # Han titles would sniff zh, but an explicit code is authoritative.
    assert lang.resolve_language("ja", None, ["口述创业史"]) == "ja"


def test_explicit_en_forces_english_path_no_fallthrough():
    # language="en" means "force English" — must NOT fall through to the sniff.
    assert lang.resolve_language("en", None, ["口述创业史"]) is None


def test_declared_beats_script():
    # A declared RSS language is trusted over a script sniff of the titles.
    assert lang.resolve_language(None, "de", ["Just some English-looking text"]) == "de"


def test_declared_region_normalized():
    assert lang.resolve_language(None, "pt-BR", []) == "pt"


def test_declared_english_falls_through_to_script():
    # Declared "en" isn't authoritative (only an explicit override is), so a
    # clearly non-English script still wins.
    assert lang.resolve_language(None, "en", ["口述创业史用人观"]) == "zh"
