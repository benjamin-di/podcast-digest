"""Best-effort source-language detection for `ingest`.

The digest transcribes a feed with the multilingual whisper model + `-l <code>`
only when its feeds.toml `language` is a non-English code; English feeds stay on
the fast `.en` model, byte-for-byte unchanged (see podcast_digest.config.Feed and
pipeline). So `ingest` wants to set `language` automatically — and must never set
it wrong on an English feed, which would swap in the slower multilingual model
and force `-l en` decoding.

`resolve_language` picks a code from three signals, most authoritative first:

  1. An EXPLICIT `ingest(language=...)` override — always wins. `"en"` (or any
     empty/English value) resolves to None, i.e. "keep the English path".
  2. A language DECLARED by the source — an RSS `<language>` element, or yt-dlp's
     `language` field. Region is stripped (`zh-CN` → `zh`) to a bare ISO-639-1
     code, which is what whisper's `-l` expects.
  3. The dominant non-Latin Unicode SCRIPT of the latest titles — the reliable
     tell for a feed that declares nothing (e.g. a YouTube channel whose titles
     are Han characters). Latin-script text is deliberately left undetected:
     script can't tell English from German, so those need signal (1) or (2).

Everything is best-effort: callers pass whatever titles/metadata they already
fetched, and an undetectable feed simply stays on the English path.
"""
from __future__ import annotations

from collections.abc import Iterable


def normalize(code: str | None) -> str | None:
    """A raw language tag → a bare lower-case ISO-639-1 code, or None.

    None/empty and any English tag (`en`, `en-US`, ` EN `) return None — the
    signal for "use the English path unchanged". A regioned tag is reduced to
    its primary subtag (`pt-BR` → `pt`), since whisper's `-l` wants the bare
    code. A value that isn't a plausible 2–3 letter code returns None rather
    than being written through as a bogus `-l` argument.
    """
    if not code:
        return None
    primary = code.strip().lower().replace("_", "-").split("-", 1)[0]
    if primary in ("", "en"):
        return None
    if not (2 <= len(primary) <= 3 and primary.isalpha()):
        return None
    return primary


# Unicode scripts that map cleanly to a whisper language. Han is intentionally
# resolved to Chinese: Japanese also uses Han, but it always mixes in kana, and
# kana/Hangul are checked first below — so Han-without-kana is Chinese.
# A non-Latin script must contribute at least this many letters, and this share
# of all letters, to name the language. The floor rejects a stray foreign name;
# the (low) share tolerates feeds that pair each non-Latin title with a Latin
# one — e.g. a Mandarin channel that also lists a romanized name and an
# English-translated title per episode (the audio is still Mandarin, so its Han
# titles must win over the Latin metadata that merely labels them).
_SCRIPT_MIN_LETTERS = 6
_SCRIPT_MIN_SHARE = 0.15


def _script_language(text: str) -> str | None:
    """Dominant non-Latin script of `text` → an ISO-639-1 code, else None.

    Counts letters per script, then picks the language with the most letters and
    requires it to clear both an absolute floor and a minimum share of ALL
    letters — so a lone foreign name in an English title (a couple of kana, one
    Han character) can never flip the result, while genuine non-Latin content
    still wins despite Latin noise (romanized names, translated titles).

    Japanese is disambiguated from Chinese by kana: kana and Han are both
    Japanese *content*, so when any kana is present they're pooled under `ja`;
    Han with no kana is `zh`. Hangul is `ko`. Every path goes through the same
    floor/share gate — none resolves on mere presence.
    """
    latin = han = kana = hangul = 0
    others: dict[str, int] = {}
    for ch in text:
        o = ord(ch)
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or (0x00C0 <= o <= 0x024F):
            latin += 1
        elif 0x3040 <= o <= 0x30FF:                      # hiragana + katakana
            kana += 1
        elif 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            hangul += 1
        elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:   # CJK unified (+ ext A)
            han += 1
        elif 0x0400 <= o <= 0x04FF:
            others["ru"] = others.get("ru", 0) + 1       # Cyrillic
        elif 0x0600 <= o <= 0x06FF:
            others["ar"] = others.get("ar", 0) + 1       # Arabic
        elif 0x0590 <= o <= 0x05FF:
            others["he"] = others.get("he", 0) + 1       # Hebrew
        elif 0x0900 <= o <= 0x097F:
            others["hi"] = others.get("hi", 0) + 1       # Devanagari
        elif 0x0370 <= o <= 0x03FF:
            others["el"] = others.get("el", 0) + 1       # Greek
        elif 0x0E00 <= o <= 0x0E7F:
            others["th"] = others.get("th", 0) + 1       # Thai

    total = latin + han + kana + hangul + sum(others.values())
    if not total:
        return None
    # kana present ⇒ the Han in this text is Japanese kanji, so pool them under ja.
    cjk = ("ja", han + kana) if kana else ("zh", han)
    code, count = max([cjk, ("ko", hangul), *others.items()], key=lambda kv: kv[1])
    if count >= _SCRIPT_MIN_LETTERS and count / total >= _SCRIPT_MIN_SHARE:
        return code
    return None


def resolve_language(
    explicit: str | None,
    declared: str | None,
    texts: Iterable[str] = (),
) -> str | None:
    """Final feeds.toml `language` for a feed, or None for the English path.

    `explicit` (the `ingest` override) is authoritative when non-empty — even
    when it resolves to None (`language="en"` means "force English"). Otherwise
    fall back to the source-`declared` code, then to a script sniff of `texts`
    (feed/channel name + latest episode titles).
    """
    if explicit is not None and explicit.strip():
        return normalize(explicit)
    declared_code = normalize(declared)
    if declared_code:
        return declared_code
    joined = " ".join(t for t in texts if t)
    return _script_language(joined) if joined else None
