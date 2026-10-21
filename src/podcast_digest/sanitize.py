"""HTML sanitization for model-produced summaries.

One module so every consumer — the summarize stage (production choke point),
the email wrapper (outbound surface defense), and the one-time DB migration
that cleans rows stored before production-time sanitizing existed — applies
the identical policy. Lives outside pipeline.py because db.py needs it at
migration time and importing the orchestrator from the storage layer would be
a cycle.
"""
from __future__ import annotations

import nh3

from .net import HTTP_SCHEMES, is_http_url

# The exact tag set the summarizer system prompt is allowed to emit. Anything
# outside it — <script>, <style>, <img>, <iframe>, event handlers, styles — is
# stripped server-side, and <a href> is limited to http(s). The model output is
# untrusted input (a hostile transcript / show-notes can prompt-inject it into
# producing a phishing <a> or layout-hijacking markup that would otherwise ship
# verbatim in a trusted-sender email). We sanitize regardless of how well the
# prompt behaves — defense that doesn't depend on the model complying.
SUMMARY_ALLOWED_TAGS = {"h2", "h3", "p", "ul", "ol", "li", "strong", "em", "br", "a"}
SUMMARY_ALLOWED_ATTRS = {"a": {"href"}}


def sanitize_summary_html(html: str) -> str:
    """Strip the model's summary down to the allowed tag/attribute set."""
    return nh3.clean(
        html,
        tags=SUMMARY_ALLOWED_TAGS,
        attributes=SUMMARY_ALLOWED_ATTRS,
        url_schemes=set(HTTP_SCHEMES),
        link_rel="noopener noreferrer",
    )


def safe_listen_url(audio_url: str) -> str | None:
    """Return the URL only if it's an http(s) URL — else None (omit the link).

    The enclosure href is fully feed-controlled; a `javascript:`/`data:` scheme
    in a "Listen to the episode" link would be a live injection vector.
    """
    return audio_url if is_http_url(audio_url) else None
