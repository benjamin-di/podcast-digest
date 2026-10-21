"""podcast-digest: download, transcribe, summarize, and email podcasts."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("podcast-digest")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0+unknown"

PROJECT_URL = "https://github.com/benjamin-di/podcast-digest"
# Sent with every outbound request (feeds, pages, audio, transcripts), so
# publishers can identify the client.
USER_AGENT = f"podcast-digest/{__version__} (+{PROJECT_URL})"
