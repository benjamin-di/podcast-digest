"""Config: env (via pydantic-settings) + feeds.toml (read on every tick).

Env carries secrets, paths and backend choices. `feeds.toml` is the
user-editable podcast list and the recipient — mounted into the container so it
can be edited without a rebuild.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Feed(BaseModel):
    """One subscription — a podcast RSS feed, a YouTube channel, or a single
    YouTube/audio episode.

    Exactly one *source* must be set:
      - `rss_url`      — pinned podcast RSS URL (wins over `search`)
      - `search`       — podcast name, resolved via the iTunes Search API
      - `youtube`      — a YouTube channel: a channel URL, `@handle`, raw
                         `UC...` channel id, or a `videos.xml` URL. Behaves like
                         a podcast (newest video per tick via the channel feed).
      - `youtube_video`— a single YouTube watch URL. A one-episode "feed":
                         transcribed/summarized/emailed once, then never again.
      - `audio_url`    — a single direct-audio URL (an .mp3/.m4a enclosure). A
                         one-shot like `youtube_video`: downloaded, transcribed,
                         summarized, emailed once, then deduped forever. The MCP
                         also resolves Apple Podcasts *episode* URLs to this.
    """

    name: str = Field(min_length=1)
    rss_url: str | None = None
    search: str | None = None
    youtube: str | None = None
    youtube_video: str | None = None
    audio_url: str | None = None
    # Episode metadata captured at ingest, for `audio_url` one-shots only. Unlike
    # `youtube_video` (re-resolved live each tick) a bare audio enclosure has no
    # feed to re-read, so the MCP persists what the Apple-episode lookup returned
    # here and the pipeline uses it instead of fabricating the processing time.
    # Both stay unset for a raw .mp3 URL (no metadata to capture) and for every
    # recurring source — `published_at` then resolves to "" so the email omits
    # the release line rather than printing today's date.
    published_at: str | None = None
    duration_seconds: int | None = Field(default=None, ge=0)
    # The real podcast/show name for an `audio_url` one-shot (the Apple-episode
    # lookup's collection name). The feed `name` of a one-shot is the *episode*
    # title, so without this the email/webhook would show the episode title where
    # the show name belongs ("Episode — Episode"). Unset for every other source.
    show: str | None = None
    # Minimum episode length (minutes) below which we skip entirely — teasers,
    # promos, conference soundbites. `None` (unset) means "inherit the file-level
    # default" (FeedsFile.min_minutes); an explicit `0` opts this feed out of the
    # floor (e.g. a clips channel you actually want). Enforced two ways: at RSS
    # parse time against the declared `itunes:duration`, AND — for every source —
    # against the *real* duration probed from the downloaded audio just before
    # transcription (catches missing/wrong declared durations and YouTube /
    # direct-audio sources, which carry no duration at ingest).
    min_minutes: int | None = Field(default=None, ge=0)
    # Spoken language of the episodes as an ISO 639-1 code ("zh" for Mandarin;
    # "ja", "de", … also work). Unset or "en" = the English path, byte-for-byte
    # unchanged (.en whisper model). Any other value switches transcription to
    # the multilingual whisper model (PODCAST_DIGEST_WHISPER_MODEL_MULTILINGUAL)
    # with `-l <code>`, and the summarizer still emits English — it reads the
    # original-language transcript directly (no translation step). The episode
    # title is never translated: the delivery log dedupes on (show, title).
    language: str | None = None
    # Post this feed's emailed episodes to the delivery webhook (when one is
    # configured at all — see PODCAST_DIGEST_WEBHOOK_URL). Opt-*out* only: default
    # True, set `webhook = false` to exclude a feed (e.g. pure entertainment).
    webhook: bool = True
    # Use the transcript the publisher links in the feed (Podcasting 2.0
    # `<podcast:transcript>`) instead of transcribing the audio, when one is
    # usable. Set `publisher_transcripts = false` for a feed whose transcripts
    # are poor; PODCAST_DIGEST_PUBLISHER_TRANSCRIPTS switches it off globally.
    publisher_transcripts: bool = True

    @field_validator(
        "rss_url", "search", "youtube", "youtube_video", "audio_url",
        mode="before",
    )
    @classmethod
    def _normalize_optional_source(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("feed name may not be blank")
        return stripped

    @model_validator(mode="after")
    def _require_one_source(self) -> Feed:
        sources = {
            "rss_url": self.rss_url,
            "search": self.search,
            "youtube": self.youtube,
            "youtube_video": self.youtube_video,
            "audio_url": self.audio_url,
        }
        active = [name for name, value in sources.items() if value and value.strip()]
        if len(active) != 1:
            raise ValueError(
                f"feed {self.name!r} needs exactly one source; got "
                f"{active or 'none'}"
            )
        one_shot_metadata = (self.published_at, self.duration_seconds, self.show)
        if any(value is not None for value in one_shot_metadata) and not self.audio_url:
            raise ValueError(
                "published_at, duration_seconds, and show are only valid "
                "with audio_url"
            )
        return self

    @property
    def source_language(self) -> str | None:
        """Non-English whisper `-l` code for this feed, else None (English).

        Reduces a regioned tag to its bare primary subtag (`zh-CN` → `zh`), so a
        hand-edited feeds.toml can't feed whisper a `-l zh-cn` it won't accept.
        """
        lang = (self.language or "").strip().lower().replace("_", "-").split("-", 1)[0]
        return None if lang in ("", "en") else lang

    @property
    def display_show(self) -> str:
        """Podcast name to show in the email/webhook. Falls back to `name` —
        which is correct for every source except a one-shot, where `name` is the
        episode title and `show` carries the real podcast name."""
        return self.show or self.name

    def identity(self) -> str:
        """A stable identifier safe for filesystem use."""
        slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in self.name.lower())
        return slug.strip("-") or "feed"


class FeedsFile(BaseModel):
    """The on-disk feeds.toml shape."""

    recipient: str = Field(min_length=1)
    # Default feed-poll cadence (feeds.toml overrides). INVARIANT: if you monitor
    # the heartbeat (PODCAST_DIGEST_HEARTBEAT_URL), the monitor's window (period +
    # grace) must exceed poll_seconds + scheduler.JITTER_SECONDS + worst-case tick
    # duration — the scheduler pings once per *healthy tick*, so a poll slower
    # than the window flaps. Raise this and widen the monitor to match.
    poll_seconds: int = Field(default=21600, gt=0)
    # Global minimum-length floor (minutes) applied to every feed that doesn't
    # set its own `min_minutes`. A per-feed value (including an explicit 0)
    # overrides this. Default 0 = no floor unless configured.
    min_minutes: int = Field(default=0, ge=0)
    feeds: list[Feed] = Field(default_factory=list)

    @field_validator("recipient")
    @classmethod
    def _normalize_recipient(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("recipient may not be blank")
        return stripped

    def min_minutes_for(self, feed: Feed) -> int:
        """Effective length floor for a feed: its own value, else the file default."""
        return feed.min_minutes if feed.min_minutes is not None else self.min_minutes


class Settings(BaseSettings):
    """Environment-derived config. Loaded once at startup; refer to attributes.

    An empty variable counts as unset, so compose wiring like `X: ${Y:-}` falls
    back to the default instead of breaking a typed setting.
    """

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=True, populate_by_name=True,
        env_ignore_empty=True,
    )

    # --- Summarizer -------------------------------------------------------
    # Where the LLM work (summary + optional transcript cleanup) runs:
    #   "api"     — call a model API in-process: Anthropic, OpenAI, or any
    #               OpenAI-compatible server (Ollama, LM Studio, vLLM, …).
    #   "command" — run a command for each LLM call (locally, or on another host
    #               over SSH): e.g. `claude -p`, `llm`, `ollama run <model>`, or a
    #               runner speaking the JSON protocol (docs/command-backend.md).
    PODCAST_DIGEST_SUMMARIZER: Literal["api", "command"] = "api"

    # api backend. PODCAST_DIGEST_MODEL picks the model; the provider follows from
    # its name (claude-* → Anthropic, anything else → OpenAI-compatible) unless
    # PODCAST_DIGEST_PROVIDER says otherwise. With PODCAST_DIGEST_OPENAI_BASE_URL
    # set, requests go to that OpenAI-compatible server (Chat Completions) and
    # OPENAI_API_KEY is optional.
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    PODCAST_DIGEST_MODEL: str = "claude-opus-5-5"
    PODCAST_DIGEST_PROVIDER: Literal["auto", "anthropic", "openai"] = "auto"
    PODCAST_DIGEST_OPENAI_BASE_URL: str = ""

    # command backend. The command is split shell-style and receives the prompt
    # on stdin. Protocol "text" (default): stdin is the full prompt, stdout the
    # answer. Protocol "json": stdin is a JSON request, stdout NDJSON events (see
    # docs/command-backend.md). With PODCAST_DIGEST_SSH_HOST set, the command runs
    # on that host over SSH instead of locally.
    PODCAST_DIGEST_COMMAND: str = ""
    PODCAST_DIGEST_COMMAND_PROTOCOL: Literal["text", "json"] | None = None
    PODCAST_DIGEST_COMMAND_TIMEOUT: int = Field(default=300, gt=0)
    # Optional model tier passed through to a JSON-protocol runner (`tier`);
    # what the names mean is up to the runner. Empty = not sent.
    PODCAST_DIGEST_TIER: str = ""
    PODCAST_DIGEST_SSH_HOST: str = ""
    PODCAST_DIGEST_SSH_PORT: int = Field(default=22, ge=1, le=65535)
    PODCAST_DIGEST_SSH_USER: str = ""
    # Private key + (optional) pinned known_hosts inside the container. The
    # entrypoint materializes these from the *_B64 variables if provided.
    PODCAST_DIGEST_SSH_KEY: str = ""
    PODCAST_DIGEST_SSH_KNOWN_HOSTS: str = ""

    # Summary prompt. "general" suits any podcast; "finance" writes for a
    # finance-literate reader (tickers, levels, a markets glossary). A prompt
    # file replaces the built-in system prompt entirely; its output must be an
    # HTML fragment (see README "Summary prompt").
    PODCAST_DIGEST_SUMMARY_STYLE: Literal["general", "finance"] = "general"
    PODCAST_DIGEST_SUMMARY_PROMPT_FILE: str = ""

    # Attach the episode transcript to the digest email:
    #   "none"    — no attachment (default)
    #   "raw"     — attach the raw whisper .txt as-is
    #   "cleaned" — have the summarizer clean the transcript into Markdown
    #               (speaker labels, punctuation, ASR fixes, ad-reads removed),
    #               summarize from it, and attach the .md.
    PODCAST_DIGEST_ATTACH_TRANSCRIPT: Literal["none", "raw", "cleaned"] = "none"

    # --- Mail -------------------------------------------------------------
    # "smtp" (any mail server) or "graph" (Microsoft 365 via the Graph API with
    # an app registration holding Mail.Send). Unset = "graph" when all three
    # Graph credentials are present, else "smtp".
    PODCAST_DIGEST_MAILER: Literal["smtp", "graph"] | None = None
    PODCAST_DIGEST_MAIL_FROM: str = ""
    PODCAST_DIGEST_SMTP_HOST: str = ""
    PODCAST_DIGEST_SMTP_PORT: int = Field(default=587, ge=1, le=65535)
    PODCAST_DIGEST_SMTP_USERNAME: str = ""
    PODCAST_DIGEST_SMTP_PASSWORD: str = ""
    # "starttls" (port 587), "ssl" (implicit TLS, port 465), or "none" (a local
    # relay only — credentials would travel in clear text).
    PODCAST_DIGEST_SMTP_SECURITY: Literal["starttls", "ssl", "none"] = "starttls"
    PODCAST_DIGEST_GRAPH_TENANT_ID: str = ""
    PODCAST_DIGEST_GRAPH_CLIENT_ID: str = ""
    PODCAST_DIGEST_GRAPH_CLIENT_SECRET: str = ""

    # --- Paths ------------------------------------------------------------
    PODCAST_DIGEST_DATA_DIR: Path = Path("/data")
    PODCAST_DIGEST_CONFIG: Path = Path("/app/feeds.toml")

    # --- Transcription ----------------------------------------------------
    PODCAST_DIGEST_WHISPER_BIN: str = "/usr/local/bin/whisper-cli"
    PODCAST_DIGEST_WHISPER_MODEL: Path = Path("/data/models/ggml-medium.en.bin")
    # Model used for feeds with a non-English `language` in feeds.toml. The
    # default (`large-v3-turbo`, ~1.6 GB) is near-large-v3 accuracy at a
    # fraction of the compute; point at ggml-large-v3.bin when a feed needs the
    # quality ceiling. Only downloaded where deploy opts in — see
    # PODCAST_DIGEST_WHISPER_MODELS in deploy/prepare-models.sh.
    PODCAST_DIGEST_WHISPER_MODEL_MULTILINGUAL: Path = Path(
        "/data/models/ggml-large-v3-turbo.bin"
    )
    PODCAST_DIGEST_WHISPER_THREADS: int = Field(default=8, gt=0)
    PODCAST_DIGEST_WHISPER_LANGUAGE: str = "en"
    # Prefer a transcript the publisher links in the feed (Podcasting 2.0) over
    # downloading and transcribing the audio. See publisher.py.
    PODCAST_DIGEST_PUBLISHER_TRANSCRIPTS: bool = True

    # How often the scheduler checks the "process now" sentinel between scheduled
    # ticks. The MCP bumps the sentinel after an ingest so processing starts
    # within this many seconds instead of waiting out poll_seconds. See wake.py.
    PODCAST_DIGEST_WAKE_POLL_SECONDS: float = Field(default=15.0, gt=0)

    # YouTube (yt-dlp) knobs. Only relevant when a feed uses `youtube` /
    # `youtube_video`.
    #   FORMAT  — yt-dlp `-f` selector for the audio to grab.
    #   COOKIES — optional path to a Netscape cookies.txt, for videos that
    #             require sign-in / age-gate. Empty = no cookies.
    PODCAST_DIGEST_YTDLP_FORMAT: str = "bestaudio/best"
    PODCAST_DIGEST_YTDLP_COOKIES: str = ""

    # --- Delivery webhook -------------------------------------------------
    # After an episode is emailed, POST its metadata, summary and transcript as
    # JSON to this URL (e.g. a knowledge-base ingest route, n8n, a notes app).
    #   URL       → empty = feature OFF (the default).
    #   TOKEN     → optional bearer token (a secret; never logged).
    #   FIELDS    → JSON object of extra static fields merged into every payload.
    #   ACK_FIELD → when set, a delivery counts only if the JSON response carries a
    #               non-empty value under this key (e.g. "task_id"); otherwise
    #               any 2xx JSON-object response counts.
    # See docs/webhook.md for the payload and retry contract.
    PODCAST_DIGEST_WEBHOOK_URL: str = ""
    PODCAST_DIGEST_WEBHOOK_TOKEN: str = ""
    PODCAST_DIGEST_WEBHOOK_TIMEOUT: int = Field(default=30, gt=0)
    PODCAST_DIGEST_WEBHOOK_FIELDS: dict[str, Any] = Field(default_factory=dict)
    PODCAST_DIGEST_WEBHOOK_ACK_FIELD: str = ""

    # --- Monitoring -------------------------------------------------------
    # Dead-man's-switch: GET this URL (e.g. a Healthchecks or Uptime Kuma push
    # URL) once per tick *only when the summarizer is healthy* (zero
    # summarizer-transport failures). A summarizer outage (API down, SSH key
    # rotated) makes every tick fail → no ping → the monitor alerts after its
    # grace window. Empty = OFF. Gated on the summarizer specifically, not the
    # generic failed count, so an unrelated feed-resolution error never
    # suppresses the heartbeat. See scheduler.run_forever.
    PODCAST_DIGEST_HEARTBEAT_URL: str = ""

    # Shared delivery log (append-only TSV). When set, every episode this
    # instance emails is appended to this file, and episodes already in it are
    # treated as already sent. Point several instances (or an old and a new
    # host during a migration) at a shared or replicated copy so none of them
    # re-sends what another already delivered. It complements state.db, which
    # stays local. Empty = OFF (the single-instance default). See emitted.py.
    PODCAST_DIGEST_EMITTED_LOG: str = ""

    LOG_LEVEL: str = "INFO"

    @model_validator(mode="after")
    def _validate_mode_requirements(self) -> Settings:
        if self.summarizer_backend == "command" and not self.PODCAST_DIGEST_COMMAND.strip():
            raise ValueError(
                "PODCAST_DIGEST_COMMAND is required when "
                "PODCAST_DIGEST_SUMMARIZER=command"
            )
        return self

    # Derived settings
    @property
    def summarizer_backend(self) -> Literal["api", "command"]:
        return self.PODCAST_DIGEST_SUMMARIZER

    @property
    def command_protocol(self) -> Literal["text", "json"]:
        return self.PODCAST_DIGEST_COMMAND_PROTOCOL or "text"

    @property
    def command_model(self) -> str | None:
        """Model to request from a JSON-protocol runner: only an explicitly
        configured PODCAST_DIGEST_MODEL (otherwise the runner's tier decides)."""
        if "PODCAST_DIGEST_MODEL" in self.model_fields_set:
            return self.PODCAST_DIGEST_MODEL
        return None

    @property
    def mailer_backend(self) -> Literal["smtp", "graph"]:
        if self.PODCAST_DIGEST_MAILER:
            return self.PODCAST_DIGEST_MAILER
        graph = (
            self.PODCAST_DIGEST_GRAPH_TENANT_ID,
            self.PODCAST_DIGEST_GRAPH_CLIENT_ID,
            self.PODCAST_DIGEST_GRAPH_CLIENT_SECRET,
        )
        return "graph" if all(v.strip() for v in graph) else "smtp"

    @property
    def webhook_url(self) -> str:
        return self.PODCAST_DIGEST_WEBHOOK_URL.strip()

    @property
    def webhook_ack_field(self) -> str:
        return self.PODCAST_DIGEST_WEBHOOK_ACK_FIELD.strip()

    @property
    def webhook_fields(self) -> dict[str, Any]:
        return dict(self.PODCAST_DIGEST_WEBHOOK_FIELDS)

    @property
    def db_path(self) -> Path:
        return self.PODCAST_DIGEST_DATA_DIR / "state.db"

    @property
    def audio_dir(self) -> Path:
        return self.PODCAST_DIGEST_DATA_DIR / "audio"

    @property
    def transcripts_dir(self) -> Path:
        return self.PODCAST_DIGEST_DATA_DIR / "transcripts"

    @property
    def models_dir(self) -> Path:
        return self.PODCAST_DIGEST_DATA_DIR / "models"

    def ensure_dirs(self) -> None:
        for p in (self.audio_dir, self.transcripts_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)


def load_feeds(path: Path) -> FeedsFile:
    """Parse and validate feeds.toml. Re-call on every tick so edits propagate."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    try:
        return FeedsFile.model_validate(data)
    except ValidationError as e:
        raise RuntimeError(f"invalid {path}: {e}") from e
