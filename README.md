# podcast-digest

Follow podcasts and YouTube channels, transcribe every new episode locally with
[whisper.cpp](https://github.com/ggml-org/whisper.cpp), and get an email per
episode with an LLM-written summary — TL;DR, speakers, key takeaways, claims,
timestamped highlights, "worth listening in full?" — and optionally a cleaned-up
transcript attached.

- **Sources:** podcast RSS feeds, a podcast name (resolved via the iTunes
  Search API), Apple Podcasts links, YouTube channels, single YouTube videos and
  one-off audio URLs.
- **Publisher transcripts first:** when a feed links the episode's own
  transcript (Podcasting 2.0 `<podcast:transcript>`), it is used as is — no
  download, no transcription, speaker names included.
- **Local transcription otherwise:** whisper.cpp, accelerated by an Intel iGPU
  through OpenVINO when one is available, on the CPU otherwise. Non-English
  feeds use a multilingual model and still get an English summary.
- **Any LLM:** Anthropic, OpenAI, a local or self-hosted OpenAI-compatible
  server (Ollama, LM Studio, vLLM), or any command-line tool (`claude -p`,
  `llm`, …), locally or on another machine over SSH.
- **Email:** any SMTP server, or Microsoft 365 via the Graph API.
- **No duplicates, no floods:** a new feed sends only its latest episode; the
  same programme published under several feeds (or re-uploaded) is recognised by
  its transcript and sent once; every stage is resumable, so a crash or an
  outage never re-sends or skips an episode.
- **Optional extras:** a JSON webhook per delivered episode (for a knowledge
  base, n8n, a notes app), a dead-man's-switch heartbeat, and an MCP server to
  manage subscriptions from Claude on your phone.

## Quickstart (Docker)

```bash
git clone https://github.com/benjamin-di/podcast-digest.git && cd podcast-digest
mkdir -p config && cp feeds.toml config/feeds.toml   # your subscriptions + recipient
cp example.env .env                                  # mail + LLM settings
docker compose --env-file .env -f deploy/docker-compose.yml up -d --build
docker compose --env-file .env -f deploy/docker-compose.yml logs -f
```

The first start downloads the whisper model (~1.5 GB for `medium.en`) into the
data volume. With an Intel GPU passed through (uncomment `devices` in
`deploy/docker-compose.yml`), it also converts the model's encoder for OpenVINO:
another ~1.5 GB download and a few minutes, once. The service then polls every
`poll_seconds` (default 6 h); add `--profile mcp` to also run the MCP server.

The engine image is **x86-64 only** (its OpenVINO base image exists only for
that). On Apple Silicon or another ARM host, use
[Running without Docker](#running-without-docker); the MCP image runs anywhere.
Prebuilt images are published for every release:
`ghcr.io/benjamin-di/podcast-digest-engine`, `…-mcp-server` and
`…-transcriber`.

## Subscriptions: `feeds.toml`

The shows, addresses and IDs below are placeholders; put your own in.

```toml
recipient = "you@example.com"   # comma-separated for several
poll_seconds = 21600            # between polls (default 6 h)
min_minutes = 5                 # skip anything shorter (teasers, promos)

[[feeds]]
name = "Market Hours"
search = "Market Hours"         # a show's name, resolved via the iTunes Search API

[[feeds]]
name = "Company Histories"
rss_url = "https://feeds.example.com/company-histories.xml"  # a pinned feed (preferred)
min_minutes = 15                # per-feed override; 0 opts a feed out

[[feeds]]
name = "Lecture Hall"
youtube = "@lecturehall"        # channel URL, @handle, UC… id or videos.xml URL

[[feeds]]
name = "A talk I want once"
youtube_video = "https://www.youtube.com/watch?v=exampleVid1"

[[feeds]]
name = "A single episode"
show = "The show it belongs to" # optional: used as the show name in the email
audio_url = "https://cdn.example.com/episode-123.mp3"

[[feeds]]
name = "示例播客"
rss_url = "https://feeds.example.com/example-mandarin-show.xml"
language = "zh"                 # transcribe in Mandarin, summarize in English
webhook = false                 # leave this feed out of the delivery webhook
publisher_transcripts = false   # always transcribe locally, ignore the feed's transcripts
```

Each feed needs exactly one source: `rss_url`, `search`, `youtube`,
`youtube_video` or `audio_url`. The file is re-read on every poll, so edits need
no restart. `podcast-digest check-config path/to/feeds.toml` validates a file
without credentials or network.

`min_minutes` is checked against the declared duration and, for every recurring
source, against the real audio length measured before transcription. One-off
sources (`youtube_video`, `audio_url`) are explicit picks and exempt. For an
`audio_url`, `name` is the episode title; `show` (optional) names the show it
belongs to. Feed, audio and transcript URLs that resolve to private or local
network addresses are refused.

## Configuration

Everything else is environment variables, or a `.env` file in the working
directory. [`example.env`](example.env) is a commented starting point. An empty
variable counts as unset.

### Mail

| Variable | Default | |
|---|---|---|
| `PODCAST_DIGEST_MAIL_FROM` | _(required)_ | Sender address |
| `PODCAST_DIGEST_MAILER` | `smtp`, or `graph` when the Graph credentials are set | Backend |
| `PODCAST_DIGEST_SMTP_HOST` | | SMTP server |
| `PODCAST_DIGEST_SMTP_PORT` | `587` | |
| `PODCAST_DIGEST_SMTP_SECURITY` | `starttls` | `starttls`, `ssl` (port 465) or `none` (local relay only) |
| `PODCAST_DIGEST_SMTP_USERNAME` / `_PASSWORD` | | Login, if the server needs one |
| `PODCAST_DIGEST_GRAPH_TENANT_ID` / `_CLIENT_ID` / `_CLIENT_SECRET` | | Microsoft 365: an Entra ID app registration with the `Mail.Send` application permission (scope it to the sender mailbox) |

Sending is not idempotent, so when a message has been fully transmitted but the
server's answer is lost, the episode is treated as sent (logged as
`mail.probably_delivered`) rather than risking duplicate digests.

### Summarizer

| Variable | Default | |
|---|---|---|
| `PODCAST_DIGEST_SUMMARIZER` | `api` | `api` or `command` |
| `PODCAST_DIGEST_MODEL` | `claude-opus-5-5` | `api`: the model. `claude-*` uses Anthropic (`claude-sonnet-5-5` is cheaper), anything else OpenAI or a compatible server |
| `PODCAST_DIGEST_PROVIDER` | `auto` | Force `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | | The provider's key |
| `PODCAST_DIGEST_OPENAI_BASE_URL` | | An OpenAI-compatible server, e.g. `http://localhost:11434/v1` for Ollama (no key needed) |
| `PODCAST_DIGEST_COMMAND` | | `command`: run this per LLM call, prompt on stdin |
| `PODCAST_DIGEST_SSH_HOST` (+ `_PORT`, `_USER`, `_KEY` or `_KEY_B64`, `_KNOWN_HOSTS` or `_KNOWN_HOSTS_B64`) | | Run the command on another host over SSH |

The command backend, its `text` and `json` protocols and its failure handling
are described in [docs/command-backend.md](docs/command-backend.md). The
container image has an SSH client but no model CLIs: to run a CLI such as
`claude -p` inside it, build on the image and install the CLI, or run it on
another host over SSH.

Whatever the backend, a summarizer that is down or misconfigured (unreachable,
rejected credentials, an unknown model) does not use up an episode's retries:
affected episodes wait — one canary attempt per poll after ten failed ones — and
resume without re-downloading or re-transcribing when it works again. An episode
is dropped only after 30 such failures (about a week at the default poll), so a
permanently broken setup does not keep work pending forever. The heartbeat stays
silent while the summarizer is failing.

### Summary prompt

`PODCAST_DIGEST_SUMMARY_STYLE` picks the built-in prompt: `general` (default)
or `finance` (for a finance-literate reader: tickers, levels, a markets
glossary). To write your own, point `PODCAST_DIGEST_SUMMARY_PROMPT_FILE` at a
text file; it replaces the system prompt. The model's reply goes into the email
as HTML, so the prompt must ask for an HTML fragment using only `h2`, `h3`, `p`,
`ul`, `ol`, `li`, `strong`, `em`, `br` and `a` (anything else is stripped). The
user message carries the show, title, date, duration, show notes and the
transcript with `[hh:mm:ss]` markers.

### Transcript attachment

`PODCAST_DIGEST_ATTACH_TRANSCRIPT`:

- `none` (default) — no attachment.
- `raw` — the whisper `.txt`.
- `cleaned` — the summarizer turns the transcript into readable Markdown
  (speaker labels, punctuation, obvious ASR errors fixed, ad reads removed),
  chunked so any length fits. The summary is then written from the cleaned
  text. A failed cleanup is retried on the next two ticks; after that the
  digest ships anyway, summarized from the raw transcript. A publisher
  transcript is attached as Markdown without this pass (see below).

### Transcription

| Variable | Default | |
|---|---|---|
| `PODCAST_DIGEST_PUBLISHER_TRANSCRIPTS` | `true` | Use transcripts the feeds link (see below) before transcribing locally |
| `PODCAST_DIGEST_WHISPER_MODELS` | `medium.en` | Models fetched on first start; add `large-v3-turbo` for non-English feeds |
| `PODCAST_DIGEST_WHISPER_MODEL` | `/data/models/ggml-medium.en.bin` | English model |
| `PODCAST_DIGEST_WHISPER_MODEL_MULTILINGUAL` | `/data/models/ggml-large-v3-turbo.bin` | For feeds with a `language` |
| `PODCAST_DIGEST_WHISPER_LANGUAGE` | `en` | Language passed to whisper for feeds without a `language` |
| `PODCAST_DIGEST_WHISPER_THREADS` | `8` | CPU threads |
| `PODCAST_DIGEST_WHISPER_BIN` | `/usr/local/bin/whisper-cli` | whisper.cpp binary |
| `PODCAST_DIGEST_OPENVINO` | `auto` | Convert models for OpenVINO: `auto` (only with a `/dev/dri` GPU), `on`, `off` |
| `WHISPER_GPU_LOCK_PATH` | | A lock file shared with other containers that use the same GPU |
| `PODCAST_DIGEST_YTDLP_FORMAT` | `bestaudio/best` | yt-dlp audio selector |
| `PODCAST_DIGEST_YTDLP_COOKIES` | | A `cookies.txt` for sign-in or age-gated videos |

### Optional

| Variable | |
|---|---|
| `PODCAST_DIGEST_WEBHOOK_URL` (+ `_TOKEN`, `_FIELDS`, `_ACK_FIELD`, `_TIMEOUT`) | POST every delivered episode as JSON — see [docs/webhook.md](docs/webhook.md) |
| `PODCAST_DIGEST_HEARTBEAT_URL` | GET after every tick in which the summarizer worked — point a dead-man's switch (Healthchecks, Uptime Kuma, …) at it; its window must exceed `poll_seconds` + 10 min jitter + a slow tick |
| `PODCAST_DIGEST_EMITTED_LOG` | A shared delivery log: instances pointed at the same (or a replicated) file never re-send each other's episodes — for a standby instance or a host move |
| `PODCAST_DIGEST_DATA_DIR` | State, models, transcripts (default `/data`) |
| `PODCAST_DIGEST_CONFIG` | Path of `feeds.toml` (default `/app/feeds.toml`; in the MCP image `/app/config/feeds.toml`) |
| `PODCAST_DIGEST_WAKE_POLL_SECONDS` | How often the scheduler checks for an MCP "process now" signal (default 15) |
| `LOG_LEVEL` | Default `INFO`; logs are one JSON object per line when not on a terminal |

## Non-English podcasts

Set `language` on a feed (ISO 639-1: `zh`, `ja`, `de`, …). Its episodes are
transcribed in that language with the multilingual model, and the summary is
written in English from the original-language transcript, with a "translated
from" note in the email; titles stay as published. Mandarin gets extra guidance:
pinyin with characters on first mention, and correct 万/亿 unit conversion.
Remember to add the multilingual model to `PODCAST_DIGEST_WHISPER_MODELS`.

## Publisher transcripts

Many feeds link a transcript for each episode
([Podcasting 2.0 `<podcast:transcript>`](https://github.com/Podcastindex-org/podcast-namespace/blob/main/docs/tags/transcript.md)),
made or edited by the publisher. When one is usable it replaces the download and
the local transcription — minutes of compute per episode — and usually brings
punctuation and speaker names along.

- Formats: the JSON transcript format, WebVTT and SubRip (timed), then HTML and
  plain text. Among an item's links, the feed's language comes first, then
  timed formats.
- Each candidate must be plausible for the episode: at least 40 words per
  minute of its declared length (100 at least) and, for timed formats, timing
  that covers 60 % of it. Anything else — an error, an unknown format, a
  teaser-length text — moves on to the next link, and finally to local
  transcription in the same poll.
- With `PODCAST_DIGEST_ATTACH_TRANSCRIPT=cleaned`, the publisher's transcript is
  attached as Markdown with speaker turns and timestamps, without an LLM
  cleanup pass. Every email made from one notes "Summarized from the
  publisher's transcript".
- Switch it off globally with `PODCAST_DIGEST_PUBLISHER_TRANSCRIPTS=false`, or
  per feed with `publisher_transcripts = false`. The MCP `preview` tool says
  whether a feed's recent episodes link transcripts.

## MCP server (manage feeds from Claude)

`podcast-digest-mcp` is a small MCP server that edits `feeds.toml`, so you can
subscribe from Claude on your phone. It never downloads or processes anything
itself; it writes the file and wakes the scheduler, which starts within ~15 s.

| Tool | |
|---|---|
| `ingest(url, name?, min_minutes?, language?)` | Subscribe to a URL (see below); detects non-English feeds |
| `preview(url)` | Dry run: what `ingest` would add, with the latest episode titles |
| `subscription_list()` / `subscription_remove(name)` | List or remove entries |
| `process_now()` | Run a tick now |

| URL | Becomes |
|---|---|
| RSS feed | `rss_url` |
| Apple Podcasts show page | its RSS feed, via iTunes Lookup |
| Apple Podcasts episode link (`?i=…`) | that episode's audio, once (`audio_url`) |
| YouTube channel | `youtube` |
| YouTube video / Short / youtu.be link | `youtube_video` (once) |
| Direct audio link (served as `audio/*`) | `audio_url` (once) |

Run it with the compose profile `mcp` (published on `127.0.0.1:8000`). It
requires `PODCAST_MCP_BEARER_TOKEN`; clients send it as
`Authorization: Bearer …`. Outside compose it listens on `HOST`:`PORT`
(default `0.0.0.0:8000`) and logs at `LOG_LEVEL`. To use it from claude.ai or the Claude apps, publish
`/mcp` through a TLS reverse proxy and add it as a custom connector. The MCP
needs write access to the directory holding `feeds.toml` (it replaces the file
atomically) and shares it with the digest.

## How episodes are selected

- A feed seen for the first time sends only its **latest** episode; the rest of
  the archive is recorded as seen. After that, each poll processes the newest
  unseen episode per feed and records older ones as seen — no back-catalogue
  flood after a restart, a lost database or a long outage.
- With the shared delivery log on (`PODCAST_DIGEST_EMITTED_LOG`), globally
  unique GUIDs (UUIDs, long hex IDs, URL GUIDs, namespaced platform IDs) are
  matched across all feeds before downloading, so the same episode in an
  umbrella feed and a show feed is not even fetched twice. Without it, the copy
  is caught after transcription by the check below.
- After transcription, the transcript is compared with everything delivered in
  the last 30 days (normalized SHA-256 plus a 128-value MinHash): a lightly
  edited copy of a full programme is skipped, while a clip followed later by
  the full episode is still sent.
- Every episode moves forward through `discovered → downloaded → transcribed →
  summarized → emailed`, persisted in SQLite. A failure leaves it at its last
  completed stage and the next poll resumes from there; five consecutive
  episode-level failures retire it (summarizer outages count separately, see
  above).

## Running without Docker

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), ffmpeg and a
whisper.cpp build (`brew install whisper-cpp` on macOS, which uses Metal; or
build it from source). `uv sync` installs yt-dlp with the other dependencies.

```bash
uv sync --extra mcp
cp example.env .env                    # mail + summarizer settings, as for Docker
export PODCAST_DIGEST_DATA_DIR=$PWD/data PODCAST_DIGEST_CONFIG=$PWD/config/feeds.toml
export PODCAST_DIGEST_WHISPER_BIN=$(command -v whisper-cli)
export PODCAST_DIGEST_WHISPER_MODEL=/path/to/ggml-medium.en.bin
uv run podcast-digest run --dry-run    # one tick, no mail: downloads, transcribes, summarizes
uv run podcast-digest serve            # the scheduler
```

Models: `ggml-medium.en.bin` and friends from
[huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp).
Without uv, install the bundled transcriber first:
`pip install ./packages/local-transcriber '.[mcp]'`.

## CLI

```
podcast-digest serve                            # long-running scheduler (the container default)
podcast-digest run [--feed NAME] [--dry-run]    # one tick, then exit
podcast-digest check-config [PATH]              # validate feeds.toml, no side effects
podcast-digest --version
podcast-digest-mcp                              # the MCP server
```

`run --dry-run` is a real pipeline run (download, transcription, summary, state
in SQLite) that stops before mail and the webhook.

## Data

Under `PODCAST_DIGEST_DATA_DIR`:

- `state.db` — SQLite: feeds, episodes and their stage, terminal outcomes,
  transcript fingerprints
- `transcripts/<feed>--<hash>/<episode>--<hash>.{txt,json,manifest.json,clean.md}`
  — kept (`tmp/` holds work in progress)
- `audio/` — deleted once an episode is sent or retired
- `models/` — whisper models and OpenVINO encoders

## Development

```bash
./scripts/check        # or: make check
```

The gate: frozen dependencies, Ruff, mypy, the whole pytest suite, and
validation of the example `feeds.toml`. External services (feeds, CDNs,
whisper, LLMs, mail, SSH, webhooks) are mocked; the integration tests replay the
whole pipeline against a fixture feed and assert both what is sent and what is
persisted. `packages/local-transcriber` is the reusable "local file → whisper.cpp
→ timestamped transcript" part, with its own CLI (`local-transcribe`).

Images: `deploy/Dockerfile` builds the transcriber (`--target
transcriber-runtime`) and the app (`--target podcast-runtime`);
`deploy/Dockerfile.podcast` builds the app on top of an already-built
transcriber image (the release workflow does this); `deploy/Dockerfile.mcp`
builds the MCP server.

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
Report security problems privately, as described in [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE). Third-party notices: [NOTICE](NOTICE).
