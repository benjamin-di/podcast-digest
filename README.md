# podcast-digest

Follow podcasts and YouTube channels, transcribe every new episode locally with
Parakeet, Qwen3-ASR, or [whisper.cpp](https://github.com/ggml-org/whisper.cpp),
and get an email per episode with an LLM-written summary — TL;DR, speakers, key
takeaways, claims, timestamped highlights, "worth listening in full?" — and
optionally a cleaned-up transcript attached.

- **Sources:** podcast RSS feeds, a podcast name (resolved via the iTunes
  Search API), Apple Podcasts links, YouTube channels, single YouTube videos and
  one-off audio URLs.
- **Publisher transcripts first:** when a feed links the episode's own
  transcript (Podcasting 2.0 `<podcast:transcript>`), it is used as is — no
  download, no transcription, speaker names included.
- **Local transcription otherwise:** optional Parakeet Unified English INT8 on
  the CPU for English audio and Qwen3-ASR 0.6B INT8 on the CPU for Chinese
  (`zh`/`yue`), each with whisper.cpp fallback. Whisper uses OpenVINO on a
  supported Intel iGPU or runs on the CPU, and handles all other languages.
  Non-English feeds still get an English summary.
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
export VERSION="$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
export TRANSCRIBER_VERSION="$(python3 -c 'import tomllib; print(tomllib.load(open("packages/local-transcriber/pyproject.toml", "rb"))["project"]["version"])')"
export VCS_REF="$(git rev-parse HEAD)"
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
max_new_per_tick = 1            # steady-state cap per feed per tick (first run stays 1)

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

Each digest commits a stable operation ID, transport target and complete envelope
(recipient, subject, HTML and attachment bytes) to SQLite before submission.
The key is correlation, not provider deduplication. A lost verdict, process crash
or storage failure after possible submission leaves an `unknown` operation held
at `summarized`: no automatic resend, `emailed` watermark, webhook or sent cleanup.
Only a proven pre-submission failure or explicit rejection permits a later attempt
with the same saved envelope. Graph 5xx responses and socket closure are uncertain.
A safe retry still checks positive GUID and transcript delivery evidence. A match
suppresses the retry while preserving its non-submission outcome and saved envelope;
it does not manufacture an acceptance receipt. Unknown sends remain held even when
another delivery matches.
A crash immediately before sending also stays held; a local journal cannot prove
which side of submission a crashed process reached.

Inspect and reconcile under the same single-writer lock used by ticks:

```bash
uv run podcast-digest delivery list
uv run podcast-digest delivery inspect OPERATION_ID
uv run podcast-digest delivery reconcile OPERATION_ID --graph-message-id MESSAGE_ID
```

`inspect` includes the saved message and attachments; treat its output as private.
Find the candidate message in the sender mailbox's Sent Items through an authorized
mail client or Graph read tool, then pass its Graph message ID. Reconciliation reads
that item through Graph and requires a matching `x-podcast-digest-operation` header,
sender, recipients, subject and positive sent metadata. It records the receipt and
`emailed` watermark in one transaction. It never sends mail. This read requires
`Mail.Read` on the sender mailbox in addition to `Mail.Send`; lack of permission,
a missing/moved item or an observation error leaves the operation held. A missing
receipt, expired lock or old operation is never proof of non-submission.

Ordinary acceptance and positive reconciliation both publish the shared delivery
log and clean up sent audio. These steps are journaled separately from acceptance:
if interrupted or unsuccessful, the next real tick retries them without sending
mail, even if the feed has been removed from configuration. A successful recovery
command also attempts pending finalization under the writer lock. The shared log
is a cross-instance hint, not a distributed lock; independent instances can still
race before either publishes its accepted delivery.

SMTP carries the same operation header and a stable Message-ID, but has no supported
Sent Items API here. Its unknown operations remain held for provider investigation;
there is deliberately no force-retry or unchecked mark-sent command. Keep the state
database and artifacts when investigating. Do not delete unknown rows or clear their
state to retry. Restarted ticks report the operation ID and inspection command.
Keep the original transport target configured to retry a proven non-submission.
A changed mailbox/backend cannot take over an existing operation.

### Summarizer

| Variable | Default | |
|---|---|---|
| `PODCAST_DIGEST_SUMMARIZER` | `api` | `api` or `command` |
| `PODCAST_DIGEST_MODEL` | `claude-opus-5-5` | `api`: the model. `claude-*` uses Anthropic (`claude-sonnet-5-5` is cheaper), anything else OpenAI or a compatible server |
| `PODCAST_DIGEST_PROVIDER` | `auto` | Force `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | | The provider's key (or `*_FILE` to read from a mounted secret file; `PODCAST_DIGEST_SMTP_PASSWORD_FILE`, `PODCAST_DIGEST_GRAPH_CLIENT_SECRET_FILE`, and `PODCAST_DIGEST_WEBHOOK_TOKEN_FILE` work the same way) |
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
| `PODCAST_DIGEST_WHISPER_MODEL` | `/data/models/ggml-medium.en.bin` | Default Whisper model, including English fallback |
| `PODCAST_DIGEST_WHISPER_MODEL_MULTILINGUAL` | `/data/models/ggml-large-v3-turbo.bin` | For feeds with a `language` |
| `PODCAST_DIGEST_WHISPER_LANGUAGE` | `en` | Language passed to whisper for feeds without a `language` |
| `PODCAST_DIGEST_WHISPER_THREADS` | `8` | CPU threads |
| `PODCAST_DIGEST_PARAKEET_MODEL_DIR` | unset | Enable Parakeet for English and cache the pinned INT8 ONNX assets in this directory |
| `PODCAST_DIGEST_PARAKEET_WORKER_SSH_CONFIG` | unset | Optional OpenSSH config for the English decoder, using the host alias `parakeet-worker` |
| `PODCAST_DIGEST_PARAKEET_THREADS` | `4` | Parakeet CPU threads |
| `PODCAST_DIGEST_QWEN_MODEL_DIR` | unset | Enable Qwen3-ASR 0.6B for Chinese and cache the pinned INT8 ONNX assets in this directory |
| `PODCAST_DIGEST_QWEN_THREADS` | `4` | Qwen CPU threads |
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

## English transcription with Parakeet

Set `PODCAST_DIGEST_PARAKEET_MODEL_DIR` to a directory on the data volume,
such as `/data/models/parakeet-unified-en-0.6b-int8`. The engine image includes
the CPU runtime. It downloads the pinned INT8 ONNX export on first start and
verifies the cached assets on later starts. The offline worker splits long
audio near quiet boundaries around 30 seconds and writes the same timestamped
text and JSON artifacts used by the rest of the digest. Publisher transcripts
retain priority. A missing runtime, unavailable model, inference error, or
empty result falls back to Whisper. Unset the directory to select Whisper
directly. Keep the intended Whisper models available in either mode.

An enabled English model can also use a native SSH worker. Set
`PODCAST_DIGEST_PARAKEET_WORKER_SSH_CONFIG` to a read-only SSH configuration
with a pinned host key and a dedicated identity for the alias `parakeet-worker`.
Restrict that identity to a forced command running
`python -m local_transcriber.worker --model-dir MODEL_DIR --lock-path LOCK_FILE`
from an installed, revision-stamped wheel with the Parakeet dependencies.
The worker takes one normalized WAV at a time, uses four CPU threads by default,
and discards temporary audio when the request finishes or disconnects. It returns
timestamped segments and the decoder's version, model digest and source revision.
A busy or unavailable worker or invalid reply retries the same WAV with local
Parakeet. The existing Whisper fallback still applies after local Parakeet failure.

For a source install, use `pip install ./packages/local-transcriber '.[parakeet]'`
and run `deploy/prepare-onnx-model.sh parakeet <model-dir>`. The
[NVIDIA model card](https://huggingface.co/nvidia/parakeet-unified-en-0.6b)
links to the NVIDIA Open Model License for the weights.

## Chinese transcription with Qwen

Set `PODCAST_DIGEST_QWEN_MODEL_DIR` to a persistent directory such as
`/data/models/qwen3-asr-0.6b-int8`, and set the feed language to `zh` for
Mandarin or `yue` for Cantonese. Publisher transcripts still take priority.
The engine verifies the pinned CPU model on startup and falls back to
multilingual Whisper if the model is unavailable or inference fails.
Qwen transcript timestamps mark audio chunks of roughly 20 seconds.

For a source install, use the same `parakeet` extra for the shared runtime
and run `deploy/prepare-onnx-model.sh qwen <model-dir>`.
[Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)

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
| `subscription_list(include_completed_oneoffs?)` / `subscription_remove(name)` | List or remove entries (hides completed one-offs by default when `PODCAST_DIGEST_DATA_DIR` is mounted) |
| `process_now()` | Run a tick now |
| `digest_status(limit?)` | Read-only pipeline status: stage counts, recent deliveries, in-progress episodes, held deliveries, and feed errors |
| `search_transcripts(query, feed?, limit?)` | Search saved summaries and transcripts by keyword(s) |
| `get_episode(query_or_guid, include_transcript?, max_chars?, offset?)` | Read an episode's summary and paginated transcript (`.clean.md` preferred, `.txt` fallback) |

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
`pip install -e ./packages/local-transcriber -e '.[mcp]'`.

## CLI

```
podcast-digest serve                            # long-running scheduler (the container default)
podcast-digest run [--feed NAME] [--dry-run]    # one tick, then exit
podcast-digest check-config [PATH]              # validate feeds.toml, no side effects
podcast-digest delivery list                    # list saved delivery operations and outcomes
podcast-digest delivery inspect OPERATION_ID    # show a saved delivery envelope and state
podcast-digest delivery reconcile OPERATION_ID --graph-message-id MESSAGE_ID
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
persisted. `packages/local-transcriber` is the reusable local audio to timestamped
transcript boundary, with its own Whisper CLI (`local-transcribe`).

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

Published runtime candidates carry OCI source, full revision and release version
labels, and the image smoke verifies all three before promotion. When running
`scripts/smoke-image.sh` directly, set `GITHUB_REPOSITORY` to the source repository
whose artifact is being verified. This check performs no feed, mail or model work.

Runtime identity is embedded before packaging, independently of environment or
container labels. The worker (`podcast-digest`) and MCP service
(`podcast-digest-mcp`) use the root distribution version; `local-transcriber`
uses its own package version. Their log records include `service`, `version`,
`source_revision` (the full revision) and an opaque process `instance`. Existing
structured fields, redaction and log suppression remain in effect. Successful
CLI stdout (including the transcription artifact) stays unchanged. The MCP
initialize response reports the installed product version.

Each digest email ends with `Generated by podcast-digest <version>, source
revision <full revision|unknown>.` This identifies the worker build that authored
the stored summary. Its identity is saved in the nullable `episodes.summary_build`
column together with `summary_html`. A later build retrying delivery retains that
identity. Summaries stored before this column was added keep NULL and are sent
without an invented provenance line.

The delivery-time wrapper supplies the title, episode metadata, listening link
and transcript-source note. They are frozen with the first delivery intent;
retries reuse that saved envelope and its attachments. They are outside the
summary producer's identity. The email footer does not name
a separate delivery build. The stored summary and webhook `summary` contain no
build footer.

Image builds require `VERSION` and `VCS_REF` (40 or 64 hexadecimal characters).
`deploy/Dockerfile`'s `transcriber-runtime` target takes the transcriber product
`VERSION`; its alternate `podcast-runtime` target instead takes the root `VERSION`
and a separate `TRANSCRIBER_VERSION`. The MCP Dockerfile also requires both
product versions. The derived podcast Dockerfile stamps only the root product,
retaining the transcriber stamp from its input digest. Invalid or conflicting
inputs fail before installation. The three image tags and OCI version labels
continue to use the root release-wave version.

Noneditable wheels and sdists built from a clean Git checkout embed that
commit through the package's
`hatch_build.py`. A wheel built from an sdist keeps the stamp the sdist ships.
Uncommitted changes and index-hidden inputs (sparse checkouts, skip-worktree or
assume-unchanged) are refused. Without Git metadata, as in the Docker build
contexts, first run `python3 scripts/embed-build-identity.py podcast-digest
"$VERSION" "$VCS_REF"`, or `local-transcriber` with its own `VERSION`; a build
with no stamp is refused. Generated manifests are excluded from Git and Docker
build contexts. A missing stamp is allowed only in an uninstalled source tree or
an explicitly editable install, where the revision is `unknown`; a noneditable
installed package with a missing or invalid stamp fails closed. The transcriber
images still export `LOCAL_TRANSCRIBER_SOURCE_REV` with the same revision for
downstream pin checks; it never changes the installed identity.

The exact-image smoke accepts `variant image full-revision image-release-version
transcriber-product-version`. It checks the actual installed stamps and the MCP
initialize version, without invoking providers, downloading models or transcribing.
