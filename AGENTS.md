# Repository guide

## Product and code map

`podcast-digest` discovers podcast/YouTube episodes, downloads and transcribes
them locally, summarizes them with an LLM, emails the digest, and optionally
posts each delivered episode to a webhook. `podcast-digest-mcp` is the
configuration surface for chat clients; it edits the feed file and signals the
scheduler.

- `src/podcast_digest/pipeline.py`: end-to-end orchestration and user outcomes
- `src/podcast_digest/db.py`: SQLite schema and forward-only episode state
- `src/podcast_digest/{feeds,youtube,audio,transcribe}.py`: discovery and media
- `src/podcast_digest/publisher.py`: publisher transcripts (Podcasting 2.0),
  used instead of local transcription when plausible
- `src/podcast_digest/{llm,command,summarize}.py`: LLM backends (API, command)
  and the two LLM tasks (summary, transcript cleanup)
- `src/podcast_digest/{mail,webhook}.py`: delivery (SMTP / Graph, webhook)
- `src/podcast_digest_mcp/`: authenticated feed-management MCP; no processing
- `packages/local-transcriber/`: reusable local-file → whisper.cpp boundary
- `tests/test_pipeline_*.py`: mocked end-to-end state and delivery coverage
- `deploy/` and `.github/workflows/`: images, runtime entrypoint, CI, release

See [README.md](README.md) for configuration and operations,
[docs/command-backend.md](docs/command-backend.md) for the command protocol and
[docs/webhook.md](docs/webhook.md) for the webhook contract.

## Critical boundaries

- `feeds.toml` here is an example and the CI fixture. A deployment's live
  subscription list lives with that deployment; change it there (or through the
  MCP), never by shipping this file.
- Nothing about a particular deployment belongs in this repository: no
  hostnames, addresses, accounts, paths or service names of a real setup —
  configuration comes from the environment. Use `example.com` addresses and
  placeholder hosts in code, docs and tests.
- Test data, fixtures and examples are fictional: invented shows, titles,
  GUIDs, video IDs and addresses. Never copy them from a running digest, a
  mailbox, a subscription list or a log — not even "public" show names.
- Preserve the forward-only, resumable state machine and both GUID and content
  dedupe. Assert a user-visible effect and its SQLite watermark together.
- Email is the primary delivery outcome. The webhook is post-email,
  bounded and non-fatal; with an ack field configured only that value
  acknowledges delivery.
- A summarizer outage is a backend failure (`SummarizerUnavailable`), never an
  episode failure: keep that classification intact in new backends.
- Treat feeds, transcripts, model output, redirects, and URLs as untrusted.
- Keep podcast concerns out of `local-transcriber`; keep processing out of the
  MCP writer.
- Do not tag, release, merge, or deploy without explicit authorization.

## Completing changes

Start from an observable outcome and trace only its relevant interface,
pipeline, persistence, worker, and external-service path. Use focused tests
during development, then run:

```bash
./scripts/check
```

This is the shared local/CI gate: frozen dependencies, Ruff, mypy, all tests,
and validation of the example feed config. If `deploy/` changes, also exercise
the affected image or report it unverified; CI's Dockerfile lint is separate.
At handoff, state what user outcome and persisted state were exercised, which
external boundaries were mocked, and any live service or hardware path that was
not verified. Blocking defects prevent completion; optional cleanups do not.
