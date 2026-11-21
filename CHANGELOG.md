# Changelog

## [1.4.2] - 2026-11-21

### Fixed

- Rename a local variable in `tests/test_rich_delivery_contract.py` so public mirror export checks pass cleanly.

### Upgrade

- Action required: none.
- Rollback: compatible with 1.4.1.

## [1.4.1] - 2026-11-21

### Fixed

- Sanitize a test fixture label and allowlist two synthetic API-key test assertions for clean public mirror export.

### Upgrade

- Action required: none.
- Rollback: compatible with 1.4.0.

## [1.4.0] - 2026-11-21

### Added

- Redesign digest emails with responsive, dark-mode-safe typography, clean header/metadata hierarchy, clickable YouTube `[hh:mm:ss]` highlight deep links, and `<style>` stripping in plain-text alternatives while preserving the composing-build footer contract.
- Add read-only archive and status tools to `podcast-digest-mcp` (`digest_status`, `search_transcripts`, `get_episode`) backed by read-only SQLite WAL snapshotting and safe transcript path resolution, and hide completed one-offs from `subscription_list` by default.
- Support `*_FILE` secret loading in `Settings` (`ANTHROPIC_API_KEY_FILE`, `OPENAI_API_KEY_FILE`, `PODCAST_DIGEST_SMTP_PASSWORD_FILE`, `PODCAST_DIGEST_GRAPH_CLIENT_SECRET_FILE`, `PODCAST_DIGEST_WEBHOOK_TOKEN_FILE`).
- Make steady-state `max_new_per_tick` configurable globally and per feed in `feeds.toml` (first-run cap remains 1).
- Prune base64 attachment payloads from `digest_deliveries.envelope` once a delivery is accepted and finalized (and idempotently on open for previously finalized rows), keeping attachment metadata without duplicating transcripts in SQLite.

### Upgrade

- Action required: none. Optionally mount `/data` read-only and set `PODCAST_DIGEST_DATA_DIR=/data` on `podcast-digest-mcp` to enable `digest_status`, `search_transcripts`, and `get_episode`.
- Rollback: compatible with 1.3.0. Existing databases, delivery journals and transcripts remain readable.

## [1.3.0] - 2026-11-21

### Added

- Offload English Parakeet decoding to an optional SSH worker. Fall back to local Parakeet on failure and record the worker's source revision and model digest on success.

### Upgrade

- Action required: none. Set `PODCAST_DIGEST_PARAKEET_WORKER_SSH_CONFIG` to opt in after provisioning a worker.
- Rollback: compatible with 1.2.1. Existing databases, delivery journals and transcripts remain readable.

## [1.2.1] - 2026-11-20

### Fixed

- Pin validated public IP addresses at TCP connect time in `httpx` transports and per-hop in `curl --resolve`, closing the DNS rebinding bypass against the private-address SSRF guard (#15).
- Preserve Rich console capture, quiet suppression, recording, and live render hooks when attributing logs in standalone `local-transcriber` (#13).
- Refresh rotating RSS enclosure URLs and missing durations on retry for pending episodes that have not yet been transcribed.
- Add Cantonese (`yue`) display name and Chinese numeric-unit guidance, and weight non-ASCII characters when chunking transcripts for cleanup so dense CJK episodes stay within output token budgets.
- Align `README.md`, `example.env`, and `CHANGELOG.md` with Parakeet/Qwen transcription and `delivery` CLI commands.

### Upgrade

- Action required: none.
- Rollback: compatible with 1.2.0. The database, delivery journal and transcript artifact formats retain their contracts.

## [1.2.0] - 2026-11-19

### Added

- Support the offline Qwen3-ASR 0.6B INT8 model for Chinese audio, with Whisper fallback and chunk timestamps in the existing transcript attachments.
- Verify the pinned Qwen model and tokenizer assets on startup and record the engine and model bundle in the existing provenance manifest.

### Upgrade

- Action required: set `PODCAST_DIGEST_QWEN_MODEL_DIR` to enable Qwen. Existing Whisper models remain available for fallback.
- Rollback: compatible with 1.1.0. The database, delivery journal and transcript artifact formats retain their contracts.

## [1.1.0] - 2026-11-18

### Added

- Offer local Parakeet Unified English 0.6B INT8 with Whisper fallback for English podcast audio. Keep publisher transcript priority, multilingual Whisper and the existing timestamped artifacts.
- Verify pinned model assets on startup and record the transcription engine and model bundle in the existing provenance manifest.

### Upgrade

- Action required: opt in by setting `PODCAST_DIGEST_PARAKEET_MODEL_DIR`. The engine image prepares the weights on first start and uses Whisper if preparation or inference fails.
- Rollback: compatible with 1.0.0. The database, delivery journal and timestamped transcript formats retain their contracts.

## [1.0.0] - 2026-11-08

### Added

- Record an immutable delivery intent before mail submission and provide proof-bound recovery for accepted Graph messages. Uncertain sends remain held for investigation rather than being sent again.

- Embed the worker, MCP service and separately versioned transcriber's product version and full source revision in their installed artifacts, and include that identity with an opaque process instance in retained diagnostics.
- Stamp noneditable wheels and sdists built from a clean Git checkout, including Git-URL installs of `local-transcriber`, with that commit; the transcriber images keep exporting `LOCAL_TRANSCRIBER_SOURCE_REV`.
- Verify installed identity and the native MCP product version in exact-image smoke checks; require validated product versions and source revision in all image build paths.
- End each digest email with the product, version and source revision of the build that composed its summary. A later build that retries the send keeps that label. Summaries stored before this release are sent without one.

### Fixed

- Preserve summarized episode progress when transcript files disappear, and finish accepted publication and cleanup after interruptions without another send.

- Keep HTTP access diagnostics on stdout and startup diagnostics on stderr.
- Refuse source builds whose sparse checkout or index flags hide package inputs from verification.

- Preserve enclosing Rich console capture and recording while attributing diagnostics, and retain native sink-error handling and terminal rendering.

### Upgrade

- Action required: stop every worker sharing the database and preserve a consistent backup before the first startup. Existing subscriptions, episode watermarks, shared delivery history and artifacts remain in place. Investigate legacy summarized rows before enabling delivery: earlier builds did not retain submission intents and cannot prove whether those messages were accepted. Never reset or resend an uncertain delivery.
- Rollback: earlier workers do not understand the delivery journal. After the new worker records an operation, keep the compatible worker and recover or fix forward; restoring source alone does not restore delivery safety.

## [0.4.7] - 2026-11-01

### Fixed

- Bind each published runtime image to its source repository, full revision and release version, rather than retaining inherited base-image metadata. Exact-image smoke checks reject mismatched provenance before promotion (#7).

### Upgrade

- Action required: none.
- Rollback: compatible with 0.4.6. New consumers requiring full OCI provenance must retain 0.4.7 or newer.

[1.4.0]: https://github.com/benjamin-di/podcast-digest/releases/tag/v1.4.0
[1.3.0]: https://github.com/benjamin-di/podcast-digest/releases/tag/v1.3.0
[1.2.1]: https://github.com/benjamin-di/podcast-digest/releases/tag/v1.2.1
[1.2.0]: https://github.com/benjamin-di/podcast-digest/releases/tag/v1.2.0
[1.1.0]: https://github.com/benjamin-di/podcast-digest/releases/tag/v1.1.0
[1.0.0]: https://github.com/benjamin-di/podcast-digest/releases/tag/v1.0.0
[0.4.7]: https://github.com/benjamin-di/podcast-digest/releases/tag/v0.4.7
