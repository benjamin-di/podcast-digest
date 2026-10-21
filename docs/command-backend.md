# Command summarizer backend

With `PODCAST_DIGEST_SUMMARIZER=command`, every LLM call (the summary, and each
chunk of the transcript cleanup) runs `PODCAST_DIGEST_COMMAND` once, with the
prompt on stdin. Use it for:

- a CLI on a subscription instead of a metered API (`claude -p`, `codex exec -`);
- a local model runner (`ollama run llama3.1:8b`, `llm -m …`);
- a model host elsewhere: set `PODCAST_DIGEST_SSH_HOST` and the command runs
  there over SSH, so the transcription box never holds model credentials.

The command string is split shell-style (`shlex`) when run locally and passed
as-is to the remote shell over SSH.

## Settings

| Env var | Default | Purpose |
|---|---|---|
| `PODCAST_DIGEST_COMMAND` | _(required)_ | The command to run |
| `PODCAST_DIGEST_COMMAND_PROTOCOL` | `text` | `text` or `json` (below) |
| `PODCAST_DIGEST_COMMAND_TIMEOUT` | `300` | Seconds per call (+60 s grace) |
| `PODCAST_DIGEST_SSH_HOST` | _(empty = local)_ | Run the command on this host |
| `PODCAST_DIGEST_SSH_PORT` | `22` | |
| `PODCAST_DIGEST_SSH_USER` | _(ssh default)_ | |
| `PODCAST_DIGEST_SSH_KEY` | _(ssh default)_ | Private key path; or `PODCAST_DIGEST_SSH_KEY_B64` (base64) in the container |
| `PODCAST_DIGEST_SSH_KNOWN_HOSTS` | _(empty = trust on first use)_ | Pinned known_hosts file; or put its content in `PODCAST_DIGEST_SSH_KNOWN_HOSTS_B64` (base64) and the container writes it |
| `PODCAST_DIGEST_TIER` | _(empty)_ | `json` protocol only: passed as `tier` |
| `PODCAST_DIGEST_MODEL` | | `json` protocol only: passed as `model` when you set it explicitly |

## `text` protocol

stdin is the system prompt, a `---` separator line, and the user prompt. stdout
is the answer. A non-zero exit, a timeout or empty output is a backend failure;
it is retried twice with backoff, then the episode waits for the next tick.

## `json` protocol

For a runner that wants structure — provider fallback, machine-readable errors.

stdin is one JSON object:

```json
{
  "prompt": "…",
  "system_prompt": "…",
  "tools": "",
  "cwd": "/tmp",
  "source": "podcast-digest",
  "timeout_seconds": 300,
  "stdout_tail_limit": 64000,
  "tier": "…",
  "model": "…"
}
```

`tools: ""` asks the runner to disable any tool use (these are pure text
transforms). `model` is a request; a runner that picks models per provider or
tier may map or ignore it. `source` is `podcast-digest` for the summary and
`podcast-digest-clean` for cleanup chunks. `tier` and `model` are present only
when configured.

stdout is newline-delimited JSON events. podcast-digest reads two kinds and
ignores the rest:

- `{"type": "result", "is_error": false, "text": "…"}` — an answer. With
  provider fallback there may be several; the last non-empty, non-error one
  wins, so a failed first provider rescued by a second still succeeds. If every
  `result` is `is_error: true`, the call fails (and is retried).
- `{"type": "run_failed", "reason": "…", "auth_failure": false}` — the run
  failed. `auth_failure: true`, or a `reason` of `auth_failure`,
  `config_error` or `geo_blocked`, is treated as terminal (no retries this
  tick); anything else is retried.

## Failure handling

Backend failures (process errors, timeouts, SSH failures, `run_failed`) are
counted separately from episode failures: after ten, an affected episode gets
one canary attempt per poll, and only after 30 is it dropped (about a week at
the default poll). The heartbeat (`PODCAST_DIGEST_HEARTBEAT_URL`) stays silent
until a call succeeds.
A reply that is not the expected HTML (for example an agent that wrote the
answer to a file and replied with its path) is retried and never emailed.
