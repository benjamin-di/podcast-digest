# Delivery webhook

After an episode's digest email is delivered, podcast-digest can POST the
episode — metadata, the summary and the transcript — to a URL of your choice:
a knowledge-base ingest route, an automation tool (n8n, Zapier, Make), a notes
service, your own script. The email stays the primary outcome: a webhook
outage never blocks, delays or rolls back the digest.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `PODCAST_DIGEST_WEBHOOK_URL` | _(empty = off)_ | Where to POST |
| `PODCAST_DIGEST_WEBHOOK_TOKEN` | _(empty)_ | Sent as `Authorization: Bearer …` when set; never logged |
| `PODCAST_DIGEST_WEBHOOK_FIELDS` | `{}` | JSON object of static fields added to every payload, e.g. `{"collection": "podcasts"}` |
| `PODCAST_DIGEST_WEBHOOK_ACK_FIELD` | _(empty)_ | Require this key in the JSON response (see below) |
| `PODCAST_DIGEST_WEBHOOK_TIMEOUT` | `30` | Minimum POST timeout in seconds; scales up with the payload |

Every feed participates unless it sets `webhook = false` in `feeds.toml`
(for example a feed you only want by email).

## Payload

`POST` with `Content-Type: application/json`:

```json
{
  "show": "Show name",
  "title": "Episode title",
  "published_at": "2026-06-18T00:00:00+00:00",
  "guid": "the feed item's GUID",
  "listen_url": "https://cdn.example.com/episode.mp3",
  "summary": "<h2>TL;DR</h2>…",
  "transcript": "[00:00:00] …"
}
```

plus any `PODCAST_DIGEST_WEBHOOK_FIELDS` (the episode fields win on a name
clash). `summary` is the sanitized HTML that was emailed. `transcript` is the
cleaned Markdown transcript when one exists (`PODCAST_DIGEST_ATTACH_TRANSCRIPT=cleaned`),
otherwise the raw whisper text. `listen_url` is empty unless it is http(s).

## Acknowledgement and retries

- Without `PODCAST_DIGEST_WEBHOOK_ACK_FIELD`, any 2xx response counts as
  delivered.
- With it set (say `task_id`), a delivery counts only when the response is a
  JSON object carrying a non-empty value under that key. Use this when the
  receiver queues slow work: a proxy's generic success page or a truncated
  response then never counts as delivery.

A delivered episode is recorded in SQLite and never posted again. A failure
(HTTP error, network error, missing acknowledgement) leaves it pending and is
retried on later ticks, up to `WIKI_PUSH_MAX_ATTEMPTS` (5, in `db.py`; the
webhook state lives in the episodes table's `wiki_*` columns). Exhausted
attempts stay visible in SQLite.

The receiver should answer quickly and do slow work asynchronously. A timeout
after the receiver accepted the POST means it is posted again on the next tick,
so make the receiver idempotent on `guid`.

Turning the webhook on never posts history: only episodes emailed after the
first poll with a URL configured are sent. Turning it off and on again starts
afresh from that moment, so episodes emailed while it was off are not posted.

## Verification

- `uv run pytest tests/test_webhook.py tests/test_pipeline_webhook.py -q`
  exercises the request shape, acknowledgements, retries, feed opt-out,
  transcript selection and the SQLite watermark with HTTP mocked.
- A live check against your receiver is separate: it needs its credentials and
  usually mutates its data.
