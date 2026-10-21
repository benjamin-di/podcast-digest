# Contributing

Issues and pull requests are welcome: bug reports, fixes, new feed or
summarizer backends, documentation.

## Before you open a pull request

```bash
./scripts/check        # or: make check — frozen dependencies, Ruff, mypy, tests, example config
```

The gate needs [uv](https://docs.astral.sh/uv/) and no network, credentials or
external tools: feeds, CDNs, whisper.cpp, LLMs, mail, SSH and webhooks are all
mocked. Please:

- add or update tests for behaviour you change — the pipeline tests assert both
  what is sent and what is persisted in SQLite;
- keep test data fictional: invented shows, titles, IDs, `example.com`
  addresses — never copy from a real digest, mailbox or subscription list;
- update README.md / docs/ when you change a setting or a behaviour;
- if you touch `deploy/`, say whether you built and ran the image.

[AGENTS.md](AGENTS.md) describes the code layout and the invariants that matter
(the resumable state machine, deduplication, delivery semantics).

## How changes land

Each release of this repository is exported from a private working repository as one
`Release X.Y.Z` commit, so pull requests are not merged here: the next release would replace
them. When a change fits, the maintainer applies it there, credits you in the release notes
and closes the pull request with a link to the release that contains it. Issues are welcome;
answers are best effort.

## Reporting bugs

Include the version (`podcast-digest --version`, or the image tag), the
summarizer and mailer backends, and the relevant log lines (one JSON object per line). Remove addresses, tokens and
anything else private from logs and configuration before posting.

Security problems: see [SECURITY.md](SECURITY.md) instead of opening an issue.
