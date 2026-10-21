"""MCP server for podcast-digest.

A thin FastMCP wrapper that lets mobile clients (Claude iPhone app, claude.ai)
add subscriptions and one-off YouTube videos by appending entries to the same
`feeds.toml` the digest container reads. The digest hot-reloads `feeds.toml` on
every poll tick, so additions take effect on the next tick with no restart.

Why thin: the digest pipeline already handles every source type natively
(`rss_url`, `search`, `youtube`, `youtube_video`). The MCP doesn't download,
transcribe, or summarize — it only writes the TOML entry. All the heavy
lifting stays in the existing pipeline.
"""
