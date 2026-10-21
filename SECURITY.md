# Security

Please report vulnerabilities privately through GitHub: **Security → Report a
vulnerability** on this repository. Do not open a public issue for them.

Relevant areas include the handling of untrusted input (feeds, transcripts,
model output, redirects and URLs — outbound requests refuse private and local
network addresses), the MCP server's bearer authentication, and the command
summarizer backend (it runs the configured command with transcript text on
stdin).
