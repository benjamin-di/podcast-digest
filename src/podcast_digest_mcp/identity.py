"""Identity of the MCP service in the shared application distribution."""
from pathlib import Path

from podcast_digest.identity import load_build

BUILD = load_build(Path(__file__).with_name("build_identity.json"), service="podcast-digest-mcp")
