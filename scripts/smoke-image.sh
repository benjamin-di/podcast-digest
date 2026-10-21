#!/usr/bin/env bash
# Packaging verification only: no model downloads, feeds, mail or live state.
set -euo pipefail
variant="${1:?variant required}"
image="${2:?image required}"
revision="${3:?revision required}"
version="${4:?version required}"
docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
test "$(docker image inspect "$image" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')" = "$revision"
case "$variant" in
  transcriber|podcast)
    docker run --rm --network none --read-only --tmpfs /tmp --entrypoint /bin/bash \
      -e EXPECTED_REVISION="$revision" -e EXPECTED_VERSION="$version" \
      -e SMOKE_VARIANT="$variant" "$image" -euc '
        test "$LOCAL_TRANSCRIBER_SOURCE_REV" = "$EXPECTED_REVISION"
        python -c "import torch, openvino, local_transcriber"
        LD_LIBRARY_PATH="$WHISPER_LD_PATH" whisper-cli --help >/dev/null
        local-transcribe --help >/dev/null
        if [ "$SMOKE_VARIANT" = podcast ]; then
          python -c '\''import importlib.metadata, os; assert importlib.metadata.version("podcast-digest") == os.environ["EXPECTED_VERSION"]'\''
          podcast-digest check-config /app/feeds.toml
        fi
      '
    ;;
  mcp)
    cid="$(docker run -d --network none --read-only --tmpfs /tmp \
      -e PODCAST_MCP_BEARER_TOKEN=isolated-image-smoke "$image")"
    trap 'code=$?; docker logs --tail 40 "$cid"; docker rm -f "$cid" >/dev/null; exit "$code"' EXIT
    docker exec -e EXPECTED_VERSION="$version" "$cid" python -c \
      'import importlib.metadata, os; assert importlib.metadata.version("podcast-digest") == os.environ["EXPECTED_VERSION"]'
    docker exec "$cid" curl --fail --silent --show-error --retry 20 --retry-connrefused \
      --retry-delay 1 --max-time 2 http://127.0.0.1:8000/health
    # Unauthenticated MCP requests are refused (FastMCP bearer verifier). A
    # stateless server routes only POST on /mcp, so probe with a real request.
    test "$(docker exec "$cid" curl --max-time 5 -s -o /dev/null -w '%{http_code}' \
      -X POST -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
      --data '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"image-smoke","version":"1"}}}' \
      http://127.0.0.1:8000/mcp)" = 401
    docker exec -i "$cid" python - <<'PY'
import json
import urllib.request

headers = {"Authorization": "Bearer isolated-image-smoke",
           "Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
payload = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-03-26", "capabilities": {},
    "clientInfo": {"name": "image-smoke", "version": "1"}}}
request = urllib.request.Request("http://127.0.0.1:8000/mcp",
                                 data=json.dumps(payload).encode(), headers=headers)
with urllib.request.urlopen(request, timeout=10) as response:
    body = response.read().decode()
    content_type = response.headers.get("Content-Type", "")
    session = response.headers.get("Mcp-Session-Id")
# Stateless JSON transport: a plain JSON reply, no SSE stream, no session.
assert content_type.startswith("application/json"), content_type
assert session is None, "stateless server issued an Mcp-Session-Id"
message = json.loads(body)
assert message.get("id") == 1, message
result = message["result"]
assert result["serverInfo"]["name"] == "podcast-digest", result
assert "tools" in result["capabilities"], result
print("authenticated MCP initialize passed; no tools invoked")
PY
    ;;
  *) exit 2 ;;
esac
