#!/usr/bin/env bash
# Promote the tested manifest, preserving any attestation index and refusing tag drift.
set -euo pipefail
repository="${1:?repository required}"
digest="${2:?tested digest required}"
version="${3:?version required}"
[[ "$digest" =~ ^sha256:[a-f0-9]{64}$ ]] || exit 2
[[ "$version" =~ ^([0-9]+\.[0-9]+\.[0-9]+|sha-[a-f0-9]{7,40})$ ]] || exit 2
release="$repository:$version"
error_file="$(mktemp)"
trap 'rm -f "$error_file"' EXIT
manifest_digest() {
    docker buildx imagetools inspect "$1" --format '{{json .Manifest}}' | jq -er '.digest'
}
# Resolve the tested digest before touching any release ref.
[ "$(manifest_digest "$repository@$digest")" = "$digest" ]
if current="$(manifest_digest "$release" 2>"$error_file")"; then
    [ "$current" = "$digest" ] || { echo "immutable ref conflict: $release" >&2; exit 1; }
else
    # An authentication/network failure is not evidence that the tag is absent.
    grep -Eqi 'manifest unknown|not found' "$error_file" || { cat "$error_file" >&2; exit 1; }
    docker buildx imagetools create --prefer-index=false --tag "$release" "$repository@$digest"
fi
[ "$(manifest_digest "$release")" = "$digest" ]
printf 'verified %s@%s\n' "$release" "$digest"
