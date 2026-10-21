"""iTunes Lookup helpers — resolve Apple Podcasts URLs to RSS.

Public, key-free API; the same surface the digest already uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

LOOKUP_URL = "https://itunes.apple.com/lookup"


@dataclass(frozen=True)
class ApplePodcast:
    collection_id: str
    name: str
    rss_url: str


@dataclass(frozen=True)
class ApplePodcastEpisode:
    track_id: str
    title: str
    audio_url: str
    duration_seconds: int | None
    published_at: str | None
    collection_name: str


async def lookup_episode(
    collection_id: str, episode_track_id: str, *, http: httpx.AsyncClient
) -> ApplePodcastEpisode:
    """Resolve an Apple Podcasts episode deep-link to its direct enclosure.

    An episode trackId can't be looked up directly (the Lookup API returns 0
    results for it), so we enumerate the collection's recent episodes via
    `entity=podcastEpisode` and match by trackId. Episodes older than the lookup
    window won't be found — we raise with a "subscribe instead" hint.
    """
    r = await http.get(
        LOOKUP_URL,
        params={"id": collection_id, "entity": "podcastEpisode", "limit": 200},
    )
    r.raise_for_status()
    want = str(episode_track_id)
    for result in r.json().get("results") or []:
        if str(result.get("trackId")) != want:
            continue
        url = result.get("episodeUrl")
        if not url:
            raise ValueError(f"Apple episode {want} has no episodeUrl")
        ms = result.get("trackTimeMillis")
        return ApplePodcastEpisode(
            track_id=want,
            title=str(result.get("trackName") or "Episode"),
            audio_url=str(url),
            duration_seconds=int(ms) // 1000 if ms else None,
            published_at=result.get("releaseDate"),
            collection_name=str(result.get("collectionName") or "Podcast"),
        )
    raise ValueError(
        f"could not resolve Apple episode {want} in podcast {collection_id} — it "
        "may be older than the lookup window; subscribe to the podcast instead."
    )


async def lookup_podcast(collection_id: str, *, http: httpx.AsyncClient) -> ApplePodcast:
    r = await http.get(LOOKUP_URL, params={"id": collection_id, "entity": "podcast"})
    r.raise_for_status()
    data = r.json()
    for result in data.get("results") or []:
        feed = result.get("feedUrl")
        if feed:
            return ApplePodcast(
                collection_id=str(collection_id),
                name=str(result.get("collectionName") or "Podcast"),
                rss_url=str(feed),
            )
    raise ValueError(f"iTunes Lookup returned no feedUrl for collectionId={collection_id}")
