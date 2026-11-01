"""Cached, rate-limited client for the USPTO Open Data Portal (api.uspto.gov).

Covers the endpoints the pipeline needs:

- application metadata and the file-wrapper document list
- XML archives of individual file-wrapper documents (office actions, claims)
- per-document full-text XML of publications and grants (APPXML-SPLT / PTGRXML-SPLT)

Every response is cached on disk, so reruns don't hit the API again.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Any

import requests

API = "https://api.uspto.gov/api/v1"


def load_api_key(env_file: str | Path = ".env") -> str:
    """Read USPTO_API_KEY from the environment, falling back to a local .env file."""
    if key := os.environ.get("USPTO_API_KEY"):
        return key
    path = Path(env_file)
    if path.exists():
        for line in path.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "USPTO_API_KEY" and value.strip():
                return value.strip().strip("'\"")
    raise RuntimeError("USPTO_API_KEY is not set (environment or .env)")


class NotFound(Exception):
    pass


class ODPClient:
    def __init__(
        self,
        cache_dir: str | Path,
        api_key: str | None = None,
        min_interval: float = 0.35,
        max_retries: int = 5,
    ):
        self.cache_dir = Path(cache_dir)
        self.session = requests.Session()
        self.session.headers["X-API-KEY"] = api_key or load_api_key()
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0
        self.requests_made = 0

    # -- transport ----------------------------------------------------------

    def _cache_path(self, kind: str, key: str, suffix: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        return self.cache_dir / kind / f"{digest}{suffix}"

    def _get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response | None:
        """GET with throttling and retry on 429/5xx and network errors. Returns None on 404."""
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests_made += 1
            try:
                resp = self.session.get(url, params=params, timeout=(15, 90))
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(2**attempt)
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                # Cap the server's Retry-After so one throttled response can't stall a run.
                time.sleep(min(60.0, float(resp.headers.get("Retry-After", 2**attempt))))
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"Gave up on {url} after {self.max_retries} attempts")

    def _cached(self, kind: str, key: str, suffix: str, fetch) -> bytes | None:
        path = self._cache_path(kind, key, suffix)
        missing = path.with_suffix(path.suffix + ".404")
        if path.exists():
            return path.read_bytes()
        if missing.exists():
            return None
        resp = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        if resp is None:
            missing.touch()
            return None
        path.write_bytes(resp.content)
        return resp.content

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        key = path + "?" + json.dumps(params or {}, sort_keys=True)
        body = self._cached("json", key, ".json", lambda: self._get(f"{API}/{path}", params))
        return None if body is None else json.loads(body)

    def get_bytes(self, url: str) -> bytes | None:
        return self._cached("files", url, ".bin", lambda: self._get(url))

    # -- endpoints ----------------------------------------------------------

    def application(self, app_id: str) -> dict[str, Any] | None:
        data = self.get_json(f"patent/applications/{app_id}")
        return data["patentFileWrapperDataBag"][0] if data else None

    def documents(self, app_id: str) -> list[dict[str, Any]]:
        data = self.get_json(f"patent/applications/{app_id}/documents")
        return data.get("documentBag", []) if data else []

    def search_applications(self, query: str) -> list[dict[str, Any]]:
        data = self.get_json("patent/applications/search", {"q": query, "limit": 5})
        return data.get("patentFileWrapperDataBag", []) if data else []

    def document_xml(self, download_url: str) -> str | None:
        """Fetch a file-wrapper document's XML archive (a tar with one XML file)."""
        body = self.get_bytes(download_url)
        if body is None:
            return None
        with tarfile.open(fileobj=io.BytesIO(body)) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".xml")]
            if not members:
                return None
            return tar.extractfile(members[0]).read().decode("utf-8", errors="replace")

    def fulltext_xml(self, file_location_uri: str) -> str | None:
        """Fetch a publication or grant full-text XML (APPXML-SPLT / PTGRXML-SPLT)."""
        body = self.get_bytes(file_location_uri)
        return None if body is None else body.decode("utf-8", errors="replace")
