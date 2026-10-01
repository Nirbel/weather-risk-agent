"""Shared async HTTP helper: one client, timeouts, bounded retries on 429/5xx."""

import asyncio
from typing import Any

import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504}


class SourceError(Exception):
    """A data source failed after retries. The message is safe to show to analysts."""

    def __init__(self, source: str, message: str):
        super().__init__(f"{source}: {message}")
        self.source = source
        self.message = message


def make_client(contact_email: str, timeout: float = 30.0,
                transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    """`transport` lets the online eval inject network failures; the app uses the default."""
    return httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": f"(weather-risk-agent, {contact_email})"},
        follow_redirects=True,
        transport=transport,
    )


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    source: str,
    params: dict[str, Any] | None = None,
    retries: int = 2,
    backoff_s: float = 1.0,
) -> Any:
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = await client.get(url, params=params)
        except httpx.TransportError as exc:
            last_error = f"network error: {exc.__class__.__name__}"
        else:
            if response.status_code in RETRY_STATUSES:
                last_error = f"HTTP {response.status_code}"
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    backoff_s = max(backoff_s, min(float(retry_after), 30.0))
            elif response.status_code >= 400:
                raise SourceError(source, f"HTTP {response.status_code}: {response.text[:200]}")
            else:
                try:
                    return response.json()
                except ValueError as exc:
                    raise SourceError(source, "response was not valid JSON") from exc
        if attempt < retries:
            await asyncio.sleep(backoff_s * 2**attempt)
    raise SourceError(source, f"failed after {retries + 1} attempts ({last_error})")
