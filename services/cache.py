"""Small cache abstraction with a zero-dependency in-memory default."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any


class MemoryCache:
    def __init__(self) -> None:
        self._items: dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            item = self._items.get(key)
            if not item:
                return None
            expires, value = item
            if expires and expires < time.time():
                self._items.pop(key, None)
                return None
            return value

    async def setex(self, key: str, ttl: int, value: str) -> None:
        async with self._lock:
            self._items[key] = (time.time() + ttl if ttl else 0.0, value)

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class RedisCache:
    def __init__(self, client: Any) -> None:
        self.client = client

    async def get(self, key: str) -> str | None:
        return await self.client.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        await self.client.setex(key, ttl, value)

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def close(self) -> None:
        close = getattr(self.client, "aclose", None) or getattr(self.client, "close", None)
        if close:
            result = close()
            if hasattr(result, "__await__"):
                await result


async def create_cache(backend: str, redis_url: str):
    if backend == "memory":
        return MemoryCache(), "memory"
    try:
        import redis.asyncio as aioredis  # type: ignore
        client = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        await client.ping()
        return RedisCache(client), "redis"
    except Exception:
        if backend == "redis":
            raise
        return MemoryCache(), "memory"


async def get_json(cache: Any, key: str) -> dict | None:
    raw = await cache.get(key)
    return json.loads(raw) if raw else None


async def set_json(cache: Any, key: str, value: dict, ttl: int) -> None:
    await cache.setex(key, ttl, json.dumps(value, separators=(",", ":"), default=str))
