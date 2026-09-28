"""Deterministic test environment.

Local developers may have a real `.env` with provider credentials. Tests must
never consume live provider quota or change behaviour based on those secrets.
Environment variables override pydantic's `.env` loading before application
modules are imported.
"""
from __future__ import annotations

import os

os.environ["ENV"] = "test"
os.environ["ALPHA_VANTAGE_KEY"] = ""
os.environ["REDDIT_CLIENT_ID"] = ""
os.environ["REDDIT_SECRET"] = ""
os.environ["CACHE_BACKEND"] = "memory"
os.environ["MARKET_DATA_PROVIDER"] = "bundled"
