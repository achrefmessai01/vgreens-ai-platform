"""CLI config — reads ~/.vgreens-ai.toml.

Example ~/.vgreens-ai.toml:
    api_url = "http://172.16.20.89:8000"
    token   = "xyz"

Env vars VGREENS_AI_API_URL / VGREENS_AI_TOKEN override the file.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:  # py311+ stdlib; fall back to tomli on older
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

CONFIG_PATH = Path.home() / ".vgreens-ai.toml"


def load() -> tuple[str, str]:
    """Return (api_url, token). Exit with a friendly message if missing."""
    data: dict = {}
    if CONFIG_PATH.exists():
        data = tomllib.loads(CONFIG_PATH.read_text())

    api_url = os.environ.get("VGREENS_AI_API_URL") or data.get("api_url")
    token = os.environ.get("VGREENS_AI_TOKEN") or data.get("token")

    if not api_url or not token:
        sys.stderr.write(
            "error: missing config. Create ~/.vgreens-ai.toml with:\n"
            '  api_url = "http://172.16.20.89:8000"\n'
            '  token   = "<your token>"\n'
        )
        raise SystemExit(2)

    return api_url.rstrip("/"), token
