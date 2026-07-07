"""Static bearer-token auth (MVP).

Tokens come from the env var TOKENS in the form:
    TOKENS=achref:xyz,bilal:abc
The token *name* (left of the colon) becomes `requested_by` on the job.

This is deliberately simple for the internal MVP. Swap for real auth later without
touching callers — they only depend on `require_token`.
"""
from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


def _load_tokens() -> dict[str, str]:
    """Return {token_value: token_name}."""
    raw = os.environ.get("TOKENS", "").strip()
    mapping: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        name, value = pair.split(":", 1)
        mapping[value.strip()] = name.strip()
    return mapping


def require_token(authorization: str = Header(default="")) -> str:
    """FastAPI dependency. Returns the caller name, or raises 401.

    Expects header:  Authorization: Bearer <token>
    """
    tokens = _load_tokens()
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
        )
    value = authorization[len("Bearer "):].strip()
    name = tokens.get(value)
    if name is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
        )
    return name
