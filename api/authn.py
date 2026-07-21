"""Authentication: bcrypt passwords, opaque session cookies, and a unified
identity resolver that accepts (in priority order):

  1. a browser session cookie  (vg_session)   -> the logged-in user
  2. a DB-backed API token      (Bearer …)     -> its owner + scopes
  3. a legacy static token      (Bearer …)      -> TOKENS env, treated as admin

Static tokens keep the existing CLI/CI working while we migrate. Sessions and
DB tokens are stored only as sha256 hashes; raw values never hit the database.
"""
from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Cookie, Depends, Header, HTTPException

import auth as static_auth
import db

SESSION_COOKIE = "vg_session"
SESSION_TTL_HOURS = int(os.environ.get("SESSION_TTL_HOURS", "12"))
ROLES = ("viewer", "operator", "admin")
_ROLE_RANK = {r: i for i, r in enumerate(ROLES)}


@dataclass
class Identity:
    username: str
    role: str
    via: str                     # "session" | "api_token" | "static"
    scopes: Optional[list] = None
    user_id: Optional[int] = None


# ---- passwords ----
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except (ValueError, TypeError):
        return False


# ---- sessions ----
def _sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_session(user_id: int) -> str:
    """Create a session and return the RAW token to put in the cookie."""
    raw = secrets.token_urlsafe(32)
    expires = (_now() + timedelta(hours=SESSION_TTL_HOURS)).isoformat()
    db.create_session(_sha256(raw), user_id, expires)
    return raw


def resolve_session(raw: str) -> Optional[dict]:
    row = db.get_session(_sha256(raw))
    if row is None or row["revoked"]:
        return None
    try:
        if datetime.fromisoformat(row["expires_at"]) < _now():
            return None
    except (TypeError, ValueError):
        return None
    user = db.get_user_by_id(row["user_id"])
    if user is None or not user["active"]:
        return None
    db.touch_session(row["token_hash"])
    return user


def revoke_session(raw: str) -> None:
    db.revoke_session(_sha256(raw))


# ---- API tokens ----
def new_api_token() -> tuple[str, str, str]:
    """Return (token_id, raw_token, token_hash). Show `raw` to the caller once;
    only the hash is persisted."""
    token_id = secrets.token_hex(6)
    raw = "vgt_" + secrets.token_urlsafe(24)
    return token_id, raw, _sha256(raw)


# ---- unified identity resolution ----
def _from_bearer(token: str) -> Optional[Identity]:
    # DB-backed API token first
    row = db.get_api_token(_sha256(token))
    if row is not None and not row["revoked"]:
        scopes = (row.get("scopes") or "").split() or None
        return Identity(username=row.get("owner") or row.get("name") or "token",
                        role="operator", via="api_token", scopes=scopes)
    # legacy static token (TOKENS env) -> admin-equivalent for back-compat
    name = static_auth._load_tokens().get(token)
    if name is not None:
        return Identity(username=name, role="admin", via="static")
    return None


def resolve_identity(authorization: str = "", session: str = "") -> Optional[Identity]:
    if session:
        user = resolve_session(session)
        if user is not None:
            return Identity(username=user["username"], role=user["role"],
                            via="session", user_id=user["id"])
    if authorization.startswith("Bearer "):
        return _from_bearer(authorization[len("Bearer "):].strip())
    return None


def role_at_least(role: str, minimum: str) -> bool:
    return _ROLE_RANK.get(role, -1) >= _ROLE_RANK.get(minimum, 99)


def actor(minimum: str = "viewer"):
    """FastAPI dependency factory. Resolves the caller (session cookie / API
    token / static token), enforces a minimum role, and returns the username
    string (so existing handlers that used the caller name keep working)."""
    def dep(authorization: str = Header(default=""),
            vg_session: str = Cookie(default="")) -> str:
        ident = resolve_identity(authorization, vg_session)
        if ident is None:
            raise HTTPException(status_code=401, detail="authentication required")
        if not role_at_least(ident.role, minimum):
            raise HTTPException(status_code=403, detail=f"requires '{minimum}' role or higher")
        return ident.username
    return dep


# ---- FastAPI dependency ----
def current_identity(authorization: str = Header(default=""),
                     vg_session: str = Cookie(default="")) -> Optional[Identity]:
    return resolve_identity(authorization, vg_session)


# ---- bootstrap ----
def seed_admin_if_empty() -> None:
    """On first boot, create an admin from ADMIN_USERNAME/ADMIN_PASSWORD if set
    and there are no users yet. The password is never logged."""
    if db.count_users() > 0:
        return
    username = os.environ.get("ADMIN_USERNAME")
    password = os.environ.get("ADMIN_PASSWORD")
    if not username or not password:
        print("[authn] no users and ADMIN_USERNAME/ADMIN_PASSWORD unset — "
              "create an admin to use browser login", flush=True)
        return
    db.create_user(username, hash_password(password), role="admin", display_name=username)
    print(f"[authn] seeded initial admin user '{username}'", flush=True)
