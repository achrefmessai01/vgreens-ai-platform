"""CLI configuration — multi-profile, backward-compatible with the old flat file.

File: ~/.vgreens-ai.toml  (chmod 600). New format:

    default_profile = "prod"
    [profiles.prod]
    api_url = "https://ai.internal/api"
    token   = "..."

Legacy flat format (still read):
    api_url = "..."
    token   = "..."

Resolution order for api_url/token: explicit CLI flag > env
(VGREENS_AI_API_URL / VGREENS_AI_TOKEN) > selected profile > legacy top-level.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import tomli_w

CONFIG_PATH = Path(os.environ.get("VGREENS_AI_CONFIG", str(Path.home() / ".vgreens-ai.toml")))


class ConfigError(Exception):
    pass


def _read() -> dict:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        return tomllib.loads(CONFIG_PATH.read_text())
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"could not parse {CONFIG_PATH}: {exc}")


def raw() -> dict:
    return _read()


def profiles() -> dict:
    return _read().get("profiles", {})


def default_profile() -> str | None:
    data = _read()
    return data.get("default_profile") or (next(iter(data.get("profiles", {})), None))


def resolve(profile: str | None = None,
            api_url: str | None = None,
            token: str | None = None) -> tuple[str, str, str]:
    """Return (api_url, token, profile_name). Raises ConfigError if unresolved."""
    data = _read()
    name = (profile or os.environ.get("VGREENS_AI_PROFILE")
            or data.get("default_profile")
            or (next(iter(data.get("profiles", {})), None)) or "default")

    prof = data.get("profiles", {}).get(name, {})
    # legacy flat file: treat top-level api_url/token as the "default" profile
    if not prof and name == "default":
        prof = {k: data[k] for k in ("api_url", "token") if k in data}

    url = api_url or os.environ.get("VGREENS_AI_API_URL") or prof.get("api_url")
    tok = token or os.environ.get("VGREENS_AI_TOKEN") or prof.get("token")

    if not url or not tok:
        raise ConfigError(
            "no API URL/token configured — run `vgreens-ai config init` "
            "(or pass --api-url/--token, or set VGREENS_AI_API_URL/VGREENS_AI_TOKEN)"
        )
    return url.rstrip("/"), tok, name


def save_profile(name: str, api_url: str, token: str, make_default: bool = True) -> None:
    data = _read()
    data.setdefault("profiles", {})[name] = {"api_url": api_url.rstrip("/"), "token": token}
    if make_default or "default_profile" not in data:
        data["default_profile"] = name
    for k in ("api_url", "token"):   # drop legacy top-level keys so they don't shadow profiles
        data.pop(k, None)
    CONFIG_PATH.write_text(tomli_w.dumps(data))
    CONFIG_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600


def set_default(name: str) -> None:
    data = _read()
    if name not in data.get("profiles", {}):
        raise ConfigError(f"no such profile: {name}")
    data["default_profile"] = name
    CONFIG_PATH.write_text(tomli_w.dumps(data))
    CONFIG_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)


def redact(token: str) -> str:
    if not token:
        return "(none)"
    return token[:4] + "…" + token[-2:] if len(token) > 8 else "set"
