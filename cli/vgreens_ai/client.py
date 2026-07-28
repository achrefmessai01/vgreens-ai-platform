"""Thin HTTP client over the Job API. All business logic stays server-side.

Stable exit codes (documented in docs/cli/exit-codes.md):
  0 ok · 1 generic · 2 usage (Typer) · 3 network/unreachable · 4 auth · 5 not found
  6 conflict · 7 server error
"""
from __future__ import annotations

from typing import Any

import requests

from . import output

EXIT_GENERIC = 1
EXIT_NETWORK = 3
EXIT_AUTH = 4
EXIT_NOT_FOUND = 5
EXIT_CONFLICT = 6
EXIT_SERVER = 7


class CLIError(Exception):
    def __init__(self, message: str, code: int = EXIT_GENERIC):
        super().__init__(message)
        self.code = code


class Client:
    def __init__(self, api_url: str, token: str, timeout: float = 30.0, quiet: bool = False):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.quiet = quiet
        self._s = requests.Session()
        self._s.headers["Authorization"] = f"Bearer {token}"

    def request(self, method: str, path: str, *, spinner: str | None = None, **kw) -> Any:
        url = f"{self.api_url}{path}"
        try:
            if spinner and not self.quiet:
                with output.err.status(spinner):
                    r = self._s.request(method, url, timeout=self.timeout, **kw)
            else:
                r = self._s.request(method, url, timeout=self.timeout, **kw)
        except requests.exceptions.SSLError as exc:
            raise CLIError(f"TLS error talking to {self.api_url}: {exc}", EXIT_NETWORK)
        except requests.exceptions.ConnectionError:
            raise CLIError(f"cannot reach the API at {self.api_url} "
                           f"(is the URL right / VPN up?)", EXIT_NETWORK)
        except requests.exceptions.Timeout:
            raise CLIError(f"request to {self.api_url} timed out after {self.timeout}s", EXIT_NETWORK)

        if r.status_code == 401:
            raise CLIError("authentication failed — check your token (`vgreens-ai config init`)", EXIT_AUTH)
        if r.status_code == 403:
            raise CLIError("not permitted — your role can't perform this action", EXIT_AUTH)
        if r.status_code == 404:
            raise CLIError(self._detail(r) or "not found", EXIT_NOT_FOUND)
        if r.status_code == 409:
            raise CLIError(self._detail(r) or "conflict", EXIT_CONFLICT)
        if r.status_code >= 500:
            raise CLIError(f"server error ({r.status_code}): {self._detail(r)}", EXIT_SERVER)
        if r.status_code >= 400:
            raise CLIError(self._detail(r) or f"request failed ({r.status_code})", EXIT_GENERIC)

        self._last_headers = r.headers
        if r.content and "application/json" in r.headers.get("content-type", ""):
            return r.json()
        return None

    @staticmethod
    def _detail(r: requests.Response) -> str:
        try:
            return str(r.json().get("detail", r.text))
        except Exception:  # noqa: BLE001
            return r.text.strip()

    def get(self, path: str, **kw) -> Any:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw) -> Any:
        return self.request("POST", path, **kw)

    def delete(self, path: str, **kw) -> Any:
        return self.request("DELETE", path, **kw)

    def download(self, path: str, dest: str, *, params: dict | None = None,
                 spinner: str | None = None) -> str:
        """Stream a binary response to a local file. Maps HTTP errors to the same
        exit codes as request()."""
        import os
        url = f"{self.api_url}{path}"
        try:
            ctx = output.err.status(spinner) if (spinner and not self.quiet) else None
            if ctx:
                ctx.__enter__()
            try:
                r = self._s.get(url, params=params, timeout=self.timeout, stream=True)
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)
        except requests.exceptions.ConnectionError:
            raise CLIError(f"cannot reach the API at {self.api_url}", EXIT_NETWORK)
        except requests.exceptions.Timeout:
            raise CLIError(f"request to {self.api_url} timed out after {self.timeout}s", EXIT_NETWORK)
        if r.status_code == 401:
            raise CLIError("authentication failed — check your token", EXIT_AUTH)
        if r.status_code == 403:
            raise CLIError("not permitted — your role can't perform this action", EXIT_AUTH)
        if r.status_code == 404:
            raise CLIError(self._detail(r) or "not found", EXIT_NOT_FOUND)
        if r.status_code >= 400:
            raise CLIError(self._detail(r) or f"download failed ({r.status_code})", EXIT_GENERIC)
        if os.path.isdir(dest):
            name = os.path.basename(params.get("path", "artifact")) if params else "artifact"
            dest = os.path.join(dest, name or "artifact")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return dest
