"""Dataset source adapters.

Each adapter knows how to validate its source coordinates and stage the raw
data into a quarantine directory. Secrets (API keys, credentials) come from the
worker's own environment — never from the request/source_params.

Interface:
    validate_source(roots)      -> raise on invalid/unsafe coordinates
    stage(dest, roots)          -> stage raw data into `dest`, return its path
    licence_metadata()          -> dict describing the source licence (may be {})

5B.3 ships the local/NAS adapter. Roboflow (5B.5), HTTPS + S3 come later behind
the same interface.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

import dataset_safety as safety

ROBOFLOW_API = os.environ.get("ROBOFLOW_API_URL", "https://api.roboflow.com")
MAX_DOWNLOAD_BYTES = int(os.environ.get("DATASET_MAX_DOWNLOAD_BYTES", str(30 * 1024 ** 3)))


class LocalAdapter:
    """source_type=local | nas. Stage from an allowlisted directory or a .zip."""

    source_type = "local"

    def __init__(self, params: dict):
        self.params = params or {}

    def _source_ref(self) -> str:
        ref = self.params.get("path") or self.params.get("zip")
        if not ref:
            raise ValueError("local source requires 'path' (directory) or 'zip' (archive)")
        return ref

    def validate_source(self, roots: list[str]) -> None:
        # resolve_within raises SafetyError if the path escapes the allowlist
        safety.resolve_within(self._source_ref(), roots)

    def stage(self, dest: str, roots: list[str]):
        if self.params.get("zip"):
            zip_path = safety.resolve_within(self.params["zip"], roots)
            return safety.safe_extract_zip(zip_path, dest)
        return safety.stage_local_dir(self.params["path"], dest, roots)

    def licence_metadata(self) -> dict:
        return self.params.get("licence") or {}


class RoboflowAdapter:
    """source_type=roboflow. Download a version export (yolov8) via the REST API
    and quarantine it through the same safe zip extractor as any archive.

    The API key comes from the worker's ROBOFLOW_API_KEY env var — never from
    source_params. The signed export link is used transiently and never logged
    or stored in the manifest.
    """

    source_type = "roboflow"

    def __init__(self, params: dict):
        self.params = params or {}

    def _coords(self) -> tuple[str, str, str]:
        ws = self.params.get("workspace")
        pr = self.params.get("project")
        ver = self.params.get("rf_version", self.params.get("version"))
        if not (ws and pr and ver is not None):
            raise ValueError("roboflow source needs workspace, project, rf_version")
        return ws, pr, str(ver)

    def validate_source(self, roots: list[str]) -> None:
        self._coords()
        if not os.environ.get("ROBOFLOW_API_KEY"):
            raise ValueError("ROBOFLOW_API_KEY is not configured on the server")

    def _export_link(self, ws: str, pr: str, ver: str, key: str,
                     fmt: str = "yolov8", timeout: int = 600) -> str:
        url = f"{ROBOFLOW_API}/{ws}/{pr}/{ver}/{fmt}"
        deadline = time.time() + timeout
        while True:
            r = requests.get(url, params={"api_key": key}, timeout=60)
            r.raise_for_status()
            data = r.json()
            link = (data.get("export") or {}).get("link")
            if link:
                return link
            if data.get("ready") is False:
                if time.time() > deadline:
                    raise RuntimeError("roboflow export generation timed out")
                time.sleep(5)
                continue
            raise RuntimeError(f"unexpected roboflow export response (keys: {sorted(data)})")

    def stage(self, dest: str, roots: list[str]):
        ws, pr, ver = self._coords()
        key = os.environ["ROBOFLOW_API_KEY"]
        link = self._export_link(ws, pr, ver, key)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        zpath = dest.parent / "roboflow_export.zip"
        total = 0
        with requests.get(link, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in resp.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise safety.SafetyError("roboflow download exceeds size limit")
                    f.write(chunk)
        try:
            safety.safe_extract_zip(zpath, dest)
        finally:
            zpath.unlink(missing_ok=True)
        return dest

    def licence_metadata(self) -> dict:
        return self.params.get("licence") or {}


ADAPTERS = {
    "local": LocalAdapter,
    "nas": LocalAdapter,
    "roboflow": RoboflowAdapter,
}


def get_adapter(source_type: str, params: dict):
    cls = ADAPTERS.get(source_type)
    if cls is None:
        raise ValueError(f"no adapter for source_type={source_type!r} (available: {sorted(ADAPTERS)})")
    return cls(params)
