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

import dataset_safety as safety


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


ADAPTERS = {
    "local": LocalAdapter,
    "nas": LocalAdapter,
}


def get_adapter(source_type: str, params: dict):
    cls = ADAPTERS.get(source_type)
    if cls is None:
        raise ValueError(f"no adapter for source_type={source_type!r} (available: {sorted(ADAPTERS)})")
    return cls(params)
