"""LIVE monitoring (Phase 1): the snapshot endpoint and the SSE stream.

Covers:
  - GET /system/snapshot   one curated monitoring payload (all sections present)
  - GET /system/stream     SSE stream; first event is `data: <json>` with a
                           generated_at timestamp
"""
import os
import sys
import json
import asyncio
import tempfile

os.environ.setdefault("JOBS_DB_PATH", tempfile.mktemp(suffix=".db"))
os.environ.setdefault("TOKENS", "svc:statictok")
os.environ.setdefault("ADMIN_USERNAME", "admin")
os.environ.setdefault("ADMIN_PASSWORD", "adminpass1")
os.environ.setdefault("MONITOR_STREAM_INTERVAL", "0.1")  # snappy loop for the test
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # api/

from fastapi.testclient import TestClient  # noqa: E402
import db      # noqa: E402
import authn   # noqa: E402
import broker  # noqa: E402
import main    # noqa: E402

broker.publish = lambda job: None

db.init_db()
authn.seed_admin_if_empty()
client = TestClient(main.app)
ADMIN_PW = os.environ["ADMIN_PASSWORD"]


def _login():
    client.cookies.clear()
    assert client.post("/auth/login", json={"username": "admin", "password": ADMIN_PW}).status_code == 200


def test_snapshot_has_all_sections():
    _login()
    r = client.get("/system/snapshot")
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("overview", "gpus", "workers", "recent_jobs",
                "recent_errors", "storage", "generated_at"):
        assert key in body, f"missing key {key!r}"
    # overview keeps the /system/overview shape
    assert "nas_mounted" in body["overview"]
    assert "nas_path" in body["storage"]


def test_stream_emits_sse_json():
    """The SSE endpoint returns text/event-stream and its first frame is a
    `data: <json>` event with a generated_at timestamp.

    NB: Starlette's TestClient fully buffers a response before returning, so
    `client.stream()` would hang forever on this intentionally-infinite stream.
    We instead drive the real StreamingResponse body iterator directly (the same
    code path the ASGI server runs), with a stub Request whose is_disconnected()
    trips after the first frame so the generator's disconnect-break can end it.
    """
    async def _first_frame():
        seen = {"n": 0}

        class _Req:  # minimal Starlette Request stand-in for is_disconnected()
            async def is_disconnected(self) -> bool:
                seen["n"] += 1
                return seen["n"] > 1  # connected for the first frame, then gone

        resp = await main.system_stream(_Req())
        assert resp.media_type == "text/event-stream"
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"
        chunk = await resp.body_iterator.__anext__()
        return chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk

    frame = asyncio.run(_first_frame())
    assert frame.startswith("data: "), frame[:40]
    payload = json.loads(frame[len("data: "):].strip())
    assert "generated_at" in payload
