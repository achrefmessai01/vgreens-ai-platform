"""CPU-only Dataset Factory worker (systemd unit vgreens-dataset-worker).

Consumes `dataset_jobs`, stages an untrusted source into an isolated quarantine
directory on NVMe, and runs safety checks. NO GPU, NO docker, NO NAS writes to
the frozen tree in this stage.

Reports state transitions to the API (the sole DB writer) via
POST /datasets/imports/{id}/state — mirroring how the GPU worker reports jobs.

Pipeline (5B.3):  submitted -> staging -> quarantined
                  (safety failure -> rejected;  other error -> failed)
Audit -> normalise -> atomic freeze arrive in 5B.4, extending the same worker.

ack-last: if we crash before ack, RabbitMQ redelivers the import.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pika
import requests

import dataset_adapters as adapters
import dataset_audit as audit_mod
import dataset_freeze as freezelib
import dataset_safety as safety


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
API_TOKEN = os.environ.get("API_TOKEN", "")
DATASET_QUEUE = "dataset_jobs"
STAGING_ROOT = os.environ.get("DATASET_STAGING_ROOT", "/nvme/dataset-staging")
# allowlisted roots a 'local'/'nas' source may read from
IMPORT_ROOTS = [r for r in os.environ.get(
    "DATASET_IMPORT_ROOTS", "/mnt/vgreens_storage").split(":") if r]
WORKER_NAME = f"{socket.gethostname()}@dataset"
HEARTBEAT_SECONDS = 30
MAX_ATTEMPTS = 3


def _params() -> pika.ConnectionParameters:
    creds = pika.PlainCredentials(
        os.environ.get("RABBITMQ_USER", "vgreens"),
        os.environ.get("RABBITMQ_PASSWORD", ""),
    )
    return pika.ConnectionParameters(
        host=os.environ.get("RABBITMQ_HOST", "127.0.0.1"),
        port=int(os.environ.get("RABBITMQ_PORT", "5672")),
        credentials=creds, heartbeat=600, blocked_connection_timeout=30,
    )


def _report(import_id: str, state: str, **fields) -> None:
    try:
        requests.post(
            f"{API_URL}/datasets/imports/{import_id}/state",
            json={"state": state, **fields},
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=15,
        )
    except requests.RequestException as exc:
        print(f"[dataset-worker] WARN: could not report {state} for {import_id}: {exc}", flush=True)


def _get_import(import_id: str) -> dict | None:
    try:
        r = requests.get(
            f"{API_URL}/datasets/imports/{import_id}",
            headers={"Authorization": f"Bearer {API_TOKEN}"}, timeout=15,
        )
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def _api_post(path: str, payload: dict) -> dict:
    r = requests.post(f"{API_URL}{path}", json=payload,
                      headers={"Authorization": f"Bearer {API_TOKEN}"}, timeout=60)
    r.raise_for_status()
    return r.json() if r.content else {}


def _heartbeat_loop() -> None:
    while True:
        try:
            requests.post(f"{API_URL}/workers/{WORKER_NAME}/heartbeat",
                          headers={"Authorization": f"Bearer {API_TOKEN}"}, timeout=10)
        except requests.RequestException:
            pass
        time.sleep(HEARTBEAT_SECONDS)


def _stage_dir(import_id: str) -> Path:
    return Path(STAGING_ROOT) / import_id


def _handle(msg: dict) -> None:
    import_id = msg["import_id"]
    source_type = msg.get("source_type")
    params = msg.get("source_params") or {}
    dataset_name = msg.get("dataset_name")

    # respect a cancellation that landed after the message was queued
    current = _get_import(import_id)
    if current and current.get("state") in ("cancelled", "frozen", "rejected", "failed"):
        print(f"[dataset-worker] {import_id} already {current['state']} — skipping", flush=True)
        return

    stage = _stage_dir(import_id)
    raw = stage / "raw"
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)   # clean slate for a retry
    raw.mkdir(parents=True, exist_ok=True)

    adapter = adapters.get_adapter(source_type, params)
    _report(import_id, "staging")
    adapter.validate_source(IMPORT_ROOTS)
    adapter.stage(str(raw), IMPORT_ROOTS)
    _report(import_id, "quarantined", detail=f"staged to {raw}")

    # ---- audit (fatal findings reject the import) ----
    _report(import_id, "auditing")
    report = audit_mod.audit(raw)
    if report.get("fatal"):
        fails = [f["msg"] for f in report["findings"] if f["level"] == "fail"]
        _report(import_id, "rejected", error="audit failed: " + "; ".join(fails[:5]))
        print(f"[dataset-worker] {import_id} rejected by audit", flush=True)
        return

    # ---- licence gate ----
    # internal local/nas data is owned (no approval); external sources need an
    # admin approval, recorded on the import and honoured on the re-run.
    lic = adapter.licence_metadata()
    stored_status = (current or {}).get("licence_status")
    if source_type in ("local", "nas"):
        licence_status = "not_required"
    elif stored_status == "approved":
        licence_status = "approved"
    else:
        licence_status = lic.get("approval_status") or "unknown"
    if licence_status not in ("approved", "not_required"):
        _report(import_id, "awaiting_license_approval",
                detail=f"licence '{lic.get('name', 'unknown')}' needs approval")
        print(f"[dataset-worker] {import_id} awaiting licence approval", flush=True)
        return

    # ---- normalise to canonical layout ----
    _report(import_id, "normalising")
    normalised = stage / "normalised"
    ninfo = freezelib.normalise(raw, normalised, report["classes"], report["num_classes"])
    digest, checks = freezelib.hash_tree(normalised)

    # ---- allocate version (the API owns the DB / atomic allocation) ----
    alloc = _api_post(f"/datasets/{dataset_name}/versions/allocate", {
        "import_id": import_id,
        "requested_version": msg.get("requested_version"),
        "task": "object_detection",
        "image_count": ninfo["image_count"],
        "class_count": ninfo["class_count"],
        "licence_name": lic.get("name"),
        "licence_status": licence_status,
    })
    version, nas_path = alloc["version"], alloc["nas_path"]

    # ---- atomic freeze onto the NAS ----
    _report(import_id, "freezing", detail=f"version {version}")
    safe_params = {k: v for k, v in params.items() if k != "licence"}
    manifest = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "dataset_version": version,
        "task": "object_detection",
        "source": {"type": source_type, **safe_params},
        "licence": {"name": lic.get("name"), "url": lic.get("url"),
                    "attribution_required": lic.get("attribution_required"),
                    "approval_status": licence_status},
        "content": {"image_count": ninfo["image_count"], "class_count": ninfo["class_count"],
                    "classes": report["classes"], "content_digest": digest},
        "generation": {"tool": "vgreens-dataset-factory", "generated_at": _now(),
                       "import_id": import_id},
    }
    source_meta = {"type": source_type, "params": safe_params, "audit_findings": report["findings"]}
    freezelib.freeze(normalised, nas_path, manifest, checks, report, source_meta)

    # ---- finalize (publishes the version + flips the import to frozen) ----
    _api_post(f"/datasets/{dataset_name}/versions/{version}/finalize", {
        "import_id": import_id, "content_digest": digest,
        "image_count": ninfo["image_count"], "class_count": ninfo["class_count"],
    })
    print(f"[dataset-worker] {import_id} FROZEN as {dataset_name}:{version}", flush=True)
    shutil.rmtree(stage, ignore_errors=True)


def _on_message(ch, method, properties, body) -> None:
    try:
        msg = json.loads(body)
    except json.JSONDecodeError:
        print("[dataset-worker] dropping unparseable message", flush=True)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    import_id = msg.get("import_id", "?")
    print(f"[dataset-worker] received {import_id} ({msg.get('source_type')})", flush=True)
    try:
        _handle(msg)
    except safety.SafetyError as exc:
        print(f"[dataset-worker] REJECT {import_id}: {exc}", flush=True)
        _report(import_id, "rejected", error=str(exc))
    except (ValueError, KeyError) as exc:      # bad request / adapter args — don't retry
        print(f"[dataset-worker] REJECT {import_id}: {exc}", flush=True)
        _report(import_id, "rejected", error=str(exc))
    except Exception as exc:                    # noqa: BLE001 — transient; bounded retry
        headers = (properties.headers or {}) if properties else {}
        attempts = int(headers.get("x-attempts", 0)) + 1
        if attempts >= MAX_ATTEMPTS:
            print(f"[dataset-worker] FAIL {import_id} after {attempts}x: {exc}", flush=True)
            _report(import_id, "failed", error=f"{exc} (after {attempts} attempts)")
        else:
            print(f"[dataset-worker] retry {import_id} {attempts}/{MAX_ATTEMPTS}: {exc}", flush=True)
            ch.basic_publish(exchange="", routing_key=method.routing_key, body=body,
                             properties=pika.BasicProperties(
                                 delivery_mode=2, content_type="application/json",
                                 headers={"x-attempts": attempts}))
            time.sleep(3)
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print(f"[dataset-worker] done {import_id}", flush=True)


def main() -> None:
    print(f"[dataset-worker] starting {WORKER_NAME} "
          f"(staging={STAGING_ROOT}, roots={IMPORT_ROOTS})", flush=True)
    Path(STAGING_ROOT).mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_heartbeat_loop, daemon=True).start()
    while True:
        try:
            conn = pika.BlockingConnection(_params())
            ch = conn.channel()
            ch.queue_declare(queue=DATASET_QUEUE, durable=True)
            ch.basic_qos(prefetch_count=1)
            ch.basic_consume(DATASET_QUEUE, _on_message)
            print("[dataset-worker] waiting for dataset imports", flush=True)
            ch.start_consuming()
        except pika.exceptions.AMQPConnectionError as exc:
            print(f"[dataset-worker] broker unreachable: {exc} — retry in 5s", flush=True)
            time.sleep(5)
        except KeyboardInterrupt:
            print("[dataset-worker] shutting down", flush=True)
            break


if __name__ == "__main__":
    main()
