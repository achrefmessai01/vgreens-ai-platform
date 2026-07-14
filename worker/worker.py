"""GPU worker consumer loop (systemd template unit vgreens-worker@<GPU_ID>).

Lifecycle per message:
  1. preflight (NAS, GPU, docker, disk) — fail => nack/requeue + report
  2. report `running` to the API
  3. stage dataset NAS -> NVMe (training jobs)
  4. docker run the job image (contract mounts + env), tee logs to NAS
  5. exit 0 => copy /out to NAS job folder, report `succeeded` + output_path
     non-zero => report `failed` + log tail
  6. always wipe /nvme/jobs/<id>/
  7. ACK the message ONLY after the report — crash before ack => job requeues

prefetch=1: one job at a time on this GPU.
Registry/validation (Phase 2) hooks into the success path for training jobs.
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

import preflight
import staging
import runner
import registry

GPU_ID = int(os.environ.get("GPU_ID", "0"))
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
API_TOKEN = os.environ.get("API_TOKEN", "")
NVME_JOBS = os.environ.get("NVME_JOBS", "/nvme/jobs")
NAS_JOBS_ROOT = os.environ.get("NAS_JOBS_ROOT", "/mnt/vgreens_storage/ai/jobs")
NAS_MODEL_REGISTRY = os.environ.get("NAS_MODEL_REGISTRY", "/mnt/vgreens_storage/ai/model_registry")
WORKER_NAME = f"{socket.gethostname()}@{GPU_ID}"

TRAINING_QUEUE = "training_jobs"
PROCESSING_QUEUE = "processing_jobs"
TRAINING_LOCK = "training"           # max one training at a time (keeps a GPU free for production)
LOCK_RETRY_SLEEP = 5                 # seconds to wait before requeueing a blocked training job
MAX_PREFLIGHT_ATTEMPTS = 3           # delivery limit: give up after this many preflight failures
HEARTBEAT_SECONDS = 30               # how often to check in with the API (worker-stopped alert)


def _params() -> pika.ConnectionParameters:
    creds = pika.PlainCredentials(
        os.environ.get("RABBITMQ_USER", "vgreens"),
        os.environ.get("RABBITMQ_PASSWORD", ""),
    )
    return pika.ConnectionParameters(
        host=os.environ.get("RABBITMQ_HOST", "127.0.0.1"),
        port=int(os.environ.get("RABBITMQ_PORT", "5672")),
        credentials=creds,
        heartbeat=600,
        blocked_connection_timeout=30,
    )


def _report(job_id: str, status: str, **fields) -> None:
    payload = {"status": status, "worker": WORKER_NAME, "gpu": GPU_ID, **fields}
    try:
        requests.post(
            f"{API_URL}/jobs/{job_id}/status",
            json=payload,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=15,
        )
    except requests.RequestException as exc:
        print(f"[worker] WARN: could not report {status} for {job_id}: {exc}", flush=True)


def _heartbeat_loop() -> None:
    """Check in with the API so a stopped worker can be detected/alerted."""
    while True:
        try:
            requests.post(
                f"{API_URL}/workers/{WORKER_NAME}/heartbeat",
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                timeout=10,
            )
        except requests.RequestException:
            pass
        time.sleep(HEARTBEAT_SECONDS)


def _acquire_lock(name: str, job_id: str) -> bool:
    """Ask the API for a resource lock. Fail-closed: if the API can't be reached,
    treat it as 'not acquired' so we don't run two trainings by accident."""
    try:
        r = requests.post(
            f"{API_URL}/locks/{name}/acquire",
            json={"holder": WORKER_NAME, "job_id": job_id},
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=15,
        )
        return bool(r.ok and r.json().get("acquired"))
    except requests.RequestException as exc:
        print(f"[worker] WARN: lock acquire failed ({name}): {exc}", flush=True)
        return False


def _release_lock(name: str, job_id: str) -> None:
    try:
        requests.post(
            f"{API_URL}/locks/{name}/release",
            json={"job_id": job_id},
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            timeout=15,
        )
    except requests.RequestException as exc:
        print(f"[worker] WARN: lock release failed ({name}): {exc}", flush=True)


def _nas_job_folder(job_id: str) -> Path:
    year = datetime.now(timezone.utc).strftime("%Y")
    return Path(NAS_JOBS_ROOT) / year / job_id


def _active_model_dir(model_name: str) -> tuple[Path | None, str | None]:
    """Resolve the active version for a model from active.json in the registry.
    Returns (version_dir, version) or (None, None) if nothing is promoted."""
    active = Path(NAS_MODEL_REGISTRY) / model_name / "active.json"
    if not active.is_file():
        return None, None
    try:
        version = json.loads(active.read_text()).get("active_version")
    except (OSError, json.JSONDecodeError):
        return None, None
    if not version:
        return None, None
    vdir = Path(NAS_MODEL_REGISTRY) / model_name / "versions" / version
    return (vdir, version) if vdir.is_dir() else (None, None)


def _handle(job: dict) -> None:
    job_id = job["job_id"]
    nas_folder = _nas_job_folder(job_id)
    log_path = str(nas_folder / "logs" / "container.log")
    work = Path(NVME_JOBS) / job_id
    out_dir = work / "out"
    work.mkdir(parents=True, exist_ok=True)   # ensure the job workdir exists before we write into it

    _report(job_id, "running")

    # stage the dataset (any job that names one — training data or images to process)
    if job.get("dataset_name"):
        data_dir = staging.stage_dataset(job["dataset_name"], job["dataset_version"])
    else:
        data_dir = str(work / "data")
        Path(data_dir).mkdir(parents=True, exist_ok=True)

    # processing jobs serve the ACTIVE promoted model (mounted read-only at /model)
    model_dir = None
    model_version = None
    if job["job_type"] == "processing":
        mdir, model_version = _active_model_dir(job["model_name"])
        if mdir is None:
            raise RuntimeError(f"no active (promoted) model for '{job['model_name']}'")
        model_dir = str(mdir)
        print(f"[worker] serving {job['model_name']} active version {model_version}", flush=True)

    env = {
        "JOB_ID": job_id,
        "JOB_TYPE": job["job_type"],
        "MODEL_NAME": job.get("model_name", ""),
        "MODEL_VERSION": model_version or "",
        "DATASET_VERSION": job.get("dataset_version") or "",
        "TENANT": job.get("tenant") or "",
    }
    config_path = job.get("config_path") or str(work / "job.yaml")
    if not Path(config_path).exists():
        Path(config_path).write_text("{}\n")

    t0 = time.time()
    code, tail = runner.run_container(
        image=job["docker_image"],
        gpu_id=GPU_ID,
        data_dir=data_dir,
        config_path=config_path,
        out_dir=str(out_dir),
        env=env,
        log_path=log_path,
        model_dir=model_dir,
    )
    duration = time.time() - t0

    if code == 0:
        nas_out = nas_folder / "out"
        nas_out.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            shutil.copytree(out_dir, nas_out, dirs_exist_ok=True)

        # Training jobs: publish to the model registry + run the validation gate.
        result_version = None
        if job["job_type"] == "training":
            summary = registry.publish(
                job, out_dir, config_path, GPU_ID, WORKER_NAME, duration
            )
            result_version = summary["version"]
            extra = f" ({summary['reason']})" if summary.get("reason") else ""
            print(
                f"[worker] registry: {job['model_name']} {summary['version']} "
                f"-> {summary['status']}{extra}",
                flush=True,
            )
        _report(job_id, "succeeded", output_path=str(nas_folder),
                result_version=result_version)
    else:
        _report(job_id, "failed", error=tail or f"exit code {code}")

    shutil.rmtree(work, ignore_errors=True)


def _on_message(ch, method, properties, body) -> None:
    try:
        job = json.loads(body)
    except json.JSONDecodeError:
        print("[worker] dropping unparseable message", flush=True)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    job_id = job.get("job_id", "?")
    print(f"[worker] received {job_id} ({job.get('job_type')})", flush=True)

    try:
        preflight.run_all()
    except preflight.PreflightError as exc:
        # Delivery limit: bound retries so a persistent failure (e.g. NAS down)
        # can't loop forever. After MAX_PREFLIGHT_ATTEMPTS, give up (dead-letter).
        headers = (properties.headers or {}) if properties else {}
        attempts = int(headers.get("x-attempts", 0)) + 1
        if attempts >= MAX_PREFLIGHT_ATTEMPTS:
            print(f"[worker] preflight failed {attempts}x — giving up {job_id}: {exc}", flush=True)
            _report(job_id, "failed", error=f"preflight failed {attempts}x: {exc}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
        else:
            print(f"[worker] preflight failed ({exc}) — retry {attempts}/{MAX_PREFLIGHT_ATTEMPTS}", flush=True)
            ch.basic_publish(
                exchange="", routing_key=method.routing_key, body=body,
                properties=pika.BasicProperties(
                    delivery_mode=2, content_type="application/json",
                    headers={"x-attempts": attempts}),
            )
            time.sleep(5)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    # Training lock: only ONE training runs platform-wide, so a GPU always stays
    # free for production. Processing jobs skip the lock entirely.
    is_training = job.get("job_type") == "training"
    if is_training and not _acquire_lock(TRAINING_LOCK, job_id):
        print(f"[worker] training busy elsewhere — requeueing {job_id}", flush=True)
        time.sleep(LOCK_RETRY_SLEEP)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        return

    try:
        _handle(job)
    except Exception as exc:  # noqa: BLE001 — report anything, then ack
        print(f"[worker] job {job_id} crashed: {exc}", flush=True)
        _report(job_id, "failed", error=str(exc))
    finally:
        if is_training:
            _release_lock(TRAINING_LOCK, job_id)

    # ack last: if we crash before here, the broker redelivers the job.
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print(f"[worker] done {job_id}", flush=True)


def main() -> None:
    print(f"[worker] starting {WORKER_NAME} (GPU {GPU_ID})", flush=True)
    threading.Thread(target=_heartbeat_loop, daemon=True).start()
    while True:
        try:
            conn = pika.BlockingConnection(_params())
            ch = conn.channel()
            ch.queue_declare(queue=TRAINING_QUEUE, durable=True)
            ch.queue_declare(queue=PROCESSING_QUEUE, durable=True)
            ch.basic_qos(prefetch_count=1)
            ch.basic_consume(TRAINING_QUEUE, _on_message)
            ch.basic_consume(PROCESSING_QUEUE, _on_message)
            print("[worker] preflight OK … waiting for jobs", flush=True)
            ch.start_consuming()
        except pika.exceptions.AMQPConnectionError as exc:
            print(f"[worker] broker unreachable: {exc} — retry in 5s", flush=True)
            time.sleep(5)
        except KeyboardInterrupt:
            print("[worker] shutting down", flush=True)
            break


if __name__ == "__main__":
    main()
