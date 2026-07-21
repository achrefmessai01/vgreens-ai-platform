"""RabbitMQ publisher (pika).

Queues (declared here AND by the worker, so either can start first):
  training_jobs    -> training jobs
  processing_jobs  -> processing jobs
  failed_jobs      -> dead-letter target (Phase 2 dead-letter handling)

Design choice: open a short-lived connection per publish instead of holding one
long-lived BlockingConnection in the FastAPI process. Job volume is tiny, and a
per-call connection is immune to the "stale/half-open connection" bugs that bite
long-lived pika connections inside a web server. `check_connection()` backs /health.

Messages are published with delivery_mode=2 (persistent) so queued jobs survive a
broker restart. Queues are declared durable for the same reason.
"""
from __future__ import annotations

import json
import os

import pika

TRAINING_QUEUE = "training_jobs"
PROCESSING_QUEUE = "processing_jobs"
FAILED_QUEUE = "failed_jobs"
DATASET_QUEUE = "dataset_jobs"      # CPU-only Dataset Factory (no GPU worker)

QUEUE_FOR_TYPE = {
    "training": TRAINING_QUEUE,
    "processing": PROCESSING_QUEUE,
}


def _params() -> pika.ConnectionParameters:
    host = os.environ.get("RABBITMQ_HOST", "127.0.0.1")
    port = int(os.environ.get("RABBITMQ_PORT", "5672"))
    user = os.environ.get("RABBITMQ_USER", "vgreens")
    password = os.environ.get("RABBITMQ_PASSWORD", "")
    creds = pika.PlainCredentials(user, password)
    return pika.ConnectionParameters(
        host=host,
        port=port,
        credentials=creds,
        heartbeat=30,
        blocked_connection_timeout=10,
        connection_attempts=2,
        retry_delay=1,
    )


def _declare(channel) -> None:
    channel.queue_declare(queue=TRAINING_QUEUE, durable=True)
    channel.queue_declare(queue=PROCESSING_QUEUE, durable=True)
    channel.queue_declare(queue=FAILED_QUEUE, durable=True)
    channel.queue_declare(queue=DATASET_QUEUE, durable=True)


def publish(job: dict) -> None:
    """Publish a job dict to the queue matching its job_type."""
    queue = QUEUE_FOR_TYPE.get(job["job_type"])
    if queue is None:
        raise ValueError(f"unknown job_type: {job['job_type']!r}")

    conn = pika.BlockingConnection(_params())
    try:
        channel = conn.channel()
        _declare(channel)
        channel.basic_publish(
            exchange="",
            routing_key=queue,
            body=json.dumps(job).encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=2,               # persistent
                content_type="application/json",
            ),
        )
    finally:
        conn.close()


def publish_dataset(msg: dict) -> None:
    """Publish a dataset-import message to the CPU-only dataset_jobs queue.

    `msg` must NOT contain secrets (API keys, signed URLs) — only source
    coordinates (workspace/project/path) the dataset worker resolves with its
    own server-side credentials.
    """
    conn = pika.BlockingConnection(_params())
    try:
        channel = conn.channel()
        _declare(channel)
        channel.basic_publish(
            exchange="",
            routing_key=DATASET_QUEUE,
            body=json.dumps(msg).encode("utf-8"),
            properties=pika.BasicProperties(delivery_mode=2, content_type="application/json"),
        )
    finally:
        conn.close()


def check_connection() -> bool:
    """True if the broker is reachable (used by /health)."""
    try:
        conn = pika.BlockingConnection(_params())
        conn.close()
        return True
    except Exception:
        return False
