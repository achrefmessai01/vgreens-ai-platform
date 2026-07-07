# vgreens-ai-platform

Internal GPU-based AI training & processing platform for vGreens.

One execution path: every job (training or processing) is submitted to a **Job API**
(FastAPI + SQLite on local disk), published to **RabbitMQ**, and consumed by **GPU workers**
(systemd template units, one GPU each). Models are versioned as files in a **model registry**
on the NAS; only validated versions can be promoted to production (manual, auditable,
instantly rollback-able).

No webapp integration. No Kubernetes. No cloud GPUs. No auto-promotion / auto-rollback.

## Layout

```text
api/         FastAPI service (job submission, status, health)
worker/      GPU worker: preflight -> stage -> docker run -> collect -> report
cli/         `vgreens-ai` typer CLI (thin client over the API)
deploy/      docker-compose (RabbitMQ + later Prometheus/Grafana), systemd units
dummy-job/   contract-compliant test image (torch matmul on GPU)
```

## Deploy target

`/opt/vgreens-ai/` on `gpu-worker-01` (172.16.20.89):
venv, root-only `.env`, `docker-compose.yml`, local `jobs.db`.

## Container contract

Every job image receives:

- mounts: `/data` (ro), `/config/job.yaml` (ro), `/out` (rw)
- env: `JOB_ID`, `JOB_TYPE`, `MODEL_NAME`, `DATASET_VERSION`, `TENANT`
- exit `0` = success. Training writes model files + `eval/metrics.json` +
  `evaluation_report.md` to `/out`. Processing writes results + `report.json`.

## Phase status

- Phase 0 (infra): DONE.
- Phase 1 (this repo): 1.1 scaffold + 1.2 RabbitMQ (current), then API / CLI / worker / dummy job.

See `deploy/install.md` for step-by-step deployment.
