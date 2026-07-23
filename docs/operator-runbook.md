# vGreens AI Platform — Operator Runbook

How to run the platform day to day: access, routine operations, and fixing the
things that actually go wrong. Assumes you're an operator/admin with VPN access.

---

## Access

- **VPN** must be connected for anything below.
- **Web UI:** `https://app.vgreens.lan:9443` (trust the Caddy internal CA once, or
  accept the browser warning).
- **CLI:** `vgreens-ai …` (profile `prod` → `https://vgreens-ai.lan:9443`, with
  `REQUESTS_CA_BUNDLE=$HOME/.config/vgreens/caddy-root.crt`).
- **SSH to the GPU host** (through the sv01 jump):
  `ssh -J vgreens@10.8.0.30 vgreens@172.16.20.89`
- **SSH to sv01** (runner + proxy): `ssh vgreens@10.8.0.30`

`/etc/hosts` on your Mac maps `vgreens-ai.lan` and `app.vgreens.lan` → `10.8.0.30`.

---

## Service map

**gpu-worker-01 (systemd):**
| Unit | Role |
|------|------|
| `vgreens-ai-api` | Job API (FastAPI, :8000) |
| `vgreens-worker@0`, `@1` | GPU workers (one per GPU) |
| `vgreens-dataset-worker` | Dataset Factory worker |
| `mount-nas` | NAS mount dependency |
| `vgreens-db-backup.timer` | nightly jobs.db backup (02:30) |
| RabbitMQ, Prometheus, Grafana | broker + monitoring (containers/compose) |

**sv01 (systemd):** `gitlab-runner`, `caddy` (reverse proxy). *(Also an unrelated
`nginx` livestream on :443 — leave it alone; Caddy is on :9443.)*

Quick health:
```bash
systemctl is-active vgreens-ai-api vgreens-worker@0 vgreens-worker@1 vgreens-dataset-worker
curl -sS https://vgreens-ai.lan:9443/health/ready ; echo
vgreens-ai system status   # or: vgreens-ai list-jobs
```

---

## Routine operations

### Deploy a code change
```bash
# on gpu-worker-01
cd /opt/vgreens-ai/repo && git pull
sudo systemctl restart vgreens-ai-api            # for API changes
sudo systemctl restart vgreens-worker@0 vgreens-worker@1   # for worker changes
```
Never hand-edit the SQLite schema — schema changes ship as migrations in the code
and apply on API start. Never let anything overwrite `/opt/vgreens-ai/.env` or
`jobs.db`.

### Add a user / issue an API token (admin)
Web **Admin** page → create user (viewer/operator/admin) or create an API token
(shown once — copy it then). CLI equivalent exists under the admin endpoints.
Scoped CI tokens are operator-level.

### Publish a model's job-config
Automatic in CI now (the `submit_training` step does it). Manually if needed:
```bash
vgreens-ai config publish <model> --file configs/<model>.yaml
```

### Promote / roll back a model (admin, audited)
```bash
vgreens-ai model show <model>
vgreens-ai model promote <model> <version>
vgreens-ai model rollback <model> <previous-version>
```
Only `validated` versions promote; `baseline_only` (below threshold) cannot.

### Cancel / inspect a job
```bash
vgreens-ai list-jobs
vgreens-ai job show <job_id>          # status + error
vgreens-ai job events <job_id>        # timeline
vgreens-ai job cancel <job_id>
```

### Two-GPU training (human gate)
Requires admin and an explicit flag: `vgreens-ai job train … --gpu-count 2 --yes`.
The worker acquires both per-GPU locks before starting.

### Backup / restore jobs.db
- Backups: nightly to `/mnt/vgreens_storage/ai/backups/jobs.db.<stamp>.gz`
  (local snapshot → integrity check → copy to NAS). Run on demand:
  `sudo systemctl start vgreens-db-backup.service`.
- **Restore** (into a temp copy first, verify, then swap with the API stopped):
  ```bash
  latest=$(ls -1t /mnt/vgreens_storage/ai/backups/jobs.db.*.gz | head -1)
  gunzip -c "$latest" > /tmp/restore.db
  /opt/vgreens-ai/venv/bin/python -c "import sqlite3;print(sqlite3.connect('/tmp/restore.db').execute('PRAGMA integrity_check').fetchone())"
  sudo systemctl stop vgreens-ai-api
  sudo cp /tmp/restore.db /opt/vgreens-ai/jobs.db
  sudo systemctl start vgreens-ai-api
  ```

### Rotate secrets
- **API token:** revoke on the Admin page, issue a new one, update wherever it's used.
- **GitLab CI token (`VGREENS_CI_TOKEN`):** group → Settings → CI/CD → Variables.
- **Registry pull:** group deploy token (`read_registry`); re-`docker login registry.gitlab.com` on gpu-worker-01.
- **GitLab runner token:** re-create the group runner, update `vault_gitlab_runner_token` in the Ansible vault, re-run `playbooks/gitlab-runner.yml`.
- **Ansible Vault:** `ansible-vault edit inventories/production/group_vars/vault.yml`.
Never paste secrets into chat, tickets, or logs.

### Onboard a new model
See `model-developer-guide/`. Short version:
`vgreens-ai model init` → `vgreens-ai model onboard --spec model.yaml --create-repo`
(needs `GITLAB_TOKEN`) → click `submit_training` in CI → `model promote`.

---

## Troubleshooting (symptoms we've actually hit)

| Symptom | Cause | Fix |
|---------|-------|-----|
| CLI/`curl` to the API hangs from your Mac | VPN doesn't route the office LAN; you're hitting `172.16.20.89` directly | Use `https://vgreens-ai.lan:9443` (the proxy on sv01), not the LAN IP |
| Job fails instantly: `error from registry: access forbidden` | Worker can't pull the model image (wrong/absent registry credential) | On gpu-worker-01, `docker login registry.gitlab.com` with a **group** deploy token (`read_registry`) |
| CI pipeline runs on a GitLab cloud runner, not sv01 | instance runners enabled / jobs untagged | Jobs are tagged `vgreens-sv01-cpu`; ensure the group runner has that tag and instance runners are off |
| `apt`/CI build stalls, `archive.ubuntu.com` at 0 B/s | the default Ubuntu mirror is unreachable from these hosts | mirror is pinned to `de.archive.ubuntu.com` via `playbooks/apt-mirror.yml`; re-run it if a host reverts |
| Ansible playbook hangs on "APT prerequisites", lock held | `unattended-upgrades` grabbed the apt/dpkg lock | `sudo systemctl mask --now unattended-upgrades` during the run; unmask after |
| `vgreens-db-backup` stuck "activating" for hours, 0-byte backups | old bug: snapshot written straight to a flaky NAS | fixed — snapshot is local then copied; a `TimeoutStartSec` drop-in prevents hangs |
| Version came out `baseline_only`, won't promote | thresholds weren't published for that run | publish the config (now automatic in CI) and retrain |
| Browser shows "Not secure" on the web UI | Caddy internal CA not trusted | `security add-trusted-cert … caddy-root.crt` (macOS), once per machine |
| A worker went silent | worker crash / host issue | check `systemctl status vgreens-worker@N`, `journalctl -u vgreens-worker@N`; the worker-stopped alert should have fired |
| Job stuck retrying then failed | repeated failure hit the delivery limit | it dead-letters to `failed_jobs`; inspect `job show`, fix the image/config, resubmit |
| NVMe filling up | staged dataset cache growth | LRU eviction handles it; if urgent, clear old entries under `/nvme` staging cache |

---

## Monitoring & alerts

- **Grafana** (on gpu-worker-01): GPU utilization, queue depth, job outcomes,
  worker liveness.
- **Prometheus** scrapes `GET /metrics` on the API + node/GPU exporters. Alert
  rules: worker-stopped, queue backlog, job-failure rate.
- If an alert is noisy during maintenance, silence it in Alertmanager/Grafana for
  the window rather than disabling the rule.

---

## Golden rules

1. The **API is the only control plane** — don't run training by hand on the GPU box.
2. **Datasets and model versions are immutable** — never edit a frozen version;
   make a new one.
3. **Human gates stay manual** — licence approval, promotion, two-GPU training.
4. **Schema changes only via migrations**; `jobs.db` stays on local disk.
5. **Secrets never touch code, logs, or chat.**
