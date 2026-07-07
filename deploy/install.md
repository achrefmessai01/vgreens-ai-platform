# Deployment — gpu-worker-01 (172.16.20.89)

All commands run on **gpu-worker-01** as user `vgreens` unless noted. `sudo` where shown.

---

## Step 1.1 — /opt/vgreens-ai + NAS folders

```bash
# 1. deploy dir (owned by vgreens so the venv/services run without root)
sudo mkdir -p /opt/vgreens-ai
sudo chown vgreens:vgreens /opt/vgreens-ai

# 2. clone the repo into /opt/vgreens-ai/repo
cd /opt/vgreens-ai
git clone <gitlab-url>/vgreens-ai-platform.git repo

# 3. NAS platform folders (idempotent — already created in Phase 0, safe to re-run)
mkdir -p /mnt/vgreens_storage/ai/{datasets,model_registry,jobs,processing}
```

**Done when:** `ls /opt/vgreens-ai/repo` shows the tree and
`ls /mnt/vgreens_storage/ai` shows the four folders.

---

## Step 1.2 — RabbitMQ

```bash
# 1. create the root-only env file from the example, then edit the passwords
sudo cp /opt/vgreens-ai/repo/deploy/.env.example /opt/vgreens-ai/.env
sudo chown root:root /opt/vgreens-ai/.env
sudo chmod 600 /opt/vgreens-ai/.env
sudo nano /opt/vgreens-ai/.env      # set RABBITMQ_PASSWORD + TOKENS (long alphanumeric)

# 2. bring up the broker (compose reads ../.env via --env-file)
cd /opt/vgreens-ai/repo/deploy
docker compose --env-file /opt/vgreens-ai/.env up -d

# 3. check health
docker compose ps
docker compose --env-file /opt/vgreens-ai/.env logs rabbitmq | tail -n 20
```

Expected: `docker compose ps` shows `vgreens-rabbitmq` as `running (healthy)`.
In a browser on the LAN: `http://172.16.20.89:15672` → login with the `.env`
credentials → the management UI loads. No queues yet (they appear when the API starts).

**Restart-survival check:**

```bash
docker compose --env-file /opt/vgreens-ai/.env restart
docker compose ps          # healthy again; named volume kept the state
```

**Firewall (keep 15672 off other subnets):**

```bash
# allow the LAN subnet only; adjust to your actual LAN CIDR
sudo ufw allow from 172.16.20.0/24 to any port 15672 proto tcp
sudo ufw allow from 172.16.20.0/24 to any port 5672 proto tcp
# (do NOT `ufw allow 15672` unqualified — that opens it to everything)
```

**Done when:** UI reachable on the LAN, broker survives `docker compose restart`,
15672 not reachable from outside the LAN.

Pitfalls: credentials never in git (they live only in `/opt/vgreens-ai/.env`);
don't expose 15672 publicly.

---

## Later steps (reference — not this session)

- **1.3 API:** create venv, `pip install -r repo/api/requirements.txt`, install
  `vgreens-ai-api.service`, `curl localhost:8000/health`.
- **1.5 Worker:** `pip install -r repo/worker/requirements.txt`, install
  `vgreens-worker@.service`, `systemctl enable --now vgreens-worker@0`.
- **1.6 Dummy job:** `docker build -t vgreens/dummy-job:v1 repo/dummy-job/`.

```bash
# venv (used by both API and worker) — for reference
python3 -m venv /opt/vgreens-ai/venv
/opt/vgreens-ai/venv/bin/pip install -r /opt/vgreens-ai/repo/api/requirements.txt
/opt/vgreens-ai/venv/bin/pip install -r /opt/vgreens-ai/repo/worker/requirements.txt
```
