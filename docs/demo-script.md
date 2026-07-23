# vGreens AI Platform — PFE Defense Demo Script

A tight, ~15-minute live demonstration that tells one story: **"a new model goes
from an idea to a governed, promoted service with one command and no manual server
work."** Everything else (datasets, training, the web UI, observability) hangs off
that spine.

> Golden rule: **rehearse it once end-to-end the day before**, and pre-bake the
> slow parts (see Pre-flight). During the defense you click and narrate — you never
> wait on a cold CI build or a slow mirror.

---

## Pre-flight (do this 30–60 min before, not live)

- [ ] **VPN connected**; confirm `curl https://vgreens-ai.lan:9443/health/live` returns `alive` and `https://app.vgreens.lan:9443` loads.
- [ ] Services up on gpu-worker-01: `systemctl is-active vgreens-ai-api vgreens-worker@0 vgreens-worker@1 vgreens-dataset-worker`.
- [ ] `unattended-upgrades` masked on sv01 (so no apt lock stalls a live pipeline).
- [ ] CLI ready: `vgreens-ai list-jobs` works (profile `prod`, `REQUESTS_CA_BUNDLE` exported).
- [ ] Browser: logged into the web UI in one tab; GitLab open in another (the `vgreens-models` group).
- [ ] `GITLAB_TOKEN` exported in your demo terminal (for `--create-repo`).
- [ ] **Pre-warm the demo model** so the build isn't cold: run the whole onboarding once the day before (`demo_ripeness2` or similar), so the base image layers and runner are cached and a fresh pipeline is fast.
- [ ] Have a **fallback screen recording** of a full green pipeline + a promoted model, in case the network misbehaves.
- [ ] Terminal font large; only the panes you'll show are open.

---

## Timeline

| Min | Act | What you show |
|-----|-----|---------------|
| 0–1 | Framing | The problem + one-sentence solution |
| 1–4 | Web UI | Dashboard, jobs, models, datasets |
| 4–6 | Dataset Factory | An immutable, audited, versioned dataset |
| 6–12 | **Onboard a model** | init → create-repo → CI → train → promote |
| 12–14 | Governance & ops | thresholds, promotion gate, RBAC, Grafana, backups |
| 14–15 | Close | Recap the design principles |

---

## Act 0 — Framing (1 min, talk only)

> "vGreens has GPUs and computer-vision models but no consistent way to train,
> version, and serve them. I built an internal MLOps platform so that any engineer
> can onboard a model from a single spec, training runs as a governed job in an
> isolated container, and no model becomes 'active' until it passes validation.
> Everything goes through one API; nobody SSHes to the GPU server."

Show the **architecture diagram** (one slide) for 20 seconds, then go live.

---

## Act 1 — The web UI (3 min)

Open `https://app.vgreens.lan:9443`.

1. **Dashboard** — GPUs, workers, recent jobs. "This is served over HTTPS through a
   reverse proxy on the VPN; the GPU box has no public exposure."
2. **Jobs** — filter, open one, show the timeline of events. "Every job is tracked
   with an event history and an immutable snapshot on the NAS."
3. **Models** — open a model, show versions and the **active** one. "A version is
   only promotable if it cleared its thresholds."
4. **Datasets** — show frozen versions with manifests.

Talking point: "The CLI, this web UI, and CI all use the **same API** — one control
plane, three front-ends."

---

## Act 2 — Dataset Factory (2 min)

Show that data is governed before use. Either show the already-frozen
`demo_berries:v1`, or freeze a tiny one live (pre-generated on the box):

```bash
vgreens-ai dataset import local \
  --path /mnt/vgreens_storage/ai/incoming/demo-berries \
  --name demo_berries2 --version v1 --wait
vgreens-ai dataset versions demo_berries2
```

Narrate while it runs: "It stages the data off the source, **audits** it (labels,
boxes, train/val leakage), normalizes to a canonical layout, and **atomically
freezes** an immutable version on the NAS with a manifest and checksums. A frozen
`dataset:vN` never changes — training is always reproducible."

---

## Act 3 — Onboard a model end-to-end (6 min) — the centrepiece

This is the story. Keep narrating; the tooling does the work.

**1. Declare the model (interactive):**
```bash
vgreens-ai model init
```
Answer: name `demo_defense`, task `custom`, preset `custom_python`, dataset
`demo_berries`/`v1`, gpu 1, metric `accuracy`/`0.5`, defaults for the rest.
> "One declarative `model.yaml` — the single source of truth for this model."

**2. Generate + create the repo + push, one command:**
```bash
vgreens-ai model onboard --spec model.yaml --create-repo
```
> "That scaffolded a complete, contract-compliant repo, created the GitLab project
> **inside the group** — so it inherits the CI runner, the CI variables, and the
> shared pipeline — committed, and pushed. A pipeline is already starting."

**3. Show the pipeline in GitLab** (`vgreens-models/vgreens-model-demo-defense`
→ Build → Pipelines). Walk the stages: `lint → test → build (kaniko) → Trivy +
SBOM → submit_training`.
> "CI builds the image **unprivileged** with kaniko — no Docker daemon on the
> shared host — scans it, and produces an SBOM. CI never trains or touches the GPU;
> it only calls the API."

**4. Click ▶ `submit_training`.** Show the log: it publishes the config, then POSTs
the job.
> "One manual click — a deliberate human gate — publishes the config and enqueues
> training through the API."

**5. Watch it train and promote:**
```bash
vgreens-ai list-jobs
vgreens-ai model show demo_defense
vgreens-ai model promote demo_defense <version>
```
> "The worker pulled the exact built image, ran it in isolation on a GPU, the
> result cleared the 0.5 threshold so it's **validated**, and I promote it — an
> audited, admin-only action. It's now the active model."

*(Fallback: if the live build is slow, switch to the pre-warmed model from
pre-flight, or the screen recording. The narration is identical.)*

---

## Act 4 — Governance & operations (2 min)

Pick two or three, fast:

- **Validation gate:** show a version that came out `baseline_only` (below
  threshold) and can't be promoted. "The platform refuses to promote an unvalidated
  model."
- **RBAC:** "viewer / operator / admin — reads, job submission, and promotion are
  separated." Show the Admin page (as admin).
- **Audit:** `vgreens-ai` audit / the events timeline. "Promotions, cancellations,
  and logins are all recorded."
- **Observability:** open Grafana — GPU utilization, queue depth, job outcomes;
  mention the worker-stopped alert.
- **Resilience:** "Nightly `jobs.db` backups to the NAS, verified restorable;
  two-GPU training behind a human gate; failing jobs dead-letter after a retry
  limit."

---

## Act 5 — Close (1 min, talk only)

> "The design rests on a few principles: **one control plane** (the API);
> **immutability** (datasets and model versions never change); **isolation**
> (every job in its own container under the same contract); and **human gates** on
> the risky actions — licence approval, promotion, multi-GPU. A new model joins
> through one spec and one command, and the whole system is reproducible from four
> Git repositories and Ansible."

Invite questions.

---

## Likely questions — quick answers

- **Why not Kubernetes?** On-prem, two GPUs, a small team — systemd + a queue is
  simpler to run and reason about; the container contract keeps the door open to
  an orchestrator later.
- **How is a model isolated?** Its own image, read-only data/config mounts, a
  writable output dir, and only the GPUs it was granted (per-GPU locks).
- **What stops a bad model going live?** Threshold validation + a manual,
  admin-only, audited promotion.
- **Secrets?** Ansible Vault, masked CI variables, a read-only registry token,
  server-side API keys — never in code or logs.
- **If the NAS goes down?** Jobs fail fast (they don't hang); `jobs.db` is on local
  disk; backups are verified.

---

## Reset after the demo

```bash
# remove the throwaway model version/dataset if you created them live
vgreens-ai model rollback demo_defense <previous-or-none>   # if needed
# delete the demo GitLab project from the group UI
# (optional) drop demo_berries2 import artifacts under ai/incoming/
```
