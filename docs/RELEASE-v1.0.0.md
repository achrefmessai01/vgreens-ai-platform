# vGreens AI Platform — Release v1.0.0

First stable release of the internal, self-hosted GPU MLOps platform. This note summarises what the platform does and what shipped in this milestone across the three repositories (`vgreens-ai-platform`, `vgreens-ai-web`, `vgreens-ai-infra`). Tag each repo `v1.0.0` once the deploy is verified live.

## What the platform is

A single-control-plane MLOps system for training, validating, promoting, and serving computer-vision models on on-prem GPUs. The **Job API** (FastAPI + SQLite) is the only control plane; the CLI, web app, and CI all go through it and never touch the database, broker, Docker, the NAS registry, or Prometheus directly. Work is dispatched via RabbitMQ to systemd worker(s) that run model containers on gpu-worker-01. Datasets flow through an import → audit → licence → freeze pipeline; models flow through train → validate → promote with human gates for licence approval, production promotion, and two-GPU training.

## Highlights in this release

**Secure CI/CD.** GitLab CI pipeline (lint, test, kaniko build, Trivy + Syft security scan, auto-submit training) on a self-hosted group runner, with group-level variables and a read-scoped deploy token. No secrets in job logs.

**Custom-model onboarding.** `model.yaml` spec + presets, `vgreens-ai model init` wizard, `onboard --create-repo` (credential-helper push, no token leakage), headless container generation, and a web onboarding wizard that generates the spec and the exact CLI commands.

**Two-GPU training.** Per-GPU allocation with locks and DDP; two-GPU runs are admin-only and human-gated in both CLI and web.

**Web application (full redesign, D1–D10).** React + Vite SPA behind Caddy over the VPN. In-house design system (tokens, theming with light/dark toggle, reusable primitives), application shell, and redesigned Dashboard, Jobs, Submit, Datasets, Models, Onboarding, Monitoring, and Admin pages. Route-level code-splitting isolates the charting bundle; full keyboard a11y (skip link, focus rings, landmarks, reduced-motion).

**Observability.** Prometheus recording rules + expanded alert catalog, seven provisioned Grafana dashboards, and curated `/system/*` API endpoints (overview, queues, backups, alerts, and a locked-down `timeseries` range endpoint). The web Monitoring page renders live trend charts, workers, GPUs, and firing alerts, and shows degraded/unknown — never falsely healthy — when Prometheus is unreachable.

**Admin console + safe user management.** Web admin UI and CLI `admin` group for users, tokens, and the audit log, with hard guards: you cannot remove the final active admin, cannot disable/delete/anonymise yourself, and deletes are blocked while a user has active sessions/tokens/jobs (with clear conflict messaging). Anonymise preserves audit history.

**Guided CLI.** `vgreens-ai quickstart` (configure → verify → role-tailored next steps) and `whoami`, plus the full `admin` command group.

**Reliability.** Nightly jobs.db backup to the NAS (local snapshot → integrity check → gzip → atomic copy), restore-tested; NVMe LRU eviction; dead-letter handling for repeatedly failing jobs; worker heartbeats with offline alerting.

## Tests

`pytest api/tests/` — 21 passing, covering auth/RBAC, admin token lifecycle, and the safety-critical guards added this milestone: self-protection, last-admin protection, delete conflict codes, disable-revokes-sessions, anonymise, `gpu_count` validation, and the `/system/timeseries` allowlist. CLI has unit tests for the model spec, generator, and dataset commands.

## Security posture

API is the sole control plane. RBAC (viewer/operator/admin) is enforced server-side; the browser only hides what a role can't do. Session cookies are HttpOnly; no tokens in browser storage. Human gates (licence approval, production promotion, two-GPU) are preserved. The browser never issues arbitrary PromQL — monitoring goes through fixed, curated queries. Secrets (.env, tokens, registry/NAS/SSH credentials, Vault values) are never exposed by any endpoint.

## Known follow-ups

- **PFE screenshots** of the live UI (D12).
- **1-GPU vs 2-GPU benchmark** numbers — runbook is ready in `docs/benchmark-1v2-gpu.md`; awaiting the strawberry training image build.
- ADRs and per-failure operator runbooks (nice-to-have).

## Deploy / tag checklist

1. Push all three repos; pull + `systemctl restart vgreens-ai-api` on gpu-worker-01.
2. `npm run build` + `ansible-playbook playbooks/reverse-proxy.yml -K` for the web.
3. `pipx install --force ~/Desktop/vgreens-ai-platform/cli`.
4. Verify `https://app.vgreens.lan:9443` loads and Monitoring trends populate.
5. `git tag -a v1.0.0 -m "vGreens AI platform v1.0.0"` in each repo, then `git push --tags`.
