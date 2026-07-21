"""vgreens-ai — command-line client for the vGreens AI platform.

Thin client: every command is an HTTP call to the Job API + formatting. No business
logic lives here. Global options set the profile/output; commands are grouped under
job / model / worker / gpu / system / config, with backward-compatible top-level
aliases (train, process, status, list-jobs, models, promote, rollback, cancel).
"""
from __future__ import annotations

import socket
import sys
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import typer

from . import __version__, config, output
from .client import CLIError, Client, EXIT_GENERIC

app = typer.Typer(add_completion=True, no_args_is_help=True,
                  help="vGreens AI platform CLI. Run `vgreens-ai config init` to get started.")
job_app = typer.Typer(no_args_is_help=True, help="Submit and inspect jobs.")
model_app = typer.Typer(no_args_is_help=True, help="Model registry: versions, promote, rollback.")
worker_app = typer.Typer(no_args_is_help=True, help="GPU workers.")
gpu_app = typer.Typer(no_args_is_help=True, help="GPUs.")
system_app = typer.Typer(no_args_is_help=True, help="Platform health and diagnostics.")
config_app = typer.Typer(no_args_is_help=True, help="CLI configuration and profiles.")
dataset_app = typer.Typer(no_args_is_help=True, help="Datasets (where the server supports it).")
app.add_typer(job_app, name="job")
app.add_typer(model_app, name="model")
app.add_typer(worker_app, name="worker")
app.add_typer(gpu_app, name="gpu")
app.add_typer(system_app, name="system")
app.add_typer(config_app, name="config")
app.add_typer(dataset_app, name="dataset")


@dataclass
class G:
    profile: Optional[str]
    api_url: Optional[str]
    token: Optional[str]
    output: str
    no_color: bool
    quiet: bool
    verbose: bool
    timeout: float


def _version_cb(value: bool):
    if value:
        print(f"vgreens-ai {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    ctx: typer.Context,
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="config profile to use"),
    api_url: Optional[str] = typer.Option(None, "--api-url", help="override the API base URL"),
    token: Optional[str] = typer.Option(None, "--token", help="override the API token"),
    output_fmt: str = typer.Option("table", "--output", "-o", help="table | json | yaml"),
    no_color: bool = typer.Option(False, "--no-color", help="disable colour"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="suppress spinners/messages"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="extra detail"),
    timeout: float = typer.Option(30.0, "--timeout", help="HTTP timeout (s)"),
    _version: bool = typer.Option(False, "--version", callback=_version_cb, is_eager=True,
                                  help="show CLI version and exit"),
):
    if output_fmt not in ("table", "json", "yaml"):
        raise CLIError("--output must be one of: table, json, yaml", 2)
    nc = no_color or not sys.stdout.isatty()   # auto-disable colour when piped
    output.configure(nc)
    ctx.obj = G(profile, api_url, token, output_fmt, nc, quiet, verbose, timeout)


def _client(ctx: typer.Context) -> Client:
    g: G = ctx.obj
    try:
        url, tok, _name = config.resolve(g.profile, g.api_url, g.token)
    except config.ConfigError as exc:
        raise CLIError(str(exc), EXIT_GENERIC)
    return Client(url, tok, timeout=g.timeout, quiet=g.quiet)


# Per-command --output/-o override so `job list -o json` works (in addition to the
# global `-o` before the subcommand). Mutates the shared state read by the helpers.
OUTPUT_OPT = typer.Option(None, "-o", "--output", help="table | json | yaml (overrides global)")


def _mode(ctx: typer.Context, out: Optional[str]) -> None:
    if out:
        if out not in ("table", "json", "yaml"):
            raise CLIError("--output must be one of: table, json, yaml", 2)
        ctx.obj.output = out


# --------------------------------------------------------------------------- #
# core helpers (shared by grouped commands and legacy aliases)
# --------------------------------------------------------------------------- #
JOB_COLS = [("job_id", "job_id"), ("type", "job_type"), ("model", "model_name"),
            ("status", "status"), ("worker", "worker"), ("created", "created_at")]


def _submit(ctx, body: dict):
    g: G = ctx.obj
    c = _client(ctx)
    res = c.post("/jobs", json=body, spinner="submitting job…")
    if g.output == "table":
        output.success(f"queued {res['job_id']}")
    else:
        output.emit(res, g.output)


def _job_train(ctx, model, dataset, image, tenant, config_path):
    name, _, version = (dataset or "").partition(":")
    _submit(ctx, {"job_type": "training", "model_name": model,
                  "dataset_name": name or None, "dataset_version": version or None,
                  "docker_image": image, "tenant": tenant, "config_path": config_path})


def _job_process(ctx, model, image, dataset, tenant, config_path):
    name, _, version = (dataset or "").partition(":") if dataset else ("", "", "")
    _submit(ctx, {"job_type": "processing", "model_name": model,
                  "dataset_name": name or None, "dataset_version": version or None,
                  "docker_image": image, "tenant": tenant, "config_path": config_path})


def _job_list(ctx, status, job_type, model, limit):
    g: G = ctx.obj
    params = {"limit": limit}
    for k, v in (("status", status), ("job_type", job_type), ("model", model)):
        if v:
            params[k] = v
    jobs = _client(ctx).get("/jobs", params=params, spinner="loading jobs…")
    output.emit(jobs, g.output, columns=JOB_COLS, title="jobs")


def _job_show(ctx, job_id):
    g: G = ctx.obj
    output.emit(_client(ctx).get(f"/jobs/{job_id}"), g.output, title=job_id)


def _job_cancel(ctx, job_id):
    g: G = ctx.obj
    res = _client(ctx).post(f"/jobs/{job_id}/cancel", spinner="cancelling…")
    if g.output == "table":
        note = f" ({res['note']})" if res.get("note") else ""
        output.warn(f"{res['status']} {job_id}{note}")
    else:
        output.emit(res, g.output)


# --------------------------------------------------------------------------- #
# job group
# --------------------------------------------------------------------------- #
@job_app.command("train")
def job_train(ctx: typer.Context,
              model: str = typer.Option(..., help="model name"),
              dataset: str = typer.Option(..., help="dataset as name:version"),
              image: str = typer.Option(..., help="docker image"),
              tenant: str = typer.Option(None), config: str = typer.Option(None, "--config")):
    """Submit a training job."""
    _job_train(ctx, model, dataset, image, tenant, config)


@job_app.command("process")
def job_process(ctx: typer.Context,
                model: str = typer.Option(..., help="model name"),
                image: str = typer.Option(..., help="docker image"),
                dataset: str = typer.Option(None, help="images to process, name:version"),
                tenant: str = typer.Option(None), config: str = typer.Option(None, "--config")):
    """Submit a processing/inference job (serves the active model)."""
    _job_process(ctx, model, image, dataset, tenant, config)


@job_app.command("list")
def job_list(ctx: typer.Context,
             status: str = typer.Option(None), job_type: str = typer.Option(None, "--type"),
             model: str = typer.Option(None), limit: int = typer.Option(20),
             out: Optional[str] = OUTPUT_OPT):
    """List recent jobs."""
    _mode(ctx, out)
    _job_list(ctx, status, job_type, model, limit)


@job_app.command("show")
def job_show(ctx: typer.Context, job_id: str = typer.Argument(...), out: Optional[str] = OUTPUT_OPT):
    """Show one job."""
    _mode(ctx, out)
    _job_show(ctx, job_id)


@job_app.command("events")
def job_events(ctx: typer.Context, job_id: str = typer.Argument(...), out: Optional[str] = OUTPUT_OPT):
    """Show a job's event timeline."""
    _mode(ctx, out)
    g: G = ctx.obj
    ev = _client(ctx).get(f"/jobs/{job_id}/events")
    output.emit(ev, g.output, columns=[("time", "ts"), ("event", "event"), ("detail", "detail")],
                title=f"{job_id} events")


@job_app.command("watch")
def job_watch(ctx: typer.Context, job_id: str = typer.Argument(...),
              interval: float = typer.Option(2.0, help="poll interval (s)")):
    """Poll a job until it finishes."""
    g: G = ctx.obj
    c = _client(ctx)
    last = None
    while True:
        job = c.get(f"/jobs/{job_id}")
        st = job["status"]
        if st != last:
            output.info(f"{job_id}: {output.badge(st)}")
            last = st
        if st in ("succeeded", "failed", "cancelled"):
            if g.output != "table":
                output.emit(job, g.output)
            if st == "succeeded":
                raise typer.Exit(0)
            raise CLIError(f"job {st}: {job.get('error') or ''}".strip(), 1)
        time.sleep(interval)


@job_app.command("cancel")
def job_cancel(ctx: typer.Context, job_id: str = typer.Argument(...)):
    """Cancel a job (best-effort)."""
    _job_cancel(ctx, job_id)


# --------------------------------------------------------------------------- #
# model group
# --------------------------------------------------------------------------- #
@model_app.command("list")
def model_list(ctx: typer.Context, out: Optional[str] = OUTPUT_OPT):
    """List models in the registry."""
    _mode(ctx, out)
    g: G = ctx.obj
    res = _client(ctx).get("/models")
    output.emit([{"model": m} for m in res.get("models", [])], g.output, title="models")


@model_app.command("show")
def model_show(ctx: typer.Context, model: str = typer.Argument(...), out: Optional[str] = OUTPUT_OPT):
    """Show a model summary (active version + versions)."""
    _mode(ctx, out)
    g: G = ctx.obj
    m = _client(ctx).get(f"/models/{model}")
    if g.output != "table":
        output.emit(m, g.output)
        return
    output.info(f"[bold]{model}[/]  active: {output.badge('validated') if m['active_version'] else '-'} "
                f"{m['active_version'] or '(none)'}")
    output.emit(m["versions"], g.output,
                columns=[("version", "version"), ("status", "status"), ("active", "active")],
                title=f"{model} versions")


@model_app.command("versions")
def model_versions(ctx: typer.Context, model: str = typer.Argument(...), out: Optional[str] = OUTPUT_OPT):
    """List a model's versions."""
    _mode(ctx, out)
    g: G = ctx.obj
    vs = _client(ctx).get(f"/models/{model}/versions")
    output.emit(vs, g.output, columns=[("version", "version"), ("status", "status"),
                                        ("active", "active")], title=f"{model} versions")


@model_app.command("compare")
def model_compare(ctx: typer.Context, model: str = typer.Argument(...),
                  version_a: str = typer.Argument(...), version_b: str = typer.Argument(...),
                  out: Optional[str] = OUTPUT_OPT):
    """Compare metrics of two versions."""
    _mode(ctx, out)
    g: G = ctx.obj
    c = _client(ctx)
    a = c.get(f"/models/{model}/versions/{version_a}")["metadata"].get("metrics", {})
    b = c.get(f"/models/{model}/versions/{version_b}")["metadata"].get("metrics", {})
    rows = []
    for k in sorted(set(a) | set(b)):
        av, bv = a.get(k), b.get(k)
        delta = round(bv - av, 4) if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else "-"
        rows.append({"metric": k, version_a: av, version_b: bv, "delta": delta})
    output.emit(rows, g.output,
                columns=[("metric", "metric"), (version_a, version_a), (version_b, version_b), ("Δ", "delta")],
                title=f"{model}: {version_a} vs {version_b}")


@model_app.command("promote")
def model_promote(ctx: typer.Context, model: str = typer.Argument(...),
                  version: str = typer.Argument(...), reason: str = typer.Option(None)):
    """Promote a validated version to production."""
    g: G = ctx.obj
    res = _client(ctx).post(f"/models/{model}/promote", json={"version": version, "reason": reason},
                            spinner="promoting…")
    if g.output == "table":
        output.success(f"promoted {model} -> {res['active_version']} (was {res.get('previous_stable') or 'none'})")
    else:
        output.emit(res, g.output)


@model_app.command("rollback")
def model_rollback(ctx: typer.Context, model: str = typer.Argument(...),
                   to: str = typer.Option(None, "--to"), reason: str = typer.Option(None)):
    """Roll production back to the previous (or a given) version."""
    g: G = ctx.obj
    res = _client(ctx).post(f"/models/{model}/rollback", json={"to": to, "reason": reason},
                            spinner="rolling back…")
    if g.output == "table":
        output.warn(f"rolled back {model} -> {res['active_version']}")
    else:
        output.emit(res, g.output)


# --------------------------------------------------------------------------- #
# worker / gpu / system groups
# --------------------------------------------------------------------------- #
@worker_app.command("list")
def worker_list(ctx: typer.Context, out: Optional[str] = OUTPUT_OPT):
    """List GPU workers and their heartbeat."""
    _mode(ctx, out)
    g: G = ctx.obj
    output.emit(_client(ctx).get("/workers"), g.output,
                columns=[("worker", "name"), ("online", "online"),
                         ("since hb (s)", "seconds_since_heartbeat"), ("job", "current_job")],
                title="workers")


@gpu_app.command("list")
def gpu_list(ctx: typer.Context, out: Optional[str] = OUTPUT_OPT):
    """List GPUs (utilisation, memory, temperature)."""
    _mode(ctx, out)
    g: G = ctx.obj
    output.emit(_client(ctx).get("/gpus"), g.output,
                columns=[("idx", "index"), ("name", "name"), ("used MB", "memory_used_mb"),
                         ("total MB", "memory_total_mb"), ("util %", "utilization_pct"),
                         ("temp °C", "temperature_c")], title="gpus")


@system_app.command("version")
def system_version(ctx: typer.Context, out: Optional[str] = OUTPUT_OPT):
    """Show the API version."""
    _mode(ctx, out)
    g: G = ctx.obj
    output.emit(_client(ctx).get("/version"), g.output)


@system_app.command("health")
def system_health(ctx: typer.Context, out: Optional[str] = OUTPUT_OPT):
    """Show platform readiness (db, queue, NAS)."""
    _mode(ctx, out)
    g: G = ctx.obj
    output.emit(_client(ctx).get("/health/ready"), g.output)


@system_app.command("doctor")
def system_doctor(ctx: typer.Context):
    """Diagnose connectivity and auth from THIS machine."""
    g: G = ctx.obj
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        mark = "[green]✓[/]" if passed else "[red]✗[/]"
        output.err.print(f"{mark} {label}" + (f" [dim]{detail}[/]" if detail else ""))
        ok = ok and passed

    # config
    try:
        url, tok, name = config.resolve(g.profile, g.api_url, g.token)
        check("config resolves", True, f"profile={name}")
    except config.ConfigError as exc:
        check("config resolves", False, str(exc))
        raise CLIError("fix your config with `vgreens-ai config init`", 1)

    parsed = urlparse(url)
    check("URL format", bool(parsed.scheme and parsed.netloc), url)

    host = parsed.hostname or ""
    try:
        socket.gethostbyname(host)
        check("DNS resolves", True, host)
    except Exception as exc:  # noqa: BLE001
        check("DNS resolves", False, f"{host}: {exc}")

    c = Client(url, tok, timeout=g.timeout, quiet=True)
    try:
        live = c.get("/health/live")
        check("API reachable", live.get("status") == "alive")
    except CLIError as exc:
        check("API reachable", False, str(exc))
        raise CLIError("API not reachable — check URL / VPN", exc.code)

    try:
        ver = c.get("/version")
        check("API version", True, f"{ver.get('version')} ({ver.get('commit')})")
    except CLIError as exc:
        check("API version", False, str(exc))

    try:
        c.get("/jobs", params={"limit": 1})
        check("authentication", True)
    except CLIError as exc:
        check("authentication", False, str(exc))

    if not ok:
        raise CLIError("some checks failed (see above)", 1)
    output.success("all checks passed")


# --------------------------------------------------------------------------- #
# config group
# --------------------------------------------------------------------------- #
@config_app.command("init")
def config_init(profile: str = typer.Option("prod", help="profile name"),
                api_url: str = typer.Option(None, help="skip the prompt with this URL"),
                token: str = typer.Option(None, help="skip the prompt with this token")):
    """Guided setup: prompt, validate reachability + auth, save securely."""
    url = api_url or typer.prompt("Platform API URL (e.g. https://ai.internal/api)")
    url = url.rstrip("/")
    # validate reachability before asking for a secret
    probe = Client(url, "probe", timeout=10, quiet=True)
    try:
        probe.get("/health/live")
        output.success(f"reachable: {url}")
    except CLIError as exc:
        output.warn(f"could not reach {url}: {exc}")
        if not typer.confirm("Save anyway?", default=False):
            raise typer.Exit(1)
    tok = token or typer.prompt("API token", hide_input=True)
    # test auth
    try:
        Client(url, tok, timeout=10, quiet=True).get("/jobs", params={"limit": 1})
        output.success("authentication OK")
    except CLIError as exc:
        output.error(f"auth check failed: {exc}")
        if not typer.confirm("Save anyway?", default=False):
            raise typer.Exit(1)
    config.save_profile(profile, url, tok, make_default=True)
    output.success(f"saved profile '{profile}' to {config.CONFIG_PATH} (chmod 600). "
                   f"Token stored: {config.redact(tok)}")


@config_app.command("show")
def config_show():
    """Show configured profiles (tokens redacted)."""
    profs = config.profiles()
    if not profs:
        data = config.raw()
        if data.get("api_url"):
            profs = {"default (legacy)": {"api_url": data.get("api_url"), "token": data.get("token", "")}}
    rows = [{"profile": (n + (" *" if n == config.default_profile() else "")),
             "api_url": p.get("api_url"), "token": config.redact(p.get("token", ""))}
            for n, p in profs.items()]
    output.emit(rows or [], "table",
                columns=[("profile", "profile"), ("api_url", "api_url"), ("token", "token")],
                title="profiles (* = default)")


@config_app.command("profiles")
def config_profiles():
    """List profile names."""
    for n in config.profiles():
        marker = " *" if n == config.default_profile() else ""
        output.err.print(f"{n}{marker}")


@config_app.command("use")
def config_use(profile: str = typer.Argument(...)):
    """Set the default profile."""
    try:
        config.set_default(profile)
    except config.ConfigError as exc:
        raise CLIError(str(exc), 1)
    output.success(f"default profile is now '{profile}'")


# --------------------------------------------------------------------------- #
# dataset group — the Dataset Factory (import -> audit -> licence -> freeze)
# --------------------------------------------------------------------------- #
DS_COLS = [("dataset", "dataset_name"), ("versions", "versions"), ("latest_frozen", "latest_frozen_at")]
DSV_COLS = [("version", "version"), ("task", "task"), ("images", "image_count"),
            ("classes", "class_count"), ("licence", "licence_status"),
            ("digest", "content_digest"), ("frozen", "frozen_at")]
IMP_COLS = [("import_id", "import_id"), ("dataset", "dataset_name"), ("source", "source_type"),
            ("state", "state"), ("created", "created_at")]
# states at which `--wait`/`watch` stops (terminal, or the human licence pause)
DATASET_WAIT_STOP = {"frozen", "rejected", "failed", "cancelled", "awaiting_license_approval"}

dataset_import_app = typer.Typer(no_args_is_help=True, help="Import a dataset from a source.")
dataset_app.add_typer(dataset_import_app, name="import")


def _dataset_wait(ctx: typer.Context, import_id: str, interval: float, wait_timeout: float) -> dict:
    c = _client(ctx)
    last = None
    deadline = time.time() + wait_timeout if wait_timeout else None
    while True:
        imp = c.get(f"/datasets/imports/{import_id}")
        st = imp.get("state")
        if st != last:
            output.info(f"{import_id}: {output.badge(st)}")
            last = st
        if st in DATASET_WAIT_STOP:
            if st == "frozen":
                output.success(f"frozen as {imp['dataset_name']}:{imp.get('resolved_version')}")
            elif st == "awaiting_license_approval":
                output.warn(f"awaiting licence approval — an admin must run "
                            f"`vgreens-ai dataset approve-license {import_id}`")
            else:
                raise CLIError(f"import {st}: {imp.get('error') or ''}".strip(), 1)
            return imp
        if deadline and time.time() > deadline:
            raise CLIError(f"timed out waiting for {import_id} (last state: {st})", 1)
        time.sleep(interval)


def _dataset_submit(ctx: typer.Context, body: dict, wait: bool, interval: float, wait_timeout: float):
    g: G = ctx.obj
    body = {k: v for k, v in body.items() if v is not None}
    res = _client(ctx).post("/datasets/imports", json=body, spinner="submitting import…")
    iid = res["import_id"]
    if g.output == "table":
        output.success(f"import queued: {iid} ({res.get('state', 'submitted')})")
    if wait:
        final = _dataset_wait(ctx, iid, interval, wait_timeout)
        if g.output != "table":
            output.emit(final, g.output)
    elif g.output != "table":
        output.emit(res, g.output)


@dataset_import_app.command("roboflow")
def dataset_import_roboflow(
    ctx: typer.Context,
    workspace: str = typer.Option(..., help="Roboflow workspace slug"),
    project: str = typer.Option(..., help="Roboflow project slug"),
    source_version: int = typer.Option(..., "--source-version", help="Roboflow dataset version number"),
    name: str = typer.Option(..., help="frozen dataset name (registry key)"),
    version: Optional[str] = typer.Option(None, "--version", help="explicit frozen version (default: auto vN)"),
    idempotency_key: Optional[str] = typer.Option(None, "--idempotency-key"),
    wait: bool = typer.Option(False, "--wait", help="poll until frozen/rejected/awaiting-licence"),
    interval: float = typer.Option(3.0, help="poll interval (s)"),
    wait_timeout: float = typer.Option(0.0, "--wait-timeout", help="give up after N s (0 = no limit)"),
    out: Optional[str] = OUTPUT_OPT,
):
    """Import a Roboflow dataset (the API key is server-side; never sent from here)."""
    _mode(ctx, out)
    _dataset_submit(ctx, {
        "source_type": "roboflow", "dataset_name": name,
        "source": {"workspace": workspace, "project": project, "rf_version": source_version},
        "requested_version": version, "idempotency_key": idempotency_key,
    }, wait, interval, wait_timeout)


def _import_pathlike(ctx, source_type, path, name, version, idempotency_key, wait, interval, wait_timeout, out):
    _mode(ctx, out)
    _dataset_submit(ctx, {
        "source_type": source_type, "dataset_name": name, "source": {"path": path},
        "requested_version": version, "idempotency_key": idempotency_key,
    }, wait, interval, wait_timeout)


@dataset_import_app.command("local")
def dataset_import_local(
    ctx: typer.Context,
    path: str = typer.Option(..., help="path to a dataset dir or .zip (must be under an allowlisted root)"),
    name: str = typer.Option(...), version: Optional[str] = typer.Option(None, "--version"),
    idempotency_key: Optional[str] = typer.Option(None, "--idempotency-key"),
    wait: bool = typer.Option(False, "--wait"), interval: float = typer.Option(3.0),
    wait_timeout: float = typer.Option(0.0, "--wait-timeout"), out: Optional[str] = OUTPUT_OPT,
):
    """Import a local/NAS dataset directory or zip archive."""
    _import_pathlike(ctx, "local", path, name, version, idempotency_key, wait, interval, wait_timeout, out)


@dataset_import_app.command("nas")
def dataset_import_nas(
    ctx: typer.Context,
    path: str = typer.Option(..., help="path under an allowlisted NAS root"),
    name: str = typer.Option(...), version: Optional[str] = typer.Option(None, "--version"),
    idempotency_key: Optional[str] = typer.Option(None, "--idempotency-key"),
    wait: bool = typer.Option(False, "--wait"), interval: float = typer.Option(3.0),
    wait_timeout: float = typer.Option(0.0, "--wait-timeout"), out: Optional[str] = OUTPUT_OPT,
):
    """Import a dataset from an allowlisted NAS path."""
    _import_pathlike(ctx, "nas", path, name, version, idempotency_key, wait, interval, wait_timeout, out)


@dataset_app.command("list")
def dataset_list(ctx: typer.Context, out: Optional[str] = OUTPUT_OPT):
    """List frozen datasets."""
    _mode(ctx, out)
    g: G = ctx.obj
    res = _client(ctx).get("/datasets")
    output.emit(res.get("datasets", []), g.output, columns=DS_COLS, title="datasets")


@dataset_app.command("versions")
def dataset_versions(ctx: typer.Context, name: str = typer.Argument(...), out: Optional[str] = OUTPUT_OPT):
    """List frozen versions of a dataset."""
    _mode(ctx, out)
    g: G = ctx.obj
    output.emit(_client(ctx).get(f"/datasets/{name}/versions"), g.output,
                columns=DSV_COLS, title=f"{name} versions")


@dataset_app.command("imports")
def dataset_imports(ctx: typer.Context, state: str = typer.Option(None),
                    name: str = typer.Option(None, "--name"), limit: int = typer.Option(20),
                    out: Optional[str] = OUTPUT_OPT):
    """List dataset import workflows."""
    _mode(ctx, out)
    g: G = ctx.obj
    params = {"limit": limit}
    if state:
        params["state"] = state
    if name:
        params["dataset_name"] = name
    output.emit(_client(ctx).get("/datasets/imports", params=params), g.output,
                columns=IMP_COLS, title="dataset imports")


@dataset_app.command("show")
def dataset_show(ctx: typer.Context, import_id: str = typer.Argument(...), out: Optional[str] = OUTPUT_OPT):
    """Show a dataset import workflow with its event timeline."""
    _mode(ctx, out)
    g: G = ctx.obj
    imp = _client(ctx).get(f"/datasets/imports/{import_id}")
    if g.output != "table":
        output.emit(imp, g.output)
        return
    events = imp.pop("events", [])
    output.emit(imp, g.output, title=import_id)
    output.emit(events, g.output,
                columns=[("time", "ts"), ("state", "state"), ("detail", "detail")], title="events")


@dataset_app.command("events")
def dataset_events(ctx: typer.Context, import_id: str = typer.Argument(...), out: Optional[str] = OUTPUT_OPT):
    """Show a dataset import's event timeline."""
    _mode(ctx, out)
    g: G = ctx.obj
    imp = _client(ctx).get(f"/datasets/imports/{import_id}")
    output.emit(imp.get("events", []), g.output,
                columns=[("time", "ts"), ("state", "state"), ("detail", "detail")],
                title=f"{import_id} events")


@dataset_app.command("watch")
def dataset_watch(ctx: typer.Context, import_id: str = typer.Argument(...),
                  interval: float = typer.Option(3.0, help="poll interval (s)"),
                  wait_timeout: float = typer.Option(0.0, "--wait-timeout")):
    """Poll a dataset import until it freezes, is rejected, or awaits licence approval."""
    _dataset_wait(ctx, import_id, interval, wait_timeout)


@dataset_app.command("approve-license")
def dataset_approve_license(ctx: typer.Context, import_id: str = typer.Argument(...),
                            reason: str = typer.Option(None),
                            licence_name: str = typer.Option(None, "--licence-name")):
    """Approve an external dataset's licence (admin) — re-queues it to freeze."""
    g: G = ctx.obj
    res = _client(ctx).post(f"/datasets/imports/{import_id}/approve-license",
                            json={"reason": reason, "licence_name": licence_name},
                            spinner="approving licence…")
    if g.output == "table":
        output.success(f"licence approved for {import_id}; re-queued")
    else:
        output.emit(res, g.output)


@dataset_app.command("reject-license")
def dataset_reject_license(ctx: typer.Context, import_id: str = typer.Argument(...),
                           reason: str = typer.Option(None)):
    """Reject an external dataset's licence (admin) — the import is rejected."""
    g: G = ctx.obj
    res = _client(ctx).post(f"/datasets/imports/{import_id}/reject-license",
                            json={"reason": reason}, spinner="rejecting licence…")
    if g.output == "table":
        output.warn(f"licence rejected for {import_id}")
    else:
        output.emit(res, g.output)


@dataset_app.command("cancel")
def dataset_cancel(ctx: typer.Context, import_id: str = typer.Argument(...)):
    """Cancel a dataset import (best-effort)."""
    g: G = ctx.obj
    res = _client(ctx).post(f"/datasets/imports/{import_id}/cancel", spinner="cancelling…")
    if g.output == "table":
        output.warn(f"{res.get('state', 'cancelled')} {import_id}")
    else:
        output.emit(res, g.output)


# --------------------------------------------------------------------------- #
# backward-compatible top-level aliases
# --------------------------------------------------------------------------- #
@app.command("train", hidden=True)
def _train(ctx: typer.Context, model: str = typer.Option(...), dataset: str = typer.Option(...),
           image: str = typer.Option("vgreens/dummy-job:v1"), tenant: str = typer.Option(None),
           config: str = typer.Option(None, "--config")):
    _job_train(ctx, model, dataset, image, tenant, config)


@app.command("process", hidden=True)
def _process(ctx: typer.Context, model: str = typer.Option(...), image: str = typer.Option(...),
             dataset: str = typer.Option(None), tenant: str = typer.Option(None),
             config: str = typer.Option(None, "--config")):
    _job_process(ctx, model, image, dataset, tenant, config)


@app.command("status", hidden=True)
def _status(ctx: typer.Context, job_id: str = typer.Option(..., "--job-id")):
    _job_show(ctx, job_id)


@app.command("list-jobs", hidden=True)
def _list_jobs(ctx: typer.Context, status: str = typer.Option(None), model: str = typer.Option(None),
               limit: int = typer.Option(20)):
    _job_list(ctx, status, None, model, limit)


@app.command("cancel", hidden=True)
def _cancel(ctx: typer.Context, job_id: str = typer.Option(..., "--job-id")):
    _job_cancel(ctx, job_id)


@app.command("models", hidden=True)
def _models(ctx: typer.Context, model: str = typer.Option(None)):
    if model:
        model_show(ctx, model)
    else:
        model_list(ctx)


@app.command("promote", hidden=True)
def _promote(ctx: typer.Context, model: str = typer.Option(...), version: str = typer.Option(...),
             reason: str = typer.Option(None)):
    model_promote(ctx, model, version, reason)


@app.command("rollback", hidden=True)
def _rollback(ctx: typer.Context, model: str = typer.Option(...), to: str = typer.Option(None, "--to"),
              reason: str = typer.Option(None)):
    model_rollback(ctx, model, to, reason)


def run() -> None:
    """Entry point: wrap the Typer app so CLIError prints cleanly with an exit code."""
    try:
        app()
    except CLIError as exc:
        output.error(str(exc))
        raise SystemExit(exc.code)


if __name__ == "__main__":
    run()
