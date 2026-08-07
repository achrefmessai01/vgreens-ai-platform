"""vgreens-ai — command-line client for the vGreens AI platform.

Thin client: every command is an HTTP call to the Job API + formatting. No business
logic lives here. Global options set the profile/output; commands are grouped under
job / model / worker / gpu / system / config, with backward-compatible top-level
aliases (train, process, status, list-jobs, models, promote, rollback, cancel).
"""
from __future__ import annotations

import json
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import typer

from . import __version__, config, output, modelspec, modelgen
from .client import CLIError, Client, EXIT_GENERIC

app = typer.Typer(
    add_completion=True, no_args_is_help=True, rich_markup_mode="rich",
    help=(
        "[bold]vGreens AI platform[/] — command-line client for datasets, models, "
        "training/inference jobs and production control.\n\n"
        "Everything goes through the Job API (the single control plane); this CLI never "
        "touches the GPU box or the NAS directly. [bold]New here?[/] run "
        "[cyan]vgreens-ai quickstart[/] (or [cyan]config init[/]) first.\n\n"
        "[bold]Command groups[/]\n"
        "  [cyan]dataset[/]  import & freeze training data       [cyan]job[/]     submit/track training & inference\n"
        "  [cyan]model[/]    versions, compare, promote, delete  [cyan]worker[/]  GPU worker health\n"
        "  [cyan]gpu[/]      live GPU utilisation                [cyan]system[/]  API health & diagnostics\n"
        "  [cyan]config[/]   profiles & credentials              [cyan]admin[/]   users, tokens, audit (admin)\n\n"
        "[bold]Typical flow[/]\n"
        "  [dim]1. setup   [/] vgreens-ai config init  →  vgreens-ai whoami\n"
        "  [dim]2. data    [/] vgreens-ai dataset import roboflow --dataset berries ...  →  dataset watch <id>\n"
        "  [dim]3. train   [/] vgreens-ai job train --model vgreens_demo_detector --dataset demo_berries:v1\n"
        "  [dim]4. results [/] vgreens-ai job watch <id>  →  job result <id>  →  job download <id> --path model.pt\n"
        "  [dim]5. release [/] vgreens-ai model promote <model> <version>  →  job process --model <model>\n\n"
        "[bold]Tips[/]  add [cyan]-o json[/]/[cyan]-o yaml[/] to any read command for scripting · "
        "[cyan]--profile prod[/] switches environments · exit codes: 0 ok, 3 network, 4 auth, 5 not-found, 6 conflict."
    ),
)
job_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich",
    help="Submit and track jobs — [cyan]train[/], [cyan]process[/], [cyan]watch[/], "
         "[cyan]result[/], [cyan]logs[/], [cyan]download[/], [cyan]cancel[/].")
model_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich",
    help="Model registry — [cyan]list/show/versions[/], [cyan]compare[/], "
         "[cyan]promote[/]/[cyan]rollback[/] (production control), [cyan]onboard[/], [cyan]delete[/].")
worker_app = typer.Typer(no_args_is_help=True, help="GPU workers — heartbeat and current job.")
gpu_app = typer.Typer(no_args_is_help=True, help="GPUs — live utilisation, memory and temperature.")
system_app = typer.Typer(no_args_is_help=True, help="Platform health and diagnostics (version, health, doctor).")
config_app = typer.Typer(no_args_is_help=True, help="CLI configuration and profiles (API URL + token per environment).")
dataset_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich",
    help="Datasets — [cyan]import[/] (roboflow/local/nas), [cyan]freeze[/] tracking, "
         "list versions, licence approval.")
admin_app = typer.Typer(no_args_is_help=True, help="Administration: users, tokens, audit (admin only).")
images_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich",
    help="Container images — [cyan]resolve[/] a short ref to a full URL, "
         "[cyan]validate[/] a ref, [cyan]suggest[/] images for a model, show registry [cyan]config[/]. "
         "You can pass short refs like [cyan]--image strawberry-detector:v1.0.2[/] to job train/process.")
templates_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich",
    help="Job templates — reusable submit presets ([cyan]list[/], [cyan]show[/], [cyan]rm[/]). "
         "Start a job from one with [cyan]job create --template <id>[/].")
init_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich",
    help="Scaffold starter projects — [cyan]training-project[/] creates a ready-to-edit "
         "training repo (Dockerfile, train.py, config.yaml, job template).")
app.add_typer(job_app, name="job")
app.add_typer(model_app, name="model")
app.add_typer(worker_app, name="worker")
app.add_typer(gpu_app, name="gpu")
app.add_typer(system_app, name="system")
app.add_typer(config_app, name="config")
app.add_typer(dataset_app, name="dataset")
app.add_typer(admin_app, name="admin")
app.add_typer(images_app, name="images")
app.add_typer(templates_app, name="templates")
app.add_typer(init_app, name="init")

tutorial_app = typer.Typer(no_args_is_help=True, rich_markup_mode="rich",
    help="Guided tutorials for AI engineers (first run, training, inference, GPUs, outputs, troubleshooting).")
app.add_typer(tutorial_app, name="tutorial")


def _tut(title: str, body: str) -> None:
    """Print a tutorial page. Uses rich if available, else plain text."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        c = Console()
        c.print(Panel.fit(body.strip(), title=f"[bold]{title}[/]", border_style="green"))
    except Exception:  # noqa: BLE001 — tutorials must never crash the CLI
        print(f"\n=== {title} ===\n{body.strip()}\n")


@tutorial_app.command("overview")
def tutorial_overview() -> None:
    """What the platform is and how a job flows through it."""
    _tut("vGreens AI Platform — overview", """
The platform runs your AI training and processing jobs on a shared GPU server
(gpu-worker-01) instead of your laptop. You submit a job; the platform queues it,
runs it in a Docker container on a GPU, stores outputs on the NAS, and shows it live.

Flow:
  you (CLI / web)  ->  Job API  ->  RabbitMQ queue  ->  GPU worker
                    ->  Docker + NVIDIA runtime  ->  NAS (/mnt/vgreens_storage)
                    ->  model registry  +  live monitoring

Learn each part:
  vgreens-ai tutorial first-run        first-time setup
  vgreens-ai tutorial train            submit a training job
  vgreens-ai tutorial inference        submit an inference/processing job
  vgreens-ai tutorial gpu              1 GPU vs 2 GPUs (multi-GPU rules)
  vgreens-ai tutorial outputs          where results are written
  vgreens-ai tutorial troubleshooting  common problems and fixes
""")


@tutorial_app.command("first-run")
def tutorial_first_run() -> None:
    """First-time setup: configure, connect, check GPUs and status."""
    _tut("First run", """
1. Configure the CLI (stores API URL + token in a local profile; token never printed):
     vgreens-ai config init
     vgreens-ai whoami            # confirm who you are and your role

2. Check the platform is reachable and healthy:
     vgreens-ai system doctor
     vgreens-ai gpu list          # see available GPUs and utilisation

3. You are ready. Next: 'vgreens-ai tutorial train'.

Roles: viewer (read), operator (submit/cancel jobs, import datasets),
admin (promote/rollback, manage users, request 2 GPUs).
""")


@tutorial_app.command("train")
def tutorial_train() -> None:
    """How to submit a training job (single- and multi-GPU)."""
    _tut("Submit a training job", """
Single GPU (the common case):
  vgreens-ai job train \\
    --model strawberry-detector \\
    --dataset strawberry_disease:v1 \\
    --image registry.vgreens.lan:9443/vgreens-models/strawberry-detector:latest \\
    --config /mnt/vgreens_storage/ai/job-configs/strawberry-detector.yaml \\
    --gpu-count 1

Two GPUs (admin only):
  vgreens-ai job train --model strawberry-detector --dataset strawberry_disease:v1 --gpu-count 2

IMPORTANT — multi-GPU: requesting 2 GPUs only *exposes* both GPUs to the container.
Both are actually used only if the training code runs a distributed strategy (DDP).
Models generated by 'vgreens-ai model onboard' relaunch training under torchrun
automatically, so they use both GPUs. A custom image must start its own
multi-process training (torchrun / DDP) or it will use only 1 GPU.
See 'vgreens-ai tutorial gpu'.

Follow it:
  vgreens-ai job watch <job-id>
  vgreens-ai job result <job-id>
""")


@tutorial_app.command("inference")
def tutorial_inference() -> None:
    """How to submit an inference / processing job."""
    _tut("Submit an inference (processing) job", """
Processing jobs serve the ACTIVE promoted model for a given model name:
  vgreens-ai job process \\
    --model strawberry-counter \\
    --dataset incoming_frutania:v1 \\
    --image registry.vgreens.lan:9443/vgreens-models/strawberry-counter:latest \\
    --gpu-count 1

The container reads the active model at /model, reads /data, and writes
/out/report.json (images, total detections, per-class counts). Retrieve results:
  vgreens-ai job result <job-id>
""")


@tutorial_app.command("gpu")
def tutorial_gpu() -> None:
    """1 GPU vs 2 GPUs — what the platform does and what your code must do."""
    _tut("GPU resources & multi-GPU", """
What the platform does when you request N GPUs:
  - reserves N GPU locks so no other job uses them at the same time;
  - runs the container with exactly those devices (--gpus "device=0,1" for 2);
  - sets GPU_COUNT in the container environment.

What your training code must do to actually use 2 GPUs:
  - run a distributed strategy (PyTorch DDP / torchrun, TF MirroredStrategy, etc.).
  - Models from 'vgreens-ai model onboard' do this automatically: the entrypoint
    relaunches train.py under 'torchrun --nproc_per_node=$GPU_COUNT'.
  - A single-process script will see 2 GPUs but only use cuda:0 (1 GPU).

Verify on the GPU host (gpu-worker-01):
  nvidia-smi -L
  docker run --rm --gpus '"device=0,1"' nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

Verify inside a training container:
  python -c "import torch; print(torch.cuda.device_count())"   # expect 2

Safety: a 2-GPU job waits in the queue until BOTH GPUs are free — it is not run on
one GPU and it is not failed immediately.
""")


@tutorial_app.command("outputs")
def tutorial_outputs() -> None:
    """Where results are written and how they are structured."""
    _tut("Outputs", """
Every job gets a unique output folder on the NAS (the platform picks the path):
  /mnt/vgreens_storage/ai/jobs/<year>/<job-id>/
    out/
      model.pt                 (training)
      eval/metrics.json        (training metrics the validation gate reads)
      evaluation_report.md
      report.json              (processing: detections / counts)
    logs/container.log

Get results without touching the NAS:
  vgreens-ai job result <job-id>
  vgreens-ai job logs <job-id>
  vgreens-ai job download <job-id> --path model.pt
""")


@tutorial_app.command("troubleshooting")
def tutorial_troubleshooting() -> None:
    """Common problems and how to resolve them."""
    _tut("Troubleshooting", """
Job stuck in 'queued'      -> no worker free, or a 2-GPU job waiting for both GPUs;
                              check 'vgreens-ai worker list' and 'vgreens-ai gpu list'.
GPU unavailable            -> both GPUs busy; the job waits (it is not failed).
NAS not mounted            -> job rejected fast; check the NAS mount on gpu-worker-01.
Docker image pull failed   -> check the image ref and that CI published it.
CUDA / out-of-memory       -> lower training.batch_size or image_size (raise grad
                              accumulation); the error names VRAM.
Output folder missing      -> the run did not reach exit 0; read 'job logs <id>'.
2 GPUs requested, 1 used    -> training code is single-process; use an onboarded
                              model or add torchrun/DDP. See 'tutorial gpu'.
Monitoring disconnected     -> the web dashboard reconnects automatically; if not,
                              the API/stream may be down — check 'system doctor'.
Worker offline             -> 'vgreens-ai worker list' shows heartbeat; restart the
                              worker service on gpu-worker-01 if stale.
""")


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


def _job_train(ctx, model, dataset, image, tenant, config_path, gpu_count=1):
    name, _, version = (dataset or "").partition(":")
    _submit(ctx, {"job_type": "training", "model_name": model,
                  "dataset_name": name or None, "dataset_version": version or None,
                  "docker_image": image, "tenant": tenant, "config_path": config_path,
                  "gpu_count": gpu_count})


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
              image: Optional[str] = typer.Option(None, help="docker image (optional; resolved from the model registry if omitted)"),
              tenant: str = typer.Option(None), config: str = typer.Option(None, "--config"),
              gpu_count: int = typer.Option(1, "--gpu-count", help="1 or 2 (2 = both GPUs, admin only)"),
              yes: bool = typer.Option(False, "--yes", help="skip the two-GPU confirmation")):
    """Submit a training job."""
    if gpu_count == 2:
        typer.secho(
            "Note: 2 GPUs are exposed to the container, but they are only used if the "
            "training code runs distributed (DDP). Models generated by 'vgreens-ai "
            "model onboard' launch multi-GPU automatically; a custom image must start "
            "its own multi-process training or it will use only 1 GPU. "
            "See: vgreens-ai tutorial gpu",
            fg=typer.colors.YELLOW)
        if not yes and sys.stdout.isatty():
            typer.confirm("Two-GPU training uses BOTH GPUs — no GPU stays free for "
                          "production while it runs. Continue?", abort=True)
    _job_train(ctx, model, dataset, image, tenant, config, gpu_count)


@job_app.command("process")
def job_process(ctx: typer.Context,
                model: str = typer.Option(..., help="model name"),
                image: Optional[str] = typer.Option(None, help="docker image (optional; resolved from the model registry if omitted)"),
                dataset: str = typer.Option(None, help="images to process, name:version"),
                tenant: str = typer.Option(None), config: str = typer.Option(None, "--config")):
    """Submit a processing/inference job (serves the active model)."""
    _job_process(ctx, model, image, dataset, tenant, config)


# --------------------------------------------------------------------------- #
# images group — resolve short refs so you never paste a full registry URL
# --------------------------------------------------------------------------- #
@images_app.command("config")
def images_config(ctx: typer.Context, out: Optional[str] = OUTPUT_OPT):
    """Show the registry base and namespace used to expand short image refs."""
    _mode(ctx, out)
    g: G = ctx.obj
    output.emit(_client(ctx).get("/registry/config"), g.output, title="registry")


@images_app.command("resolve")
def images_resolve(ctx: typer.Context,
                   ref: str = typer.Argument(..., help="short ref, e.g. strawberry-detector:v1.0.2"),
                   model: Optional[str] = typer.Option(None, help="model, so a bare ':tag' resolves to that model's repo"),
                   out: Optional[str] = OUTPUT_OPT):
    """Preview the full image URL a short ref expands to (no job submitted)."""
    _mode(ctx, out)
    g: G = ctx.obj
    params = {"ref": ref}
    if model:
        params["model"] = model
    res = _client(ctx).get("/registry/resolve", params=params, spinner="resolving…")
    if g.output == "table":
        output.success(res["resolved"] or "(blank — API resolves from the model registry)")
    else:
        output.emit(res, g.output)


@images_app.command("suggest")
def images_suggest(ctx: typer.Context,
                   model: str = typer.Argument(..., help="model name"),
                   job_type: str = typer.Option("training", "--type", help="training or processing"),
                   out: Optional[str] = OUTPUT_OPT):
    """Suggest images for a model: its registry default plus recently-used images."""
    _mode(ctx, out)
    g: G = ctx.obj
    res = _client(ctx).get("/registry/suggest",
                           params={"model": model, "job_type": job_type},
                           spinner="loading suggestions…")
    if g.output == "table":
        output.success(f"default: {res.get('default_image') or '(resolves at submit time)'}")
        recent = res.get("recent_images") or []
        if recent:
            output.emit([{"recent_image": r} for r in recent], "table",
                        columns=[("recent images", "recent_image")])
        output.info(f"tip: vgreens-ai job train --model {model} "
                    f"--image {res.get('suggested_ref', model + ':latest')} --dataset <name>:<ver>")
    else:
        output.emit(res, g.output)


@images_app.command("validate")
def images_validate(ctx: typer.Context,
                    ref: str = typer.Argument(..., help="image ref, e.g. strawberry-detector:v1.0.2"),
                    model: Optional[str] = typer.Option(None, help="model for a bare ':tag'"),
                    out: Optional[str] = OUTPUT_OPT):
    """Resolve a ref and sanity-check it before you launch a long job (#3)."""
    _mode(ctx, out)
    g: G = ctx.obj
    params = {"ref": ref}
    if model:
        params["model"] = model
    res = _client(ctx).get("/registry/resolve", params=params, spinner="resolving…")
    resolved = res.get("resolved") or ""
    # Reuse the API's planning validator via /jobs/plan is overkill; do a light
    # local check here and show the resolved URL.
    if g.output == "table":
        if not resolved:
            output.warn("blank — the API would resolve this from the model registry")
        else:
            output.success(f"resolved: {resolved}")
            if ":" not in resolved.rsplit("/", 1)[-1]:
                output.warn("no explicit tag — will default to :latest")
    else:
        output.emit({"ref": ref, "resolved": resolved}, g.output)


# --------------------------------------------------------------------------- #
# templates group (#1): reusable submit presets
# --------------------------------------------------------------------------- #
TPL_COLS = [("id", "id"), ("name", "name"), ("type", "job_type"),
            ("image", "docker_image"), ("gpu", "gpu_count"), ("dataset", "dataset_name")]


@templates_app.command("list")
def templates_list(ctx: typer.Context, job_type: str = typer.Option(None, "--type"),
                   out: Optional[str] = OUTPUT_OPT):
    """List job templates."""
    _mode(ctx, out)
    g: G = ctx.obj
    params = {"job_type": job_type} if job_type else {}
    res = _client(ctx).get("/templates", params=params, spinner="loading templates…")
    output.emit(res.get("templates", []), g.output, columns=TPL_COLS, title="templates")


@templates_app.command("show")
def templates_show(ctx: typer.Context, template_id: str = typer.Argument(...),
                   out: Optional[str] = OUTPUT_OPT):
    """Show one template."""
    _mode(ctx, out)
    g: G = ctx.obj
    output.emit(_client(ctx).get(f"/templates/{template_id}"), g.output, title=template_id)


@templates_app.command("rm")
def templates_rm(ctx: typer.Context, template_id: str = typer.Argument(...)):
    """Delete a template (built-ins cannot be deleted)."""
    g: G = ctx.obj
    res = _client(ctx).delete(f"/templates/{template_id}", spinner="deleting…")
    if g.output == "table":
        output.success(f"deleted {template_id}")
    else:
        output.emit(res, g.output)


@templates_app.command("create")
def templates_create(ctx: typer.Context,
                     name: str = typer.Option(..., help="template name"),
                     job_type: str = typer.Option("training", "--type"),
                     image: Optional[str] = typer.Option(None, help="short ref or full URL"),
                     gpu_count: int = typer.Option(1, "--gpu-count"),
                     dataset: Optional[str] = typer.Option(None, help="default dataset name"),
                     description: Optional[str] = typer.Option(None)):
    """Create a job template from flags."""
    g: G = ctx.obj
    body = {"name": name, "job_type": job_type, "docker_image": image,
            "gpu_count": gpu_count, "dataset_name": dataset, "description": description}
    res = _client(ctx).post("/templates", json=body, spinner="creating template…")
    if g.output == "table":
        output.success(f"created {res.get('id')}")
    else:
        output.emit(res, g.output)


# --------------------------------------------------------------------------- #
# init training-project (#16): scaffold a ready-to-edit training repo
# --------------------------------------------------------------------------- #
_INIT_TRAIN_PY = '''\
"""Training entrypoint — platform contract.

Reads /data (read-only) + /config/job.yaml, writes /out/model.pt,
/out/eval/metrics.json and one JSON line per epoch to /out/metrics.jsonl
(the platform shows live progress from that file). Replace the marked block
with your real model + training loop.
"""
import json, os, time
from pathlib import Path

OUT = Path("/out"); DATA = Path("/data"); CONFIG = Path("/config/job.yaml")


def _progress(**row):
    row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(OUT / "metrics.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\\n")


def main() -> int:
    for d in ("eval", "artifacts", "predictions"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    cfg = {}
    if CONFIG.is_file():
        try:
            import yaml; cfg = yaml.safe_load(CONFIG.read_text()) or {}
        except Exception:
            cfg = {}
    epochs = int((cfg.get("train") or {}).get("epochs", 10))

    # ===== REPLACE: your real training loop =====
    accuracy = 0.0
    for ep in range(1, epochs + 1):
        accuracy = ep / epochs
        _progress(epoch=ep, total_epochs=epochs, accuracy=round(accuracy, 4))
    (OUT / "model.pt").write_bytes(b"REPLACE_WITH_REAL_WEIGHTS")
    # ============================================

    (OUT / "eval" / "metrics.json").write_text(json.dumps({"accuracy": round(accuracy, 4)}))
    (OUT / "report.json").write_text(json.dumps(
        {"model": os.environ.get("MODEL_NAME", "model"), "epochs": epochs,
         "metrics": {"accuracy": round(accuracy, 4)}}, indent=2))
    print(f"[train] done, accuracy={accuracy}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_INIT_DOCKERFILE = '''\
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "/app/train.py"]
'''

_INIT_CONFIG = '''\
# Hyperparameters read by train.py (edit freely).
train:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  image_size: 640
  gpus: 1
thresholds:      # validation gate: minimums the result must meet to pass
  accuracy: 0.5
'''

_INIT_REQS = "pyyaml\n"

_INIT_JOB_TEMPLATE = '''\
# .vgreens/job-template.yaml — defaults for `vgreens-ai job create --template`.
name: my-training
job_type: training
gpu_count: 1
# docker_image: my-model:latest    # short ref; resolved to the full registry URL
# dataset_name: my-dataset
'''

_INIT_README = '''\
# {name}

A vGreens training project scaffold.

## Layout
- `train.py` — training entrypoint (writes model.pt + metrics.json + metrics.jsonl).
- `Dockerfile` — how the training image is built.
- `config.yaml` — hyperparameters + validation thresholds.
- `.vgreens/job-template.yaml` — defaults for `vgreens-ai job create`.

## Build & run
1. Edit the REPLACE block in `train.py` with your real training loop.
2. Build + push the image via CI (or `docker build -t <image> .`).
3. Submit: `vgreens-ai job create --interactive` (or `job train --model <m> --image <img> --dataset <d>:v1`).
'''


@init_app.command("training-project")
def init_training_project(
        directory: str = typer.Argument(".", help="target directory"),
        name: str = typer.Option("my-training", help="project name"),
        force: bool = typer.Option(False, "--force", help="write into a non-empty dir")):
    """Scaffold a ready-to-edit training repo (Dockerfile, train.py, config, template)."""
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    existing = [p for p in root.iterdir()] if root.is_dir() else []
    if existing and not force:
        output.error(f"{root} is not empty — use --force to write anyway"); raise typer.Exit(2)
    files = {
        "train.py": _INIT_TRAIN_PY,
        "Dockerfile": _INIT_DOCKERFILE,
        "config.yaml": _INIT_CONFIG,
        "requirements.txt": _INIT_REQS,
        ".vgreens/job-template.yaml": _INIT_JOB_TEMPLATE,
        "README.md": _INIT_README.format(name=name),
    }
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    output.success(f"scaffolded training project in {root}")
    for rel in files:
        output.info(f"  {rel}")
    output.info("next: edit the REPLACE block in train.py, then build + submit.")


# --------------------------------------------------------------------------- #
# job create (interactive builder, #17) + plan (#6/#14/#15) + clone (#10)
# --------------------------------------------------------------------------- #
def _print_plan(plan: dict) -> None:
    """Render a /jobs/plan result: checks, recommended GPU, command preview."""
    from rich.table import Table
    tbl = Table(title="pre-flight checks", show_header=True)
    tbl.add_column("check"); tbl.add_column("status"); tbl.add_column("detail")
    icon = {"ok": "[green]✓ ok[/]", "warning": "[yellow]! warn[/]", "error": "[red]✗ error[/]"}
    for c in plan.get("checks", []):
        tbl.add_row(c["name"], icon.get(c["status"], c["status"]), c.get("detail", ""))
    output.err.print(tbl)
    adv = plan.get("gpu_advice", {})
    if adv:
        output.info(f"recommended GPU: {adv.get('recommended_gpu')} — {adv.get('reason','')}")
    output.info(f"resolved image: {plan.get('resolved_image') or '(from registry)'}")
    output.info(f"output path: {plan.get('output_path_preview')}")
    output.err.print("[dim]command preview:[/]\n" + (plan.get("command_preview") or ""))


@job_app.command("plan")
def job_plan(ctx: typer.Context,
             model: str = typer.Option(..., help="model name"),
             job_type: str = typer.Option("training", "--type"),
             dataset: str = typer.Option(None, help="name:version"),
             image: Optional[str] = typer.Option(None),
             gpu_count: int = typer.Option(1, "--gpu-count"),
             out: Optional[str] = OUTPUT_OPT):
    """Dry-run pre-flight: checks, recommended GPU, resolved image, command preview (#6/#14/#15)."""
    _mode(ctx, out)
    g: G = ctx.obj
    name, _, version = (dataset or "").partition(":")
    body = {"job_type": job_type, "model_name": model, "gpu_count": gpu_count,
            "docker_image": image, "dataset_name": name or None,
            "dataset_version": version or None}
    plan = _client(ctx).post("/jobs/plan", json=body, spinner="planning…")
    if g.output == "table":
        _print_plan(plan)
    else:
        output.emit(plan, g.output)


@job_app.command("create")
def job_create(ctx: typer.Context,
               template: Optional[str] = typer.Option(None, help="start from a template id"),
               model: Optional[str] = typer.Option(None),
               job_type: Optional[str] = typer.Option(None, "--type"),
               dataset: Optional[str] = typer.Option(None, help="name:version"),
               image: Optional[str] = typer.Option(None),
               gpu_count: Optional[int] = typer.Option(None, "--gpu-count"),
               priority: int = typer.Option(0),
               yes: bool = typer.Option(False, "--yes", help="skip the confirm prompt")):
    """Interactive job builder — asks what it needs, runs pre-flight, then submits (#17)."""
    g: G = ctx.obj
    c = _client(ctx)
    tpl = c.get(f"/templates/{template}") if template else {}

    def ask(label, default=None):
        if yes or not sys.stdin.isatty():
            return default
        return typer.prompt(label, default=default if default is not None else "")

    job_type = job_type or tpl.get("job_type") or ask("Job type (training/processing)", "training")
    model = model or ask("Model name", tpl.get("model_name"))
    if not model:
        output.error("model is required"); raise typer.Exit(2)
    if image is None:
        image = tpl.get("docker_image") or (ask("Docker image (blank = registry)", "") or None)
    if gpu_count is None:
        gpu_count = int(tpl.get("gpu_count") or (ask("GPU count", "1") if job_type == "training" else 1))
    if dataset is None and job_type == "training":
        dsn = tpl.get("dataset_name") or ""
        dataset = ask("Dataset (name:version)", dsn)
    name, _, version = (dataset or "").partition(":")

    body = {"job_type": job_type, "model_name": model, "gpu_count": gpu_count,
            "priority": priority, "docker_image": image,
            "dataset_name": name or None, "dataset_version": version or None}

    # Pre-flight before submit so nothing fails 10s in.
    try:
        plan = c.post("/jobs/plan", json={k: v for k, v in body.items() if k != "priority"},
                      spinner="pre-flight…")
        if g.output == "table":
            _print_plan(plan)
        if not plan.get("ok") and not yes and sys.stdin.isatty():
            typer.confirm("Pre-flight found errors. Submit anyway?", abort=True)
    except Exception as exc:  # noqa: BLE001 — planning is advisory, never blocks
        output.warn(f"pre-flight skipped: {exc}")

    if not yes and sys.stdin.isatty():
        typer.confirm(f"Submit {job_type} job for '{model}'?", abort=True)
    _submit(ctx, body)


@job_app.command("clone")
def job_clone(ctx: typer.Context,
              job_id: str = typer.Argument(..., help="job to clone"),
              dataset: Optional[str] = typer.Option(None, help="override name:version"),
              image: Optional[str] = typer.Option(None, help="override image"),
              gpu_count: Optional[int] = typer.Option(None, "--gpu-count"),
              submit: bool = typer.Option(False, "--submit", help="submit immediately"),
              out: Optional[str] = OUTPUT_OPT):
    """Clone an existing job (optionally with tweaks) and re-run it (#10)."""
    _mode(ctx, out)
    g: G = ctx.obj
    c = _client(ctx)
    body = c.get(f"/jobs/{job_id}/clone", spinner="loading job…")
    if dataset is not None:
        name, _, version = dataset.partition(":")
        body["dataset_name"] = name or None
        body["dataset_version"] = version or None
    if image is not None:
        body["docker_image"] = image
    if gpu_count is not None:
        body["gpu_count"] = gpu_count
    if not submit:
        output.emit(body, g.output, title=f"clone of {job_id} (use --submit to run)")
        return
    _submit(ctx, body)


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


@job_app.command("progress")
def job_progress(ctx: typer.Context, job_id: str = typer.Argument(...),
                 out: Optional[str] = OUTPUT_OPT):
    """Show live training progress (epochs/metrics/ETA) from metrics.jsonl (#7)."""
    _mode(ctx, out)
    g: G = ctx.obj
    res = _client(ctx).get(f"/jobs/{job_id}/progress", spinner="loading progress…")
    if g.output != "table":
        output.emit(res, g.output); return
    if not res.get("available"):
        output.info(f"no progress yet (status: {res.get('status')})"); return
    cur, total = res.get("current_epoch"), res.get("total_epochs")
    eta = res.get("eta_seconds")
    output.success(f"epoch {cur}/{total}" + (f" — ETA {output.fmt_duration(eta)}" if eta else ""))
    latest = res.get("latest") or {}
    metrics = {k: v for k, v in latest.items()
               if k not in ("epoch", "total_epochs", "ts", "stage", "world_size")}
    if metrics:
        output.emit(metrics, "table", title="latest metrics")


@job_app.command("explain")
def job_explain(ctx: typer.Context, job_id: str = typer.Argument(...),
                out: Optional[str] = OUTPUT_OPT):
    """Explain why a job failed, with suggested fixes (#9)."""
    _mode(ctx, out)
    g: G = ctx.obj
    res = _client(ctx).get(f"/jobs/{job_id}/explain", spinner="analysing…")
    if g.output != "table":
        output.emit(res, g.output); return
    output.warn(res.get("cause", ""))
    for s in res.get("suggestions", []):
        output.info(f"  • {s}")


@job_app.command("report")
def job_report(ctx: typer.Context, job_id: str = typer.Argument(...),
               save: Optional[str] = typer.Option(None, help="write the report JSON to a file"),
               out: Optional[str] = OUTPUT_OPT):
    """Fetch the standard job report; optionally save it as JSON (#19)."""
    _mode(ctx, out)
    g: G = ctx.obj
    res = _client(ctx).get(f"/jobs/{job_id}/report", spinner="building report…")
    if save:
        Path(save).write_text(json.dumps(res, indent=2))
        output.success(f"wrote {save}")
        return
    output.emit(res, g.output, title=f"report {job_id}")


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


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


@job_app.command("result")
def job_result(ctx: typer.Context, job_id: str = typer.Argument(...),
               out: Optional[str] = OUTPUT_OPT):
    """Show a finished job's results.

    Training  -> validation metrics (mAP50, precision, recall) + the registered
                 model version.
    Processing-> detection counts (total + per class).
    Both      -> the list of output files; download one with `job download`.
    """
    _mode(ctx, out)
    g: G = ctx.obj
    r = _client(ctx).get(f"/jobs/{job_id}/result", spinner="loading result…")
    if g.output != "table":
        output.emit(r, g.output)
        return
    if not r.get("available"):
        output.warn(f"no outputs on record for {job_id} (status: {r.get('status')}). "
                    f"Results appear once the job succeeds.")
        return
    if r.get("result_version"):
        output.info(f"registered model version: [bold]{r['result_version']}[/]")
    if r.get("metrics"):
        output.emit([{"metric": k, "value": v} for k, v in r["metrics"].items()], "table",
                    columns=[("metric", "metric"), ("value", "value")], title=f"{job_id} metrics")
    rep = r.get("report")
    if isinstance(rep, dict):
        summ = {k: rep[k] for k in ("images", "total_detections", "conf_threshold") if k in rep}
        if summ:
            output.emit(summ, "table", title=f"{job_id} processing summary")
        if isinstance(rep.get("by_class"), dict) and rep["by_class"]:
            output.emit([{"class": k, "detections": v} for k, v in rep["by_class"].items()], "table",
                        columns=[("class", "class"), ("detections", "detections")],
                        title="detections by class")
    arts = r.get("artifacts") or []
    if arts:
        output.emit([{"file": a["path"], "size": _fmt_bytes(a["size_bytes"])} for a in arts], "table",
                    columns=[("file", "file"), ("size", "size")], title=f"{job_id} artifacts")
        output.info(f"outputs on NAS: [dim]{r.get('output_path')}/out[/]")
        output.info(f"download a file: [bold]vgreens-ai job download {job_id} --path <file>[/]")


@job_app.command("download")
def job_download(ctx: typer.Context, job_id: str = typer.Argument(...),
                 path: str = typer.Option(..., "--path", help="artifact path as shown by `job result`"),
                 dest: Optional[str] = typer.Option(None, "--dest",
                     help="local file or directory to save into (default: current dir)")):
    """Download one output file from a finished job."""
    import os
    target = dest or os.getcwd()
    saved = _client(ctx).download(f"/jobs/{job_id}/artifacts/download", target,
                                  params={"path": path}, spinner="downloading…")
    output.success(f"saved {saved}")


@job_app.command("logs")
def job_logs(ctx: typer.Context, job_id: str = typer.Argument(...),
             tail: int = typer.Option(200, "--tail", help="last N lines"),
             out: Optional[str] = OUTPUT_OPT):
    """Show the tail of a job's container log."""
    _mode(ctx, out)
    g: G = ctx.obj
    r = _client(ctx).get(f"/jobs/{job_id}/logs", params={"tail": tail})
    if g.output != "table":
        output.emit(r, g.output)
        return
    if not r.get("available"):
        output.warn(f"no log on record for {job_id} yet.")
        return
    for line in r.get("lines", []):
        print(line)
    output.info(f"[dim]showing last {len(r.get('lines', []))} of {r.get('total_lines')} lines[/]")


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


@model_app.command("delete")
def model_delete(ctx: typer.Context, model: str = typer.Argument(...),
                 force: bool = typer.Option(False, "--force",
                     help="delete even if the model has an ACTIVE production version"),
                 yes: bool = typer.Option(False, "--yes", "-y", help="skip the confirmation prompt")):
    """Delete a model from the registry (admin only).

    Safe by default: the model is ARCHIVED to a recoverable `_trash` area on the
    NAS, not hard-deleted. Refuses if the model still serves production (has an
    active version) unless you pass --force.
    """
    g: G = ctx.obj
    if not yes and sys.stdout.isatty():
        typer.confirm(f"Delete model '{model}'? It will be archived (recoverable) and "
                      f"disappear from the registry.", abort=True)
    res = _client(ctx).delete(f"/models/{model}", params={"force": str(force).lower()},
                              spinner="deleting…")
    if g.output == "table":
        note = " (forced — had an active version)" if res.get("forced") else ""
        output.warn(f"deleted {model}{note}; archived to {res.get('archived_to')}")
    else:
        output.emit(res, g.output)


@model_app.command("presets")
def model_presets(ctx: typer.Context, out: Optional[str] = OUTPUT_OPT):
    """List the model onboarding presets."""
    _mode(ctx, out)
    g: G = ctx.obj
    rows = [{"preset": k, "task": v["task"], "working_code": v["generates_working_code"]}
            for k, v in modelspec.PRESETS.items()]
    output.emit(rows, g.output,
                columns=[("preset", "preset"), ("task", "task"), ("working code", "working_code")],
                title="model presets")


def _create_gitlab_repo(spec: dict, out_dir: str) -> None:
    """Create the GitLab project in the spec's group and push the skeleton.

    Uses GITLAB_TOKEN (a PAT with 'api' scope) from the environment; the token is
    never printed or persisted in the git remote.
    """
    import json
    import os
    import subprocess
    import urllib.error
    import urllib.parse
    import urllib.request

    token = os.environ.get("GITLAB_TOKEN")
    if not token:
        raise CLIError("set GITLAB_TOKEN (GitLab PAT with 'api' scope) to use --create-repo", 2)
    host = os.environ.get("GITLAB_HOST", "https://gitlab.com").rstrip("/")
    ns = spec.get("repository", {}).get("namespace", "vgreens-models")
    name = spec.get("repository", {}).get("name") or spec["model"]["name"]

    def _api(method: str, path: str, data: dict | None = None) -> dict:
        body = urllib.parse.urlencode(data).encode() if data else None
        req = urllib.request.Request(f"{host}/api/v4{path}", data=body, method=method,
                                     headers={"PRIVATE-TOKEN": token})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            raise CLIError(f"GitLab API {method} {path} failed: {exc.code} {exc.reason}", 1)

    grp = _api("GET", f"/groups/{urllib.parse.quote_plus(ns)}")
    proj = _api("POST", "/projects",
                {"name": name, "namespace_id": grp["id"], "visibility": "private"})
    clean_url = proj["http_url_to_repo"]
    output.success(f"created GitLab project {proj['path_with_namespace']}")

    def _git(*args: str) -> None:
        subprocess.run(["git", "-C", out_dir, *args], check=True)

    def _git_rc(*args: str) -> int:
        return subprocess.run(["git", "-C", out_dir, *args],
                              capture_output=True).returncode

    _git("init", "-q", "-b", "main")
    _git("add", "-A")
    # Idempotent on re-runs: commit only if there is something staged.
    if _git_rc("diff", "--cached", "--quiet") != 0:
        _git("-c", "user.email=cli@vgreens", "-c", "user.name=vgreens-ai",
             "commit", "-q", "-m", "chore: scaffold model repo")
    # Add-or-update origin so a second --create-repo run doesn't crash on an
    # existing remote. clean URL (no token) is what gets persisted.
    if _git_rc("remote", "set-url", "origin", clean_url) != 0:
        _git("remote", "add", "origin", clean_url)
    # Push with a one-shot credential helper so the token is never written into
    # .git/config or printed in git output (embedding it in the URL leaks it).
    helper = '!f() { echo "username=oauth2"; echo "password=$GITLAB_TOKEN"; }; f'
    subprocess.run(["git", "-C", out_dir, "-c", f"credential.helper={helper}",
                    "push", "-u", "origin", "main"],
                   check=True, env={**os.environ, "GITLAB_TOKEN": token})
    output.success(f"pushed to {proj['web_url']} — pipeline starts on the group runner")


@model_app.command("onboard")
def model_onboard(ctx: typer.Context,
                  spec_path: str = typer.Option(..., "--spec", help="path to model.yaml"),
                  output_dir: str = typer.Option(None, "--output-dir", help="target dir (default: repo name)"),
                  dry_run: bool = typer.Option(False, "--dry-run", help="show files, write nothing"),
                  force: bool = typer.Option(False, "--force", help="overwrite a non-empty dir"),
                  create_repo: bool = typer.Option(False, "--create-repo",
                                                   help="create the GitLab project in the group and push (needs GITLAB_TOKEN)")):
    """Generate a model repository from a model.yaml spec."""
    try:
        spec = modelspec.load_spec(spec_path)
    except OSError as exc:
        raise CLIError(f"cannot read {spec_path}: {exc}", 2)
    except Exception as exc:  # noqa: BLE001
        raise CLIError(f"invalid YAML in {spec_path}: {exc}", 2)
    errors = modelspec.validate(spec)
    if errors:
        for e in errors:
            output.error(e)
        raise CLIError(f"{len(errors)} problem(s) in the spec", 2)

    out = output_dir or (spec.get("repository", {}).get("name") or spec["model"]["name"])
    try:
        files = modelgen.generate(spec, out, dry_run=dry_run, force=force)
    except FileExistsError as exc:
        raise CLIError(str(exc), 6)
    except ValueError as exc:
        raise CLIError(str(exc), 2)

    verb = "would generate" if dry_run else "generated"
    output.info(f"{verb} {len(files)} file(s) in [bold]{out}/[/]:")
    for f in files:
        output.err.print(f"  {f}")
    if dry_run:
        return
    output.success(f"model repo ready at {out}/")
    if modelspec.threshold_state(spec) != "configured":
        output.warn("thresholds not configured — trained versions will be 'baseline_only'")
    if create_repo:
        _create_gitlab_repo(spec, out)
    else:
        output.info("next: cd in, run `pytest tests/`, then git push "
                    "(or re-run with --create-repo to create + push automatically)")


@model_app.command("validate-spec")
def model_validate_spec(ctx: typer.Context, path: str = typer.Argument(..., help="path to model.yaml")):
    """Validate a model.yaml specification."""
    try:
        spec = modelspec.load_spec(path)
    except OSError as exc:
        raise CLIError(f"cannot read {path}: {exc}", 2)
    except Exception as exc:  # noqa: BLE001 — yaml parse error
        raise CLIError(f"invalid YAML in {path}: {exc}", 2)
    errors = modelspec.validate(spec)
    if errors:
        for e in errors:
            output.error(e)
        raise CLIError(f"{len(errors)} problem(s) in {path}", 2)
    state = modelspec.threshold_state(spec)
    output.success(f"{path} is valid")
    if state != "configured":
        output.warn("validation thresholds not configured — a trained version will be "
                    "'baseline_only' and cannot be promoted until thresholds are set")


@model_app.command("init")
def model_init(ctx: typer.Context,
               path: str = typer.Option("model.yaml", "--out", help="where to write the spec")):
    """Interactively scaffold a valid model.yaml (no hand-editing YAML)."""
    import yaml as _yaml
    from pathlib import Path as _Path

    output.info("Scaffolding model.yaml — press Ctrl-C to abort.")
    name = typer.prompt("model name (lowercase_underscore)")
    task = typer.prompt(f"task {modelspec.TASKS}", default="object_detection")
    preset_by_task = {v["task"]: k for k, v in modelspec.PRESETS.items()}
    preset = typer.prompt(f"preset {tuple(modelspec.PRESETS)}",
                          default=preset_by_task.get(task, "custom_python"))
    ds_name = typer.prompt("dataset name")
    ds_ver = typer.prompt("dataset version", default="v1")
    gpu = typer.prompt("gpu_count (1 or 2)", default=1, type=int)
    metric = typer.prompt("primary metric name (e.g. mAP50, accuracy)", default="mAP50")
    bar = typer.prompt(f"minimum {metric} for promotion (blank = none yet)",
                       default="", show_default=False)
    ns = typer.prompt("GitLab group/namespace", default="vgreens-models")
    repo = typer.prompt("repository name", default=f"vgreens-model-{name.replace('_', '-')}")

    spec = {
        "schema_version": modelspec.SCHEMA_VERSION,
        "model": {"name": name, "task": task, "owner": ""},
        "template": {"preset": preset},
        "dataset": {"name": ds_name, "version": ds_ver},
        "training": {"epochs": 50, "image_size": 640, "batch_size": 16, "gpu_count": int(gpu)},
        "validation": {"thresholds": ({metric: float(bar)} if str(bar).strip() else {})},
        "processing": {"confidence_threshold": 0.25},
        "repository": {"namespace": ns, "name": repo},
    }
    errors = modelspec.validate(spec)
    if errors:
        for e in errors:
            output.error(e)
        raise CLIError("could not build a valid spec from those answers", 2)
    _Path(path).write_text(_yaml.safe_dump(spec, sort_keys=False))
    output.success(f"wrote {path}")
    output.info(f"next: vgreens-ai model onboard --spec {path} --create-repo")


@config_app.command("publish")
def config_publish(ctx: typer.Context,
                   model: str = typer.Argument(..., help="model name"),
                   file: str = typer.Option(..., "--file", "-f", help="path to the job-config yaml"),
                   out: Optional[str] = OUTPUT_OPT):
    """Publish a model's job-config to the platform (no SSH/cp on the GPU box)."""
    _mode(ctx, out)
    try:
        content = open(file).read()
    except OSError as exc:
        raise CLIError(f"cannot read {file}: {exc}", 2)
    res = _client(ctx).post(f"/model-configs/{model}", json={"content": content})
    output.success(f"published {model} config -> {res.get('path')}")


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
def _config_init_flow(profile: str = "prod", api_url: Optional[str] = None,
                      token: Optional[str] = None) -> None:
    """Prompt, validate reachability + auth, save securely. Shared by `config init`
    and `quickstart` so the guided setup lives in one place."""
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


@config_app.command("init")
def config_init(profile: str = typer.Option("prod", help="profile name"),
                api_url: str = typer.Option(None, help="skip the prompt with this URL"),
                token: str = typer.Option(None, help="skip the prompt with this token")):
    """Guided setup: prompt, validate reachability + auth, save securely."""
    _config_init_flow(profile, api_url, token)


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


@app.command("whoami")
def whoami(ctx: typer.Context, out: Optional[str] = OUTPUT_OPT):
    """Show who the current profile/token authenticates as, and its role."""
    _mode(ctx, out)
    g: G = ctx.obj
    me = _client(ctx).get("/auth/me")
    if g.output == "table":
        output.info(f"{me.get('username')} · role {output.badge(me.get('role'))} · via {me.get('via')}")
        if me.get("scopes"):
            output.info(f"scopes: {me['scopes']}")
    else:
        output.emit(me, g.output)


@app.command("quickstart")
def quickstart(ctx: typer.Context):
    """Guided first run: configure a profile, verify connectivity + identity,
    and print role-tailored next steps."""
    g: G = ctx.obj
    try:
        configured = bool(config.default_profile())
    except Exception:  # noqa: BLE001
        configured = False
    if not configured and not (g.api_url and g.token):
        output.info("No profile configured yet — let's set one up.")
        _config_init_flow()

    c = _client(ctx)
    try:
        ver = c.get("/version")
        output.success(f"connected · API {ver.get('version', '?')}")
    except CLIError as exc:
        raise CLIError(f"cannot reach the platform: {exc} — run `vgreens-ai config init`", exc.code)

    me = c.get("/auth/me")
    role = me.get("role", "viewer")
    output.success(f"authenticated as {me.get('username')} (role: {role})")

    try:
        ov = c.get("/system/overview")
        w = ov.get("workers", {}) or {}
        output.info(f"workers online: {w.get('online', '?')}/{w.get('total', '?')} · "
                    f"queued jobs: {ov.get('queued_total', '?')}")
    except CLIError:
        pass

    output.err.print("")
    output.info("Next steps:")
    lines = [
        "vgreens-ai job list            see recent jobs",
        "vgreens-ai model list          registered models",
        "vgreens-ai system overview     platform health",
    ]
    if role in ("operator", "admin"):
        lines += [
            "vgreens-ai model init          scaffold a new custom model",
            "vgreens-ai dataset import ...  bring in a dataset",
            "vgreens-ai job train ...       start a training run",
        ]
    if role == "admin":
        lines += [
            "vgreens-ai admin users         manage accounts",
            "vgreens-ai admin audit         review the audit trail",
        ]
    for line in lines:
        output.err.print(f"  {line}")


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


@dataset_app.command("backfill")
def dataset_backfill(ctx: typer.Context,
                     apply: bool = typer.Option(False, "--apply",
                                                help="actually index (default: dry-run report only)"),
                     out: Optional[str] = OUTPUT_OPT):
    """Index pre-existing (legacy) frozen datasets into the registry. Admin/ops."""
    _mode(ctx, out)
    g: G = ctx.obj
    dry = "false" if apply else "true"
    res = _client(ctx).post(f"/datasets/backfill?dry_run={dry}", spinner="scanning NAS…")
    if g.output != "table":
        output.emit(res, g.output)
        return
    added = res.get("added", [])
    output.emit(added, g.output,
                columns=[("dataset", "dataset_name"), ("version", "version"),
                         ("images", "image_count"), ("classes", "class_count"),
                         ("metadata", "metadata_source")],
                title=("would add" if res.get("dry_run") else "added"))
    verb = "would index" if res.get("dry_run") else "indexed"
    output.info(f"scanned {res.get('scanned', 0)} · already indexed {res.get('already_indexed', 0)} · "
                f"{verb} {len(added)} · errors {len(res.get('errors', []))}")
    if res.get("dry_run") and added:
        output.info("re-run with [bold]--apply[/] to write these records")


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
def _process(ctx: typer.Context, model: str = typer.Option(...), image: Optional[str] = typer.Option(None),
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


# ---------------------------------------------------------------------------
# admin: users / tokens / audit  (server enforces the admin role + all guards)
# ---------------------------------------------------------------------------
_CONFLICT_HINT = {
    "USER_HAS_ACTIVE_SESSIONS": "force logout first: vgreens-ai admin force-logout <id>",
    "USER_HAS_ACTIVE_TOKENS": "revoke their tokens first (vgreens-ai admin tokens)",
    "USER_HAS_ACTIVE_JOBS": "wait for or cancel their queued/running jobs first",
    "USER_IS_FINAL_ADMIN": "the platform must always keep one active admin",
}


def _admin_run(fn):
    """Call an admin endpoint and turn conflict codes into a readable hint."""
    try:
        return fn()
    except CLIError as exc:
        hint = _CONFLICT_HINT.get(str(exc).strip())
        if hint:
            raise CLIError(f"{exc} — {hint}", exc.code) from None
        raise


@admin_app.command("users")
def admin_users(ctx: typer.Context, out: Optional[str] = OUTPUT_OPT):
    """List all users."""
    _mode(ctx, out)
    g: G = ctx.obj
    rows = _client(ctx).get("/admin/users")
    output.emit(rows, g.output, columns=[
        ("id", "id"), ("username", "username"), ("role", "role"),
        ("active", "active"), ("last login", "last_login"),
    ], title="users")


@admin_app.command("user")
def admin_user(ctx: typer.Context, user_id: int = typer.Argument(...), out: Optional[str] = OUTPUT_OPT):
    """Show one user with active session / token / job counts."""
    _mode(ctx, out)
    g: G = ctx.obj
    output.emit(_client(ctx).get(f"/admin/users/{user_id}"), g.output)


@admin_app.command("create-user")
def admin_create_user(ctx: typer.Context,
                      username: str = typer.Option(..., "--username", "-u"),
                      role: str = typer.Option("viewer", "--role", help="viewer | operator | admin"),
                      password: Optional[str] = typer.Option(None, "--password",
                          help="prompted (hidden) if omitted")):
    """Create a user."""
    g: G = ctx.obj
    if not password:
        password = typer.prompt("Password (min 8 chars)", hide_input=True, confirmation_prompt=True)
    res = _client(ctx).post("/admin/users", json={"username": username, "password": password, "role": role},
                            spinner="creating user…")
    if g.output == "table":
        output.success(f"created user '{username}' ({role})")
    else:
        output.emit(res, g.output)


@admin_app.command("set-role")
def admin_set_role(ctx: typer.Context, user_id: int = typer.Argument(...),
                   role: str = typer.Argument(..., help="viewer | operator | admin")):
    """Change a user's role."""
    _admin_run(lambda: _client(ctx).request("PATCH", f"/admin/users/{user_id}",
                                            json={"role": role}, spinner="updating…"))
    output.success(f"user {user_id} → {role}")


@admin_app.command("disable")
def admin_disable(ctx: typer.Context, user_id: int = typer.Argument(...)):
    """Disable a user and revoke their sessions."""
    _admin_run(lambda: _client(ctx).post(f"/admin/users/{user_id}/disable", spinner="disabling…"))
    output.success(f"user {user_id} disabled")


@admin_app.command("enable")
def admin_enable(ctx: typer.Context, user_id: int = typer.Argument(...)):
    """Re-enable a disabled user."""
    _client(ctx).post(f"/admin/users/{user_id}/enable", spinner="enabling…")
    output.success(f"user {user_id} enabled")


@admin_app.command("force-logout")
def admin_force_logout(ctx: typer.Context, user_id: int = typer.Argument(...)):
    """Revoke all of a user's active sessions."""
    res = _client(ctx).post(f"/admin/users/{user_id}/force-logout", spinner="revoking sessions…")
    n = (res or {}).get("sessions_revoked", 0)
    output.success(f"user {user_id}: {n} session(s) revoked")


@admin_app.command("reset-password")
def admin_reset_password(ctx: typer.Context, user_id: int = typer.Argument(...),
                         password: Optional[str] = typer.Option(None, "--password")):
    """Set a new password for a user."""
    if not password:
        password = typer.prompt("New password (min 8 chars)", hide_input=True, confirmation_prompt=True)
    _client(ctx).post(f"/admin/users/{user_id}/reset-password", json={"new_password": password},
                      spinner="resetting…")
    output.success(f"password reset for user {user_id}")


@admin_app.command("anonymise")
def admin_anonymise(ctx: typer.Context, user_id: int = typer.Argument(...),
                    yes: bool = typer.Option(False, "--yes", help="skip confirmation")):
    """Anonymise a user: rename to deleted_user_<id>, revoke sessions+tokens, keep audit history."""
    if not yes:
        typer.confirm(f"Anonymise user {user_id}? This renames the account and revokes access. "
                      "It cannot be undone.", abort=True)
    res = _admin_run(lambda: _client(ctx).post(f"/admin/users/{user_id}/anonymise", spinner="anonymising…"))
    output.success(f"user {user_id} anonymised → {(res or {}).get('username', 'deleted_user')}")


@admin_app.command("delete")
def admin_delete(ctx: typer.Context, user_id: int = typer.Argument(...),
                 yes: bool = typer.Option(False, "--yes", help="skip confirmation")):
    """Permanently delete a user (only if they have no active sessions, tokens, or jobs)."""
    if not yes:
        typer.confirm(f"Permanently delete user {user_id}? Consider `anonymise` to keep audit history.",
                      abort=True)
    _admin_run(lambda: _client(ctx).request("DELETE", f"/admin/users/{user_id}", spinner="deleting…"))
    output.success(f"user {user_id} deleted")


@admin_app.command("tokens")
def admin_tokens(ctx: typer.Context, out: Optional[str] = OUTPUT_OPT):
    """List API tokens."""
    _mode(ctx, out)
    g: G = ctx.obj
    rows = _client(ctx).get("/admin/tokens")
    output.emit(rows, g.output, columns=[
        ("id", "id"), ("name", "name"), ("owner", "owner"),
        ("scopes", "scopes"), ("created", "created_at"), ("revoked", "revoked"),
    ], title="api tokens")


@admin_app.command("create-token")
def admin_create_token(ctx: typer.Context,
                       name: str = typer.Option(..., "--name"),
                       scopes: str = typer.Option("jobs:read jobs:submit", "--scopes",
                           help="space-separated scopes")):
    """Create an API token (the secret is shown once)."""
    res = _client(ctx).post("/admin/tokens", json={"name": name, "scopes": scopes},
                            spinner="creating token…")
    token = (res or {}).get("token")
    output.success("token created — copy it now, it is shown only once:")
    output.info(token or "(no token returned)")


@admin_app.command("revoke-token")
def admin_revoke_token(ctx: typer.Context, token_id: str = typer.Argument(...)):
    """Revoke an API token."""
    _client(ctx).request("DELETE", f"/admin/tokens/{token_id}", spinner="revoking…")
    output.success(f"token {token_id} revoked")


@admin_app.command("audit")
def admin_audit(ctx: typer.Context, limit: int = typer.Option(40, "--limit"),
                out: Optional[str] = OUTPUT_OPT):
    """Show the audit trail (most recent first)."""
    _mode(ctx, out)
    g: G = ctx.obj
    rows = _client(ctx).get("/audit", params={"limit": limit})
    output.emit(rows, g.output, columns=[
        ("when", "ts"), ("actor", "actor"), ("action", "action"),
        ("target", "target"), ("result", "result"), ("source", "source"),
    ], title="audit log")


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
