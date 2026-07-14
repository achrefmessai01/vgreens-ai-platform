"""vgreens-ai — thin typer CLI over the Job API.

Zero business logic: every command is one HTTP call + pretty print. All rules
live in the API so the CLI and reality can't drift apart.

Commands (Phase 1): train, process, status, list-jobs.
Later phases add: logs, models, promote, rollback, datasets, cancel.
"""
from __future__ import annotations

import sys

import requests
import typer
from rich.console import Console
from rich.table import Table

from . import config

app = typer.Typer(add_completion=False, help="vGreens AI platform CLI")
console = Console()


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _die(msg: str) -> None:
    console.print(f"[red]error:[/red] {msg}")
    raise SystemExit(1)


def _request(method: str, path: str, **kwargs) -> dict | list:
    api_url, token = config.load()
    try:
        resp = requests.request(
            method, f"{api_url}{path}", headers=_headers(token), timeout=30, **kwargs
        )
    except requests.RequestException as exc:
        _die(f"cannot reach API at {api_url}: {exc}")
    if resp.status_code >= 400:
        detail = resp.json().get("detail", resp.text) if resp.content else resp.reason
        _die(f"{resp.status_code}: {detail}")
    return resp.json()


@app.command()
def train(
    model: str = typer.Option(..., help="model name, e.g. dummy"),
    dataset: str = typer.Option(..., help="dataset as name:version, e.g. test:v1"),
    image: str = typer.Option("vgreens/dummy-job:v1", help="docker image for the job"),
    tenant: str = typer.Option(None, help="tenant tag"),
    config_path: str = typer.Option(None, "--config", help="path to job.yaml on the NAS"),
):
    """Submit a training job."""
    name, _, version = dataset.partition(":")
    body = {
        "job_type": "training",
        "model_name": model,
        "dataset_name": name,
        "dataset_version": version or None,
        "docker_image": image,
        "tenant": tenant,
        "config_path": config_path,
    }
    res = _request("POST", "/jobs", json=body)
    console.print(f"[green]queued[/green] {res['job_id']}")


@app.command()
def process(
    model: str = typer.Option(..., help="model name"),
    image: str = typer.Option(..., help="docker image for the processing job"),
    tenant: str = typer.Option(None, help="tenant tag"),
    config_path: str = typer.Option(None, "--config", help="path to job.yaml on the NAS"),
):
    """Submit a processing job."""
    body = {
        "job_type": "processing",
        "model_name": model,
        "docker_image": image,
        "tenant": tenant,
        "config_path": config_path,
    }
    res = _request("POST", "/jobs", json=body)
    console.print(f"[green]queued[/green] {res['job_id']}")


@app.command()
def status(job_id: str = typer.Option(..., "--job-id", help="job id")):
    """Show one job."""
    job = _request("GET", f"/jobs/{job_id}")
    table = Table(show_header=False, box=None)
    for key in (
        "job_id", "job_type", "model_name", "dataset_version", "status",
        "worker", "gpu", "requested_by", "created_at", "started_at",
        "finished_at", "output_path", "error",
    ):
        val = job.get(key)
        if val is not None:
            table.add_row(f"[bold]{key}[/bold]", str(val))
    console.print(table)


@app.command(name="list-jobs")
def list_jobs(
    status: str = typer.Option(None, help="filter by status"),
    model: str = typer.Option(None, help="filter by model"),
    limit: int = typer.Option(20, help="max rows"),
):
    """List recent jobs."""
    params = {"limit": limit}
    if status:
        params["status"] = status
    if model:
        params["model"] = model
    jobs = _request("GET", "/jobs", params=params)

    table = Table(title="jobs")
    for col in ("job_id", "type", "model", "status", "worker", "created"):
        table.add_column(col)
    for j in jobs:
        table.add_row(
            j["job_id"], j["job_type"], j["model_name"], j["status"],
            j.get("worker") or "-", (j.get("created_at") or "")[:19],
        )
    console.print(table)


@app.command()
def models(model: str = typer.Option(None, help="model name; omit to list all models")):
    """List models, or versions of one model (marking the active one)."""
    if not model:
        res = _request("GET", "/models")
        table = Table(title="models")
        table.add_column("model")
        for m in res.get("models", []):
            table.add_row(m)
        console.print(table)
        return

    versions = _request("GET", f"/models/{model}/versions")
    table = Table(title=f"{model} — versions")
    for col in ("version", "status", "active"):
        table.add_column(col)
    for v in versions:
        table.add_row(
            v["version"],
            v.get("status") or "-",
            "★ active" if v.get("active") else "",
        )
    console.print(table)


@app.command()
def promote(
    model: str = typer.Option(..., help="model name"),
    version: str = typer.Option(..., help="version to promote (must be validated)"),
    reason: str = typer.Option(None, help="why (recorded in the audit log)"),
):
    """Promote a validated version to production."""
    res = _request("POST", f"/models/{model}/promote",
                   json={"version": version, "reason": reason})
    console.print(f"[green]promoted[/green] {model} -> {res['active_version']}"
                  f" (was {res.get('previous_stable') or 'none'})")


@app.command()
def rollback(
    model: str = typer.Option(..., help="model name"),
    to: str = typer.Option(None, "--to", help="version to roll back to (default: previous stable)"),
    reason: str = typer.Option(None, help="why (recorded in the audit log)"),
):
    """Roll production back to the previous stable version (or an explicit one)."""
    res = _request("POST", f"/models/{model}/rollback",
                   json={"to": to, "reason": reason})
    console.print(f"[yellow]rolled back[/yellow] {model} -> {res['active_version']}")


@app.command()
def cancel(job_id: str = typer.Option(..., "--job-id", help="job to cancel")):
    """Cancel a job (best-effort: kills the running container, marks cancelled)."""
    res = _request("POST", f"/jobs/{job_id}/cancel")
    note = f" ({res['note']})" if res.get("note") else ""
    console.print(f"[yellow]{res['status']}[/yellow] {job_id}{note}")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
