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


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
