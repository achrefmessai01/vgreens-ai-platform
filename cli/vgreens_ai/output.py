"""Output helpers: Rich tables for humans, clean JSON/YAML for machines.

Rules:
- table mode -> render to stdout; human messages/spinners/errors -> stderr.
- json/yaml mode -> ONLY the data on stdout (parseable); messages -> stderr.
- colour auto-disables when stdout isn't a TTY, or with --no-color / NO_COLOR.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

# messages/spinners always go to stderr so they never pollute machine output
err = Console(stderr=True)
_out = Console()


def configure(no_color: bool) -> None:
    global err, _out
    force = True if no_color else None
    err = Console(stderr=True, no_color=no_color, highlight=not no_color)
    _out = Console(no_color=no_color, highlight=not no_color)
    _ = force


STATUS_STYLE = {
    "queued": "yellow", "running": "cyan", "succeeded": "green",
    "validated": "green", "failed": "red", "rejected": "red",
    "cancelled": "dim", "alive": "green", "ready": "green",
}


def badge(status: str | None) -> str:
    s = (status or "-")
    return f"[{STATUS_STYLE.get(s, 'white')}]{s}[/]"


def fmt_ts(ts: str | None) -> str:
    if not ts:
        return "-"
    try:
        return _dt.datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return ts


def fmt_duration(start: str | None, end: str | None) -> str:
    if not start or not end:
        return "-"
    try:
        a = _dt.datetime.fromisoformat(start)
        b = _dt.datetime.fromisoformat(end)
        secs = int((b - a).total_seconds())
        m, s = divmod(secs, 60)
        return f"{m}m{s:02d}s" if m else f"{s}s"
    except (TypeError, ValueError):
        return "-"


def info(msg: str) -> None:
    err.print(msg)


def success(msg: str) -> None:
    err.print(f"[green]✓[/] {msg}")


def warn(msg: str) -> None:
    err.print(f"[yellow]![/] {msg}")


def error(msg: str) -> None:
    err.print(f"[red]error:[/] {msg}")


def emit(data: Any, output: str, *, columns: list[tuple[str, str]] | None = None,
         title: str | None = None) -> None:
    """Render `data` per the chosen output mode.

    columns: list of (header, key) for table rendering of a list[dict].
    """
    if output == "json":
        print(json.dumps(data, indent=2, default=str))
        return
    if output == "yaml":
        print(yaml.safe_dump(data, sort_keys=False, default_flow_style=False).rstrip())
        return

    # table mode
    if isinstance(data, list):
        if not data:
            err.print("[dim](no results)[/]")
            return
        cols = columns or [(k, k) for k in data[0].keys()]
        table = Table(title=title, header_style="bold")
        for header, _ in cols:
            table.add_column(header)
        for row in data:
            table.add_row(*(_cell(row.get(key), key) for _, key in cols))
        _out.print(table)
    elif isinstance(data, dict):
        table = Table(show_header=False, box=None, title=title)
        for k, v in data.items():
            table.add_row(f"[bold]{k}[/]", _cell(v, k))
        _out.print(table)
    else:
        _out.print(str(data))


_TS_KEYS = {"created_at", "started_at", "finished_at", "ts", "last_seen", "updated_at", "acquired_at"}


def _cell(value: Any, key: str) -> str:
    if value is None:
        return "-"
    if key == "status":
        return badge(str(value))
    if key in _TS_KEYS:
        return fmt_ts(str(value))
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


def is_tty() -> bool:
    return sys.stdout.isatty()
