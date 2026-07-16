# vgreens-ai CLI reference

A client over the Job API. All actions are HTTP calls; nothing runs locally.

## Setup

```bash
pipx install .            # from the cli/ directory  (or: pip install -e cli/)
vgreens-ai config init    # guided: URL, token (hidden), validates, saves ~/.vgreens-ai.toml (600)
vgreens-ai system doctor  # verify connectivity + auth from your machine
vgreens-ai --install-completion   # shell completion
```

## Global options

`-p/--profile` · `--api-url` · `--token` · `-o/--output table|json|yaml` ·
`--no-color` · `-q/--quiet` · `-v/--verbose` · `--timeout` · `--version`

Colour auto-disables when output is piped. In `json`/`yaml` mode only data is on
stdout (messages go to stderr) — safe for scripting.

## Commands

### config
- `config init [--profile P] [--api-url U] [--token T]` — guided setup, secure save.
- `config show` — list profiles (tokens redacted).
- `config profiles` — profile names (`*` = default).
- `config use <profile>` — set the default profile.

### job
- `job train --model M --dataset name:ver --image IMG [--tenant T] [--config PATH]`
- `job process --model M --image IMG [--dataset name:ver] [--tenant T] [--config PATH]`
- `job list [--status S] [--type training|processing] [--model M] [--limit N]`
- `job show <job-id>` · `job events <job-id>` · `job watch <job-id>` · `job cancel <job-id>`

### model
- `model list` · `model show <model>` · `model versions <model>`
- `model compare <model> <ver-a> <ver-b>` — metric diff
- `model promote <model> <version> [--reason R]`
- `model rollback <model> [--to VERSION] [--reason R]`

### worker / gpu / system
- `worker list` · `gpu list`
- `system health` · `system version` · `system doctor`

### Backward-compatible aliases (hidden)
`train`, `process`, `status --job-id`, `list-jobs`, `cancel --job-id`, `models`,
`promote`, `rollback` still work as before.

## Examples

```bash
vgreens-ai job list --status failed -o json | jq '.[].job_id'
vgreens-ai job watch trn_20260716_8498
vgreens-ai model show strawberry_detection
vgreens-ai model promote strawberry_detection 2026-07-16_002 --reason "beats baseline"
vgreens-ai --profile prod worker list
```

See `exit-codes.md` for scripting.
