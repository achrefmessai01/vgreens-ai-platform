"""CLI tests for the dataset command group.

These exercise the command wiring and request shaping with a fake API client —
no network. They assert the CLI hits the right endpoints with the right bodies
and never bypasses the API.
"""
from typer.testing import CliRunner

from vgreens_ai import cli

runner = CliRunner()


class FakeClient:
    def __init__(self):
        self.calls = []

    def get(self, path, params=None, spinner=None):
        self.calls.append(("GET", path, params))
        if path == "/datasets":
            return {"datasets": [{"dataset_name": "d", "versions": 1, "latest_frozen_at": "t"}]}
        if path == "/datasets/imports":
            return [{"import_id": "imp1", "dataset_name": "d", "source_type": "local",
                     "state": "frozen", "created_at": "t"}]
        if path.startswith("/datasets/imports/"):
            return {"import_id": "imp1", "state": "frozen", "dataset_name": "d",
                    "resolved_version": "v1", "events": []}
        if path.endswith("/versions"):
            return [{"version": "v1", "task": "object_detection", "image_count": 10,
                     "class_count": 2, "licence_status": "approved",
                     "content_digest": "abc", "frozen_at": "t"}]
        return []

    def post(self, path, json=None, spinner=None):
        self.calls.append(("POST", path, json))
        if "backfill" in path:
            return {"dry_run": "dry_run=true" in path, "scanned": 3, "already_indexed": 1,
                    "added": [{"dataset_name": "legacy_ds", "version": "v1",
                               "image_count": 5, "class_count": 2, "metadata_source": "inferred"}],
                    "errors": []}
        return {"import_id": "imp1", "state": "submitted"}


def _patch(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(cli, "_client", lambda ctx: fake)
    return fake


def test_import_local_shapes_request(monkeypatch):
    fake = _patch(monkeypatch)
    r = runner.invoke(cli.app, ["dataset", "import", "local", "--path", "/mnt/x", "--name", "d"])
    assert r.exit_code == 0, r.output
    post = next(c for c in fake.calls if c[0] == "POST")
    assert post[1] == "/datasets/imports"
    assert post[2]["source_type"] == "local"
    assert post[2]["dataset_name"] == "d"
    assert post[2]["source"] == {"path": "/mnt/x"}
    # None options are stripped, not sent
    assert "requested_version" not in post[2]


def test_import_roboflow_shapes_request(monkeypatch):
    fake = _patch(monkeypatch)
    r = runner.invoke(cli.app, ["dataset", "import", "roboflow", "--workspace", "w",
                                "--project", "p", "--source-version", "1", "--name", "d"])
    assert r.exit_code == 0, r.output
    post = next(c for c in fake.calls if c[0] == "POST")
    assert post[2]["source_type"] == "roboflow"
    assert post[2]["source"] == {"workspace": "w", "project": "p", "rf_version": 1}


def test_list_json_output(monkeypatch):
    _patch(monkeypatch)
    r = runner.invoke(cli.app, ["dataset", "list", "-o", "json"])
    assert r.exit_code == 0, r.output
    assert '"dataset_name"' in r.stdout


def test_versions_hits_endpoint(monkeypatch):
    fake = _patch(monkeypatch)
    r = runner.invoke(cli.app, ["dataset", "versions", "d", "-o", "json"])
    assert r.exit_code == 0, r.output
    assert ("GET", "/datasets/d/versions", None) in fake.calls


def test_approve_license_posts(monkeypatch):
    fake = _patch(monkeypatch)
    r = runner.invoke(cli.app, ["dataset", "approve-license", "imp1", "--reason", "ok"])
    assert r.exit_code == 0, r.output
    post = next(c for c in fake.calls if c[0] == "POST")
    assert post[1] == "/datasets/imports/imp1/approve-license"
    assert post[2]["reason"] == "ok"


def test_backfill_dry_run_by_default(monkeypatch):
    fake = _patch(monkeypatch)
    r = runner.invoke(cli.app, ["dataset", "backfill"])
    assert r.exit_code == 0, r.output
    post = next(c for c in fake.calls if c[0] == "POST")
    assert post[1] == "/datasets/backfill?dry_run=true"


def test_backfill_apply(monkeypatch):
    fake = _patch(monkeypatch)
    r = runner.invoke(cli.app, ["dataset", "backfill", "--apply"])
    assert r.exit_code == 0, r.output
    post = next(c for c in fake.calls if c[0] == "POST")
    assert post[1] == "/datasets/backfill?dry_run=false"
