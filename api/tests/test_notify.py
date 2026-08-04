"""Email notifications — pure-logic + safe-no-op tests.

The pure helpers (valid_email, *_subject/*_body, recipients_for_*) do no I/O, so
they are tested directly without an SMTP server. send_email is exercised as a
no-op (disabled) and, with smtplib.SMTP monkeypatched, as a build/send path — so
the whole flow is covered without a real relay.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # api/

import notify  # noqa: E402


# --------------------------------------------------------------------------- #
# valid_email
# --------------------------------------------------------------------------- #
def test_valid_email():
    assert notify.valid_email("a@b.com")
    assert notify.valid_email("  owner@vgreens.lan  ")   # trimmed
    assert not notify.valid_email("")
    assert not notify.valid_email(None)
    assert not notify.valid_email("nope")
    assert not notify.valid_email("a@@b.com")
    assert not notify.valid_email("a@b")                 # no dot in domain
    assert not notify.valid_email("has space@b.com")


# --------------------------------------------------------------------------- #
# job_subject / job_body
# --------------------------------------------------------------------------- #
_SUCCEEDED = {
    "job_id": "trn_20260804_ab12", "job_type": "training", "model_name": "berry",
    "dataset_name": "demo_berries", "dataset_version": "v3", "status": "succeeded",
    "result_version": "v7", "output_path": "/mnt/vgreens_storage/ai/jobs/2026/trn_20260804_ab12",
    "requested_by": "alice",
}
_FAILED = {
    "job_id": "trn_20260804_cd34", "job_type": "training", "model_name": "berry",
    "dataset_name": "demo_berries", "dataset_version": "v3", "status": "failed",
    "error": "CUDA out of memory on GPU 0", "requested_by": "alice",
}


def test_job_subject_and_body_succeeded():
    subj = notify.job_subject(_SUCCEEDED)
    assert "trn_20260804_ab12" in subj
    assert "succeeded" in subj
    assert "berry" in subj
    body = notify.job_body(_SUCCEEDED, "https://app.vgreens.lan:9443")
    assert "succeeded" in body
    assert "berry" in body
    assert "demo_berries:v3" in body
    assert "v7" in body                                        # result_version
    assert "https://app.vgreens.lan:9443/#/jobs/trn_20260804_ab12" in body


def test_job_subject_and_body_failed():
    subj = notify.job_subject(_FAILED)
    assert "failed" in subj
    assert "CUDA out of memory" in subj                        # error hint
    body = notify.job_body(_FAILED, "https://app.vgreens.lan:9443")
    assert "failed" in body
    assert "CUDA out of memory on GPU 0" in body               # full error
    assert "/#/jobs/trn_20260804_cd34" in body


# --------------------------------------------------------------------------- #
# model_subject/body + dataset_subject/body
# --------------------------------------------------------------------------- #
def test_model_subject_body():
    subj = notify.model_subject("promoted", "berry", "v7")
    assert "berry" in subj and "promoted" in subj and "v7" in subj
    body = notify.model_body("promoted", "berry", "v7", "admin",
                             "https://app.vgreens.lan:9443", reason="beats prod")
    assert "v7" in body and "admin" in body and "beats prod" in body
    assert "/#/models/berry" in body


def test_dataset_subject_body():
    imp = {"import_id": "imp_20260804_zz01", "dataset_name": "weeds",
           "source_type": "roboflow", "licence_name": "CC-BY-4.0", "requested_by": "bob"}
    subj = notify.dataset_subject(imp)
    assert "imp_20260804_zz01" in subj and "weeds" in subj and "licence" in subj.lower()
    body = notify.dataset_body(imp, "https://app.vgreens.lan:9443")
    assert "weeds" in body and "CC-BY-4.0" in body
    assert "/#/datasets/imports/imp_20260804_zz01" in body


# --------------------------------------------------------------------------- #
# recipients_for_job / recipients_for_dataset_licence
# --------------------------------------------------------------------------- #
def test_recipients_for_job_dedupe_and_drop():
    r = notify.recipients_for_job(
        {}, "alice@vgreens.lan", "owner@vgreens.lan", "ops@vgreens.lan")
    assert r == ["owner@vgreens.lan", "alice@vgreens.lan", "ops@vgreens.lan"]

    # None/empty/invalid dropped; case-insensitive dedupe keeps first-seen
    r2 = notify.recipients_for_job(
        {}, None, "Owner@vgreens.lan", "owner@VGREENS.LAN")
    assert r2 == ["Owner@vgreens.lan"]

    # no valid recipients at all -> empty list
    assert notify.recipients_for_job({}, None, "bad", "") == []


def test_recipients_for_dataset_licence():
    r = notify.recipients_for_dataset_licence(
        ["a@vgreens.lan", "b@vgreens.lan", "a@vgreens.lan"], "ops@vgreens.lan")
    assert r == ["a@vgreens.lan", "b@vgreens.lan", "ops@vgreens.lan"]
    assert notify.recipients_for_dataset_licence([], None) == []


# --------------------------------------------------------------------------- #
# send_email — safe no-op when disabled, and build/send when enabled
# --------------------------------------------------------------------------- #
def test_send_email_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("NOTIFICATIONS_ENABLED", "false")
    monkeypatch.setenv("SMTP_HOST", "smtp.vgreens.lan")   # host set, but disabled
    assert notify.send_email(["a@vgreens.lan"], "subject", "body") is False


def test_send_email_noop_when_no_host(monkeypatch):
    monkeypatch.setenv("NOTIFICATIONS_ENABLED", "true")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    assert notify.send_email(["a@vgreens.lan"], "subject", "body") is False


def test_send_email_noop_when_no_recipients(monkeypatch):
    monkeypatch.setenv("NOTIFICATIONS_ENABLED", "true")
    monkeypatch.setenv("SMTP_HOST", "smtp.vgreens.lan")
    assert notify.send_email(["", None, "bad"], "subject", "body") is False


def test_send_email_builds_and_sends_when_enabled(monkeypatch):
    monkeypatch.setenv("NOTIFICATIONS_ENABLED", "true")
    monkeypatch.setenv("SMTP_HOST", "smtp.vgreens.lan")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "svc")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")
    monkeypatch.setenv("SMTP_STARTTLS", "true")
    monkeypatch.setenv("SMTP_FROM", "vgreens-ai@vgreens.lan")

    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            sent["host"] = host
            sent["port"] = port
            sent["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            sent["starttls"] = True

        def login(self, user, pw):
            sent["login"] = (user, pw)

        def send_message(self, msg, from_addr=None, to_addrs=None):
            sent["from_addr"] = from_addr
            sent["to_addrs"] = to_addrs
            sent["subject"] = msg["Subject"]
            sent["body"] = msg.get_content()

    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)

    ok = notify.send_email(["a@vgreens.lan", "a@vgreens.lan", "b@vgreens.lan"],
                           "hello", "world")
    assert ok is True
    assert sent["host"] == "smtp.vgreens.lan"
    assert sent["port"] == 587
    assert sent["starttls"] is True
    assert sent["login"] == ("svc", "secret")
    assert sent["from_addr"] == "vgreens-ai@vgreens.lan"
    assert sent["to_addrs"] == ["a@vgreens.lan", "b@vgreens.lan"]   # deduped
    assert sent["subject"] == "hello"
    assert "world" in sent["body"]


def test_send_email_never_raises_on_smtp_error(monkeypatch):
    monkeypatch.setenv("NOTIFICATIONS_ENABLED", "true")
    monkeypatch.setenv("SMTP_HOST", "smtp.vgreens.lan")

    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(notify.smtplib, "SMTP", boom)
    assert notify.send_email(["a@vgreens.lan"], "s", "b") is False
