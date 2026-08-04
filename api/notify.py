"""Email notifications — fire-and-forget alerts for the vGreens Job API.

Sends plain-text email when:
  * a training/processing job finishes (succeeded/failed/cancelled),
  * a model is promoted or rolled back,
  * a dataset import needs a human licence-approval decision.

Design rules (match the rest of the API):
  * stdlib only — smtplib/email, no new heavy deps;
  * config is read from the environment AT SEND TIME (not import), so a systemd
    EnvironmentFile change takes effect without re-importing the module;
  * everything is best-effort: a misconfigured SMTP server or a network blip
    logs and returns False, it NEVER raises into the request path;
  * the pure helpers (valid_email, *_subject/*_body, recipients_for_*) do no
    I/O so they can be unit-tested without an SMTP server.

--------------------------------------------------------------------------- #
CONFIGURATION (environment variables)
--------------------------------------------------------------------------- #
These belong in /opt/vgreens-ai/.env (the systemd EnvironmentFile) — do NOT
commit real secrets to the repo.

  NOTIFICATIONS_ENABLED   "true" to send; anything else -> no-op (default "false")
  SMTP_HOST               SMTP relay host; unset -> no-op even if enabled
  SMTP_PORT               SMTP port (default "587")
  SMTP_USERNAME           SMTP login user (optional; login skipped if unset)
  SMTP_PASSWORD           SMTP login password (optional; NEVER logged)
  SMTP_FROM               From: address (default "vgreens-ai@localhost")
  SMTP_STARTTLS           "true" to issue STARTTLS before login (default "true")
  NOTIFY_DEFAULT          optional fallback/cc address (used when no other
                          recipient resolves, and always cc'd if set)
  WEB_BASE_URL            base URL for deep-links in emails
                          (default "https://app.vgreens.lan:9443")
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

log = logging.getLogger("vgreens.notify")

_SMTP_TIMEOUT = 10.0  # seconds — never hang the request path on a dead relay


# --------------------------------------------------------------------------- #
# config (read at send time, not import)
# --------------------------------------------------------------------------- #
def _env_true(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() == "true"


def web_base_url() -> str:
    return os.environ.get("WEB_BASE_URL", "https://app.vgreens.lan:9443").rstrip("/")


def _enabled() -> bool:
    """Notifications only fire when explicitly enabled AND a relay is set."""
    return _env_true("NOTIFICATIONS_ENABLED") and bool(os.environ.get("SMTP_HOST"))


# --------------------------------------------------------------------------- #
# pure helpers (no I/O — unit-testable)
# --------------------------------------------------------------------------- #
def valid_email(s) -> bool:
    """Basic sanity check: a non-empty string with a single '@' between two
    non-empty, whitespace-free parts. Deliberately permissive (we relay to an
    internal SMTP server, not validate RFC 5322)."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s or s.count("@") != 1:
        return False
    local, _, domain = s.partition("@")
    if not local or not domain or "." not in domain:
        return False
    return not any(c.isspace() for c in s)


def _clean_recipients(to) -> list[str]:
    """Dedupe (case-insensitively), strip, drop empties/invalid — preserving
    first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in to or []:
        if not isinstance(raw, str):
            continue
        addr = raw.strip()
        if not valid_email(addr):
            continue
        key = addr.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(addr)
    return out


def _metric_hint(job: dict) -> str:
    """A short subject-line hint: the error for a failed job, else a result
    version if we have one."""
    status = job.get("status")
    if status == "failed" and job.get("error"):
        err = str(job["error"]).splitlines()[0]
        return err[:80]
    if job.get("result_version"):
        return f"result {job['result_version']}"
    return ""


def job_subject(job: dict) -> str:
    jid = job.get("job_id", "?")
    model = job.get("model_name", "?")
    status = job.get("status", "?")
    hint = _metric_hint(job)
    tail = f" — {hint}" if hint else ""
    return f"[vGreens AI] {job.get('job_type', 'job')} job {jid} {status} ({model}){tail}"


def job_body(job: dict, base_url: str) -> str:
    jid = job.get("job_id", "?")
    ds = job.get("dataset_name")
    dsv = job.get("dataset_version")
    dataset = f"{ds}:{dsv}" if ds else "(none)"
    lines = [
        f"Job {jid} is now {job.get('status', 'unknown')}.",
        "",
        f"Type:           {job.get('job_type', '?')}",
        f"Model:          {job.get('model_name', '?')}",
        f"Dataset:        {dataset}",
        f"Requested by:   {job.get('requested_by') or '(unknown)'}",
        f"Status:         {job.get('status', '?')}",
    ]
    if job.get("status") == "failed" and job.get("error"):
        lines.append(f"Error:          {job['error']}")
    if job.get("result_version"):
        lines.append(f"Result version: {job['result_version']}")
    if job.get("output_path"):
        lines.append(f"Output path:    {job['output_path']}")
    lines += [
        "",
        f"Details: {base_url.rstrip('/')}/#/jobs/{jid}",
        "",
        "— vGreens AI Platform",
    ]
    return "\n".join(lines)


def model_subject(action: str, model: str, version) -> str:
    ver = version or "?"
    return f"[vGreens AI] model {model} {action} -> {ver}"


def model_body(action: str, model: str, version, by: str, base_url: str,
               reason: str | None = None) -> str:
    lines = [
        f"Model '{model}' was {action}.",
        "",
        f"Active version: {version or '(unknown)'}",
        f"Action by:      {by or '(unknown)'}",
    ]
    if reason:
        lines.append(f"Reason:         {reason}")
    lines += [
        "",
        f"Details: {base_url.rstrip('/')}/#/models/{model}",
        "",
        "— vGreens AI Platform",
    ]
    return "\n".join(lines)


def dataset_subject(imp: dict) -> str:
    name = imp.get("dataset_name", "?")
    iid = imp.get("import_id", "?")
    return f"[vGreens AI] dataset import {iid} needs licence approval ({name})"


def dataset_body(imp: dict, base_url: str) -> str:
    iid = imp.get("import_id", "?")
    lines = [
        "A dataset import needs a licence-approval decision before it can be frozen.",
        "",
        f"Import ID:      {iid}",
        f"Dataset:        {imp.get('dataset_name', '?')}",
        f"Source type:    {imp.get('source_type', '?')}",
        f"Licence:        {imp.get('licence_name') or '(unknown)'}",
        f"Requested by:   {imp.get('requested_by') or '(unknown)'}",
        "",
        f"Review it here: {base_url.rstrip('/')}/#/datasets/imports/{iid}",
        "",
        "— vGreens AI Platform",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# recipient resolution (pure — callers fetch the data and pass it in)
# --------------------------------------------------------------------------- #
def recipients_for_job(job: dict, requester_email: str | None,
                       owner_email: str | None, default: str | None = None) -> list[str]:
    """Who to email about a finished job: the model owner + the job requester,
    plus the fallback/cc address if configured. Invalid/None/dupes dropped."""
    return _clean_recipients([owner_email, requester_email, default])


def recipients_for_dataset_licence(admin_emails, default: str | None = None) -> list[str]:
    """Who to email about a licence gate: every active admin with an email,
    plus the fallback/cc address if configured."""
    return _clean_recipients(list(admin_emails or []) + [default])


# --------------------------------------------------------------------------- #
# the one impure entry point (SMTP). Wrapped so it can NEVER raise.
# --------------------------------------------------------------------------- #
def send_email(to: list[str], subject: str, body: str) -> bool:
    """Send a plain-text email best-effort. Returns True only if a message was
    actually handed to the SMTP server. A disabled notifier, no recipients, or
    any failure logs and returns False — this never raises."""
    try:
        if not _enabled():
            log.info("notifications disabled or SMTP_HOST unset — skipping email %r", subject)
            return False

        recipients = _clean_recipients(to)
        if not recipients:
            log.info("no valid recipients for email %r — skipping", subject)
            return False

        host = os.environ["SMTP_HOST"]
        port = int(os.environ.get("SMTP_PORT", "587"))
        username = os.environ.get("SMTP_USERNAME")
        password = os.environ.get("SMTP_PASSWORD")
        sender = os.environ.get("SMTP_FROM", "vgreens-ai@localhost")
        use_starttls = _env_true("SMTP_STARTTLS", "true")

        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(host, port, timeout=_SMTP_TIMEOUT) as smtp:
            if use_starttls:
                smtp.starttls()
            if username and password:
                smtp.login(username, password)   # password is never logged
            smtp.send_message(msg, from_addr=sender, to_addrs=recipients)

        log.info("sent email %r to %d recipient(s)", subject, len(recipients))
        return True
    except Exception as exc:  # noqa: BLE001 - notifications must never break the caller
        log.warning("failed to send email %r: %s", subject, exc)
        return False
