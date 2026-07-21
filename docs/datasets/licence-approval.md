# Licence approval (human gate)

External datasets (Roboflow, HTTPS, S3) must have their licence approved by a
human before they are frozen. Internal local/NAS data is treated as owned
(`licence_status: not_required`) and skips this gate.

## Flow

1. An external import runs download → quarantine → audit, then parks at
   `awaiting_license_approval`.
2. An admin reviews the source + licence and approves or rejects.
3. **Approve** records the approver + timestamp, sets `licence_status: approved`,
   and re-enqueues the import — it resumes and freezes.
4. **Reject** marks the import `rejected` (terminal).

## CLI

```bash
vgreens-ai dataset approve-license <import-id> --reason "public dataset, OK for internal use" --licence-name "CC BY 4.0"
vgreens-ai dataset reject-license  <import-id> --reason "licence unclear"
```

## Web

Open the import in Datasets → the workflow page shows an **Approve / Reject**
panel while it is awaiting. The action and actor are recorded in the audit log.

## Notes

- Approval/rejection is an admin action (enforced by the API once RBAC lands).
- Every decision is written to `audit_events` (`dataset.license.approve` /
  `dataset.license.reject`) with the actor and reason.
- Never store raw licence keys or signed URLs; only the licence name/status.
