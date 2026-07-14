"""Dummy GPU job — exercises the full training contract incl. registry/validation.

Reads /data (optional), does a short GPU burn, then writes to /out exactly what a
real training would:
  model.pt                 the "model" (a saved tensor stands in for weights)
  eval/metrics.json        achieved metrics (mAP50 / precision / recall)
  evaluation_report.md     human-readable summary
Exit 0 on success.

To make validation demonstrable without real training, the *achieved* mAP50 is
read from the job config at /config/job.yaml:

  simulate:
    mAP50: 0.90        # what this "model" scores; default 0.90 if absent

The platform (worker) compares that against the config's `thresholds:` block and
marks the registry version validated or rejected. The job itself knows nothing
about thresholds — it just reports its metrics, exactly like a real model.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

DATA = Path("/data")
OUT = Path("/out")
CONFIG = Path("/config/job.yaml")
BURN_SECONDS = int(os.environ.get("BURN_SECONDS", "15"))


def _simulated_map() -> float:
    if yaml is not None and CONFIG.is_file():
        try:
            cfg = yaml.safe_load(CONFIG.read_text()) or {}
            sim = (cfg.get("simulate") or {}).get("mAP50")
            if sim is not None:
                return float(sim)
        except Exception:
            pass
    return 0.90


def main() -> int:
    (OUT / "eval").mkdir(parents=True, exist_ok=True)

    job_id = os.environ.get("JOB_ID", "unknown")
    model_name = os.environ.get("MODEL_NAME", "dummy")

    input_file = DATA / "input.txt"
    raw = input_file.read_bytes() if input_file.exists() else b""
    checksum = hashlib.sha256(raw).hexdigest()

    if not torch.cuda.is_available():
        sys.stderr.write("CUDA not available inside container\n")
        return 1

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[dummy] job={job_id} model={model_name} gpu={gpu_name}", flush=True)

    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    start = time.time()
    iters = 0
    while time.time() - start < BURN_SECONDS:
        c = a @ b
        a = c / (c.abs().mean() + 1e-6)
        iters += 1
    torch.cuda.synchronize()
    elapsed = round(time.time() - start, 2)

    # "model" artifact (a real file so the registry has something to store)
    torch.save({"weights": a.detach().cpu()[:8, :8], "trained_iters": iters},
               OUT / "model.pt")

    # achieved metrics — mAP50 is simulated from the job config; the rest track it
    map50 = _simulated_map()
    metrics = {
        "mAP50": round(map50, 4),
        "precision": round(min(map50 + 0.02, 0.999), 4),
        "recall": round(max(map50 - 0.03, 0.0), 4),
    }
    (OUT / "eval" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    (OUT / "evaluation_report.md").write_text(
        f"# Evaluation report\n\n"
        f"- job: `{job_id}`\n"
        f"- model: {model_name}\n"
        f"- GPU: {gpu_name}\n"
        f"- input sha256: `{checksum}`\n"
        f"- GPU burn: {iters} matmuls in {elapsed}s\n\n"
        f"## Metrics\n\n"
        f"- mAP@0.5: {metrics['mAP50']}\n"
        f"- precision: {metrics['precision']}\n"
        f"- recall: {metrics['recall']}\n"
    )
    print(f"[dummy] done: mAP50={metrics['mAP50']} ({iters} matmuls, {elapsed}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
