"""Dummy GPU job — proves the container contract end to end.

Reads /data/input.txt, runs ~60s of torch.matmul on cuda:0, writes:
  /out/result.json  (input checksum, elapsed, GPU name, matmul iterations)
  /out/report.md    (human-readable summary)
Exit 0 on success.

Knows NOTHING about the platform — only the contract (mounts + env). That is the
whole point: the platform is model-agnostic.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

DATA = Path("/data")
OUT = Path("/out")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    job_id = os.environ.get("JOB_ID", "unknown")
    input_file = DATA / "input.txt"
    raw = input_file.read_bytes() if input_file.exists() else b""
    checksum = hashlib.sha256(raw).hexdigest()

    if not torch.cuda.is_available():
        sys.stderr.write("CUDA not available inside container\n")
        return 1

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[dummy] job={job_id} gpu={gpu_name}", flush=True)

    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)

    start = time.time()
    iters = 0
    while time.time() - start < 60:
        c = a @ b
        a = c / (c.abs().mean() + 1e-6)  # keep values bounded
        iters += 1
    torch.cuda.synchronize()
    elapsed = round(time.time() - start, 2)

    result = {
        "job_id": job_id,
        "input_sha256": checksum,
        "input_bytes": len(raw),
        "gpu": gpu_name,
        "matmul_iterations": iters,
        "elapsed_seconds": elapsed,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    (OUT / "report.md").write_text(
        f"# Dummy job report\n\n"
        f"- job: `{job_id}`\n"
        f"- GPU: {gpu_name}\n"
        f"- input sha256: `{checksum}`\n"
        f"- matmul iterations: {iters}\n"
        f"- elapsed: {elapsed}s\n"
    )
    print(f"[dummy] done: {iters} iterations in {elapsed}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
