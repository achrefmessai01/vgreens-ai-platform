"""Two-GPU diagnostic. Reports the GPUs visible inside the container and does a
small matmul on each. Contract-compliant so it runs through the platform as a
training job. Use with gpu_count=2 to verify dual-GPU allocation + exposure.
"""
import json
import os
import sys
from pathlib import Path

import torch

OUT = Path("/out")


def main() -> int:
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(f"[gpu-diag] GPU_COUNT env={os.environ.get('GPU_COUNT')} visible={n} names={names}",
          flush=True)
    for i in range(n):
        a = torch.randn(2048, 2048, device=f"cuda:{i}")
        _ = a @ a
        torch.cuda.synchronize(i)
        print(f"[gpu-diag] matmul OK on cuda:{i} ({names[i]})", flush=True)
    (OUT / "model.pt").write_text(json.dumps({"visible_gpus": n, "names": names}))
    (OUT / "eval" / "metrics.json").write_text(json.dumps({"visible_gpus": n}))
    print(f"[gpu-diag] done — {n} GPU(s) visible and usable", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
