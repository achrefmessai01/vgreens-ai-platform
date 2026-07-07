"""docker run wrapper implementing the container contract.

Contract mounts:  /data (ro)  /config/job.yaml (ro)  /out (rw)
Contract env:     JOB_ID JOB_TYPE MODEL_NAME DATASET_VERSION TENANT
GPU isolation:    --gpus device=<GPU_ID>   (NEVER --gpus all — breaks pinning)
Always --rm       (no container litter on the disk)

stdout/stderr are teed to the NAS job log. Returns (exit_code, log_tail).
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def run_container(
    *,
    image: str,
    gpu_id: int,
    data_dir: str,
    config_path: str,
    out_dir: str,
    env: dict[str, str],
    log_path: str,
    timeout_s: int = 24 * 3600,
) -> tuple[int, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        f"--gpus", f"device={gpu_id}",
        "-v", f"{data_dir}:/data:ro",
        "-v", f"{config_path}:/config/job.yaml:ro",
        "-v", f"{out_dir}:/out",
    ]
    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]
    cmd.append(image)

    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                log.write(line)
                log.flush()
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            log.write(b"\n[runner] TIMEOUT - container killed\n")
            return 124, "container timed out"

    tail = _tail(log_path, 40)
    return proc.returncode, tail


def _tail(path: str, n: int) -> str:
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""
