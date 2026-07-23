# Benchmark — 1-GPU vs 2-GPU training

Measures the wall-clock speedup from training the same model on one vs two GPUs,
using the platform's per-GPU allocation and ultralytics DDP (`device=[0,1]`).

## Method

- **Model:** `strawberry_disease` (YOLO11n detection), identical image for both runs.
- **Dataset:** `strawberry_disease_auto:v1` (~11,700 images, frozen).
- **Config:** identical (`epochs=10`, `imgsz=640`, `batch=16`); only `--gpu-count` differs.
- **Metric:** wall-clock from the job's `started_at` → `finished_at`; mAP@0.5 checked to
  confirm accuracy parity. Runs executed sequentially (2-GPU run needs both GPUs free).
- **Hardware:** gpu-worker-01, 2× GPU.

## Results

| Run | GPUs | Job ID | Wall-clock | mAP@0.5 |
|-----|------|--------|-----------|---------|
| A | 1 | `TBD` | `TBD` | `TBD` |
| B | 2 | `TBD` | `TBD` | `TBD` |

**Speedup (1→2 GPU):** `TBD` × &nbsp; (= time_A / time_B)
**Parallel efficiency:** `TBD` % &nbsp; (= speedup / 2)

## Notes / interpretation

- DDP has fixed startup overhead (process spawn, gradient sync), so speedup is
  below the ideal 2× — expected, especially at low epoch counts. Larger
  datasets/epochs amortize the overhead and push efficiency higher.
- Accuracy should be comparable across the two runs; small variation is normal
  (different effective batch/seed under DDP).
- This validates that the platform's two-GPU path (per-GPU locks + `--gpus
  "device=0,1"` + `device=[0,1]` in train.py) delivers a real, measured speedup —
  not just that both GPUs are visible.

_Fill A/B from `vgreens-ai job show <job_id>` (started_at, finished_at, metrics)._
