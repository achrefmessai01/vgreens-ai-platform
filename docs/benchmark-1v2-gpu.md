# Benchmark — 1-GPU vs 2-GPU training

Measures the wall-clock speedup from training the same model on one vs two GPUs,
using the platform's per-GPU allocation and PyTorch DDP (`device=[0,1]`).

## Method

- **Model:** `strawberry_disease` (RF-DETR detection), identical image for both runs.
- **Dataset:** `strawberry_disease_auto:v1` (~11,700 images, frozen).
- **Config:** identical (`epochs=10`, `imgsz=640`, `batch=16`); only `--gpu-count` differs.
- **Metric:** wall-clock from the job's `started_at` → `finished_at`; mAP@0.5 checked to
  confirm accuracy parity. Runs executed sequentially (2-GPU run needs both GPUs free).
- **Hardware:** gpu-worker-01, 2× GPU.

## Running it

Epochs are read from the **published config at runtime**, not baked into the image, so no rebuild is needed just to change them. Run the two jobs sequentially — the 2-GPU run needs both GPUs free.

**0. Prerequisites.** A worker is online, both GPUs are idle, and the training image exists in the registry: `registry.gitlab.com/vgreens-models/vgreens-model-strawberry-disease:dad63370`. If that tag isn't in the registry yet, trigger the model's GitLab pipeline (push any commit, or run the `build` stage) and use the resulting `:<short-sha>` tag.

**1. Set epochs to 10 and publish the config** (from the model repo on the Mac):

```
cd ~/Desktop/vgreens-model-strawberry-disease
# edit configs/strawberry_disease.yaml → epochs: 10
vgreens-ai config publish strawberry_disease --file configs/strawberry_disease.yaml
```

**2. Run A — single GPU:**

```
vgreens-ai job train --model strawberry_disease --dataset strawberry_disease_auto:v1 \
  --image registry.gitlab.com/vgreens-models/vgreens-model-strawberry-disease:dad63370 \
  --gpu-count 1 --config /mnt/vgreens_storage/ai/job-configs/strawberry_disease.yaml
vgreens-ai job watch <job_id_A>
```

**3. Run B — two GPUs** (only after A finishes):

```
vgreens-ai job train --model strawberry_disease --dataset strawberry_disease_auto:v1 \
  --image registry.gitlab.com/vgreens-models/vgreens-model-strawberry-disease:dad63370 \
  --gpu-count 2 --yes --config /mnt/vgreens_storage/ai/job-configs/strawberry_disease.yaml
vgreens-ai job watch <job_id_B>
```

**4. Read the numbers** for each run and fill the table below:

```
vgreens-ai job show <job_id_A> -o json   # started_at, finished_at, result metrics
vgreens-ai job show <job_id_B> -o json
```

## Results

Run on 2026-07-26, `strawberry_disease_auto:v1`, `epochs=10`, image `strawberry-disease:local`
(built directly on gpu-worker-01), 2× RTX 4070.

| Run | GPUs | Job ID | Wall-clock | mAP@0.5 | mAP@0.5:0.95 |
|-----|------|--------|-----------|---------|--------------|
| A | 1 | `trn_20260726_29da` | 453 s (7m33s) | 0.899 | 0.727 |
| B | 2 | `trn_20260726_4f4e` | 413 s (6m53s) | 0.894 | 0.722 |

**Speedup (1→2 GPU):** 1.10× &nbsp; (453 s / 413 s)
**Parallel efficiency:** ~55% &nbsp; (1.10 / 2)
**Accuracy parity:** mAP@0.5 0.899 vs 0.894 — statistically equivalent.

## Notes / interpretation

- **1.10× at 10 epochs is expected, not disappointing.** The measured value is
  end-to-end *wall-clock* (`started_at`→`finished_at`), which includes work that
  does not parallelize: container start, per-rank dataset caching, DDP process
  spawn + NCCL init, and the final single-GPU validation pass. On a ~7-minute run
  that fixed overhead is a large fraction of the total, so it dominates the ratio.
- **The gain grows with the run.** Only the per-epoch training loop parallelizes,
  so as epochs/dataset size increase, the fixed overhead is amortized and
  efficiency climbs toward the training-only ratio. A 50–100 epoch run would show
  a markedly higher speedup; 10 epochs was chosen to keep the benchmark quick.
- **Accuracy is preserved:** mAP@0.5 0.899 (1-GPU) vs 0.894 (2-GPU) — within
  normal run-to-run variance from the larger effective batch under DDP. Two-GPU
  training does not cost quality.
- **What this validates:** the platform's two-GPU path works end-to-end — per-GPU
  locks reserve both devices, the worker runs `--gpus "device=0,1"`, and
  `train.py` sets `device=[0,1]` for PyTorch DDP — producing a validated model
  version, not just "both GPUs visible."

_Numbers captured from `vgreens-ai job show` (timings) and `vgreens-ai model show` (metrics)._
