# strawberry_disease — training container

Fine-tunes a YOLO11n backbone on the frozen `strawberry_disease` dataset and
emits a contract-compliant result the platform can register and validate.

## Contract

| Path | Mode | Meaning |
|------|------|---------|
| `/data` | ro | staged dataset version (`train/`, `valid/`, `test/`, `data.yaml`) |
| `/config/job.yaml` | ro | hyperparameters + validation thresholds |
| `/out` | rw | `model.pt`, `eval/metrics.json`, `evaluation_report.md`, `classes.json`, `runs/` |

Exit 0 = success. The base checkpoint (`yolo11n.pt`) is baked into the image, so
the run needs no internet.

## Build (on the GPU server)

```bash
cd /opt/vgreens-ai/repo/training/strawberry_disease
docker build -t vgreens/strawberry-disease-train:v1 .
```

## Run through the platform

Put the config on the NAS so the job can reference it, then submit:

```bash
sudo mkdir -p /mnt/vgreens_storage/ai/configs
sudo cp job.yaml /mnt/vgreens_storage/ai/configs/strawberry_disease.yaml

vgreens-ai job train \
  --model strawberry_disease \
  --dataset strawberry_disease:v1 \
  --image vgreens/strawberry-disease-train:v1 \
  --config /mnt/vgreens_storage/ai/configs/strawberry_disease.yaml
```

For a quick end-to-end smoke test first, set `epochs: 3` in the config, submit,
confirm it flows queued → running → succeeded and a version is registered, then
raise epochs for the real run.

## Local sanity run (optional, outside the platform)

```bash
docker run --rm --gpus device=0 --shm-size 8g \
  -v /mnt/vgreens_storage/ai/datasets/strawberry_disease/v1:/data:ro \
  -v $PWD/job.yaml:/config/job.yaml:ro \
  -v /tmp/out:/out \
  -e JOB_ID=local -e MODEL_NAME=strawberry_disease -e DATASET_VERSION=v1 \
  vgreens/strawberry-disease-train:v1
cat /tmp/out/eval/metrics.json
```
